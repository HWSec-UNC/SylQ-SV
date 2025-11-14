# Main execution engine that orchestrates symbolic execution of SystemVerilog designs

import z3
from z3 import Solver, Int, BitVec, Context, BitVecSort, ExprRef, BitVecRef, If, BitVecVal, And
from .execution_manager import ExecutionManager
from .symbolic_state import SymbolicState
from .cfg import CFG
import re
import os
from optparse import OptionParser
from typing import Optional
import random, string
import time
import gc
from itertools import product
import logging
from helpers.utils import to_binary
import sys
from copy import deepcopy
import pyslang as ps
from helpers.slang_helpers import get_module_name, init_state

# Tuple of PySlang AST node types that represent conditional/loop statements
CONDITIONALS = (
    ps.ConditionalStatementSyntax,
    ps.CaseStatementSyntax,
    ps.ForeachLoopStatementSyntax,
    ps.ForLoopStatementSyntax,
    ps.LoopStatementSyntax,
    ps.DoWhileStatementSyntax
)
class ExecutionEngine:
    # Drives the entire symbolic execution process
    module_depth: int = 0 # Tracks current module nesting depth during execution
    debug: bool = False # Boolean flag to enable debug output
    done: bool = False # Boolean flag indicating if execution is complete

    def check_pc_SAT(self, s: Solver, constraint: ExprRef) -> bool:
        """Check if pc is satisfiable before taking path."""
        # the push adds a backtracking point if unsat
        s.push()
        s.add(constraint)
        result = s.check()
        if str(result) == "sat":
            return True
        else:
            s.pop()
            return False

    def check_dup(self, m: ExecutionManager) -> bool:
        """Checks if the current path is a duplicate/worth exploring."""
        for i in range(len(m.path_code)):
            if m.path_code[i] == "1" and i in m.completed:
                return True
        return False

    def solve_pc(self, s: Solver) -> bool:
        """Solves path condition using Z3"""
        result = str(s.check())
        if str(result) == "sat":
            model = s.model()
            return True
        else:
            return False

    def seen_all_cases(self, m: ExecutionManager, bit_index: int, nested_ifs: int) -> bool:
        """Checks if we've seen all the cases for this index in the bit string.
        We know there are no more nested conditionals within the block, just want to check 
        that we have seen the path where this bit was turned on but the thing to the left of it
        could vary."""
        # first check if things less than me have been added.
        # so index 29 shouldnt be completed before 30
        for i in range(bit_index + 1, 32):
            if not i in m.completed:
                return False
        count = 0
        seen = m.seen
        for path in seen[m.curr_module]:
            if path[bit_index] == '1':
                count += 1
        if count >  2 * nested_ifs:
            return True
        return False

    def module_count_sv(self, m: ExecutionManager, items) -> None:
        """Traverse a top level SystemVerilog module (pyslang AST) and count instances.

        This implementation uses duck-typing and classname checks so it is robust
        across pyslang node variants. It attempts to find instantiation nodes
        and increment m.instance_count[module_name].
        """
        if items is None:
            return

        # If it's a plain list/tuple of nodes, recurse over each element
        if isinstance(items, (list, tuple)):
            for it in items:
                self.module_count_sv(m, it)
            return

        # Normalize access: many pyslang nodes wrap a single statement under .statement
        # e.g., ProceduralBlockSyntax -> .statement; handle that first.
        cname = items.__class__.__name__ if hasattr(items, '__class__') else ''
        if cname == "ProceduralBlockSyntax" and hasattr(items, 'statement'):
            self.module_count_sv(m, items.statement)
            return

        # If the node exposes an `instances` collection (common for instantiation lists),
        # traverse it first so nested instance lists are handled.
        if hasattr(items, 'instances'):
            self.module_count_sv(m, items.instances)

        # Heuristic: if the class name suggests an instantiation/instance, try to extract module name
        lower_name = cname.lower()
        if 'instance' in lower_name or 'instantiat' in lower_name or 'moduleinst' in lower_name:
            # Try a set of common attribute names that may hold the referenced module name/object
            mod_name = None
            for attr in ('module', 'module_name', 'moduleName', 'module_identifier',
                         'moduleReference', 'module_ref', 'moduleIdentifier', 'moduleType',
                         'type'):
                if hasattr(items, attr):
                    val = getattr(items, attr)
                    if val is None:
                        continue
                    if isinstance(val, str):
                        mod_name = val
                    else:
                        # attempt to extract a name from an identifier node
                        mod_name = getattr(val, 'name', None) or getattr(val, 'identifier', None) or str(val)
                    break

            # If we couldn't find a direct attribute, some pyslang instantiation nodes
            # keep the module reference under a nested template like `.module` or `.moduleName`
            if not mod_name:
                # inspect all attributes for something that looks like a module identifier
                for a in dir(items):
                    if 'module' in a.lower() or 'instance' in a.lower():
                        val = getattr(items, a)
                        if isinstance(val, str):
                            mod_name = val
                            break
                        if hasattr(val, 'name'):
                            mod_name = getattr(val, 'name')
                            break

            if mod_name:
                m.instance_count[mod_name] = m.instance_count.get(mod_name, 0) + 1

            # If the instantiation node also contains nested children, traverse them
            for child_attr in ('items', 'statements', 'statement', 'instances', 'children', 'body'):
                if hasattr(items, child_attr):
                    self.module_count_sv(m, getattr(items, child_attr))
            return

        # Otherwise, descend into common container attributes to find nested instantiations
        for attr in ('items', 'statements', 'body', 'statement', 'declarationList', 'declarations'):
            if hasattr(items, attr):
                child = getattr(items, attr)
                if child is not None:
                    self.module_count_sv(m, child)



    def populate_child_paths(self, manager: ExecutionManager) -> None:
        """Populates child path codes based on number of paths."""
        for child in manager.child_num_paths:
            manager.child_path_codes[child] = []
            if manager.piece_wise:
                manager.child_path_codes[child] = []
                for i in manager.child_range:
                    manager.child_path_codes[child].append(to_binary(i))
            else:
                for i in range(manager.child_num_paths[child]):
                    manager.child_path_codes[child].append(to_binary(i))

    def populate_seen_mod(self, manager: ExecutionManager) -> None:
        """Populates child path codes but in a format to keep track of corresponding states that we've seen."""
        for child in manager.child_num_paths:
            manager.seen_mod[child] = {}
            if manager.piece_wise:
                for i in manager.child_range:
                    manager.seen_mod[child][(to_binary(i))] = {}
            else:
                for i in range(manager.child_num_paths[child]):
                    manager.seen_mod[child][(to_binary(i))] = {}

    def execute_sv(self, visitor, modules, manager: Optional[ExecutionManager], num_cycles: int) -> None:
        """Main entry point for PySlang execution
        Drives symbolic execution for SystemVerilog designs."""
        gc.collect()
        print(f"Executing for {num_cycles} clock cycles")
        self.module_depth += 1
        state: SymbolicState = SymbolicState()
        if manager is None:
            manager: ExecutionManager = ExecutionManager()
            manager.cache = self.cache
            manager.sv = True
            modules_dict = {}
            # a dictionary keyed by module name, that gives the list of cfgs
            cfgs_by_module = {}
            cfg_count_by_module = {}
            for module in modules:
                sv_module_name = get_module_name(module)
                #print(sv_module_name)
                #modules_dict[sv_module_name] = sv_module_name
                modules_dict[sv_module_name] = module
                always_blocks_by_module = {sv_module_name: []}
                manager.seen_mod[sv_module_name] = {}
                cfgs_by_module[sv_module_name] = []
                sub_manager = ExecutionManager()
                # If module is DefinitionSymbol, get the syntax tree
                module_syntax = module.syntax if isinstance(module, ps.DefinitionSymbol) else module
                sub_manager.init_run(sub_manager, module_syntax)
                self.module_count_sv(manager, module) 
                if sv_module_name in manager.instance_count:
                    print(f"Module {sv_module_name} has {manager.instance_count[sv_module_name]} instances")
                    manager.instances_seen[sv_module_name] = 0
                    manager.instances_loc[sv_module_name] = ""
                    num_instances = manager.instance_count[sv_module_name]
                    #cfgs_by_module.pop(sv_module_name, None)
                    cfgs_by_module.pop(sv_module_name, None)
                    for i in range(num_instances):
                        instance_name = f"{sv_module_name}_{i}"
                        manager.names_list.append(instance_name)
                        cfgs_by_module[instance_name] = []

                         # 1) discover always blocks once
                        probe = CFG()
                        probe.get_always_sv(manager, state, module)

                        # 2) build a fresh CFG per always block (SV walker)
                        for ab in probe.always_blocks:
                            ab_body = getattr(ab, "statement", getattr(ab, "members", ab))
                            c = CFG()
                            c.module_name = instance_name
                            c.basic_blocks_sv(manager, state, ab_body)
                            c.partition()
                            c.build_cfg(manager, state)
                            cfgs_by_module[instance_name].append(c)


                        """# build X CFGx for the particular module 
                        cfg = CFG()
                        cfg.reset()
                        cfg.get_always_sv(manager, state, module.items)
                        cfg_count = len(cfg.always_blocks)
                        for k in range(cfg_count):
                            cfg.basic_blocks(manager, state, cfg.always_blocks[k])
                            cfg.partition()
                            # print(cfg.all_nodes)
                            # print(cfg.partition_points)
                            # print(len(cfg.basic_block_list))
                            # print(cfg.edgelist)
                            cfg.build_cfg(manager, state)
                            cfg.module_name = ast.name

                            cfgs_by_module[instance_name].append(deepcopy(cfg))
                            cfg.reset()"""
                            #print(cfg.paths)
                        state.store[instance_name] = {}
                        manager.dependencies[instance_name] = {}
                        manager.intermodule_dependencies[instance_name] = {}
                        manager.cond_assigns[instance_name] = {}
                else: 
                    """print(f"Module {sv_module_name} single instance")
                    manager.names_list.append(sv_module_name)
                    # build X CFGx for the particular module 
                    cfg = CFG()
                    cfg.all_nodes = []
                    #cfg.partition_points = []
                    cfg.get_always_sv(manager, state, module)
                    cfg_count = len(cfg.always_blocks)
                    # TODO: resolve deepcopy issue here
                    always_blocks_by_module[sv_module_name] = cfg.always_blocks
                    for k in range(cfg_count):
                        cfg.basic_blocks_sv(manager, state, always_blocks_by_module[sv_module_name][k])
                        cfg.partition()
                        # print(cfg.partition_points)
                        # print(len(cfg.basic_block_list))
                        # print(cfg.edgelist)
                        cfg.build_cfg(manager, state)
                        #print(cfg.cfg_edges)

                        #TODO: double-check curr_module starts at the right spot
                        cfg.module_name = manager.curr_module
                        # TODO: used to be Deepcopy in Sylvia,too 
                        cfgs_by_module[sv_module_name].append(cfg)
                        cfg.reset()
                        #print(cfg.paths)"""
                    


                    print(f"Module {sv_module_name} single instance")
                    manager.names_list.append(sv_module_name)
                    modules_dict[sv_module_name] = module                 # store AST
                    

                    # discover always blocks once
                    probe = CFG()
                    probe.get_always_sv(manager, state, module)
                    always_blocks_by_module[sv_module_name] = probe.always_blocks
                    #print(probe.always_blocks)

                    # fresh CFG per always (SV walker)
                    cfgs_by_module[sv_module_name] = []
                    for ab in always_blocks_by_module[sv_module_name]:
                        ab_body = getattr(ab, "statement", getattr(ab, "members", ab))
                        c = CFG()
                        c.module_name = sv_module_name
                        c.basic_blocks_sv(manager, state, ab_body)
                        c.partition()
                        c.build_cfg(manager, state)
                        cfgs_by_module[sv_module_name].append(c)


                    state.store[sv_module_name] = {}
                    manager.dependencies[sv_module_name] = {}
                    manager.intermodule_dependencies[sv_module_name] = {}
                    manager.cond_assigns[sv_module_name] = {}
            total_paths = 1
            for x in manager.child_num_paths.values():
                total_paths *= x

            # have do do things piece wise
            manager.debug = self.debug


            if len(modules) > 1:
                self.populate_seen_mod(manager)
                #manager.opt_1 = True
            else:
                manager.opt_1 = False
            manager.modules = modules_dict

            mapped_paths = {}
            
            #print(total_paths)




        print("Here")
        print(f"Branch points explored: {manager.branch_count}")     #TODO



        if self.debug:
            manager.debug = True
        # NOTE: assertions_always_intersect() was removed - it depended on PyVerilog functions
        # This function call has been commented out. If assertion intersection logic is needed,
        # it should be reimplemented using PySlang AST types.
        # self.assertions_always_intersect(manager) # Where is this function defined?

        manager.seen = {}
        for name in manager.names_list:
            manager.seen[name] = []

            # each module has a mapping table of cfg idx to path list
            mapped_paths[name] = {}
        manager.curr_module = manager.names_list[0]

        # index into cfgs list
        """curr_cfg = 0
        for module_name in cfgs_by_module:
            for cfg in cfgs_by_module[module_name]:
                mapped_paths[module_name][curr_cfg] = cfg.paths
                curr_cfg += 1
            curr_cfg = 0"""
        for module_name, cfg_list in cfgs_by_module.items():
            for i, cfg in enumerate(cfg_list):
                mapped_paths[module_name][i] = cfg.paths


        #stride_length = cfg_count
        # Single-cycle path combination: Use generators (OOM fix - this is the main fix)
        # Multi-cycle and cross-module: Reverted to list-based for explanation purposes
        single_paths_by_module = {}
        total_paths_by_module = {}
        for module_name in cfgs_by_module:
            print(f"Module {module_name} has {len(cfgs_by_module[module_name])} always blocks")
            # Single-cycle path combination: Use generator (lazy evaluation - prevents OOM)
            single_paths_by_module[module_name] = product(*mapped_paths[module_name].values())
            # NOTE: This will consume the generator above, so we need to recreate it
            total_paths_by_module[module_name] = list(tuple(product(product(*mapped_paths[module_name].values()), repeat=int(num_cycles))))
        # {total_paths_by_module}")
        #print(f"single paths by module: {total_paths_by_module}")
        if not total_paths_by_module:
            total_paths = []
        else:
            keys = list(total_paths_by_module.keys())
            values = []
            for key in keys:
                module_paths = total_paths_by_module[key]
                if not module_paths:
                    module_paths = [tuple(() for _ in range(int(num_cycles)))]
                values.append(module_paths)
            print(f"Module {key} paths: {module_paths}")
            # build total_paths as a list of dicts where each dict picks one path (possibly multi-cycle)
            # from each module. This takes the Cartesian product across modules, selecting a single
            # path entry for every module in each combination.

            total_paths = []
            for path_combo in product(*values):
                # ensure each module value is a list (so iteration like `for complete_single_cycle_path in curr_path[module_name]` works)
                total_paths.append({k: list(p) for k, p in zip(keys, path_combo)})
        
        #single_paths = list(product(*mapped_paths[manager.curr_module].values()))
        #total_paths = list(tuple(product(single_paths, repeat=int(num_cycles))))

        # for each combinatoin of multicycle paths

        print(total_paths)

        for i in range(len(total_paths)):
            manager.prev_store = state.store
            init_state(state, manager.prev_store, module, visitor)
            # initalize inputs with symbols for all submodules too
            for module_name in manager.names_list:
                manager.curr_module = module_name
                # actually want to terminate this part after the decl and comb part
                #compilation.getRoot().visit(my_visitor_for_symbol.visit)
                visitor.dfs(modules_dict[module_name])
                #self.search_strategy.visit_module(manager, state, ast, modules_dict)
                
            """for cfg_idx in range(cfg_count):
                for node in cfgs_by_module[manager.curr_module][cfg_idx].decls:
                    visitor.dfs(node)
                    #self.search_strategy.visit_stmt(manager, state, node, modules_dict, None)
                for node in cfgs_by_module[manager.curr_module][cfg_idx].comb:
                    visitor.dfs(node)
                    #self.search_strategy.visit_stmt(manager, state, node, modules_dict, None) """
            for c in cfgs_by_module[manager.curr_module]:
                for node in c.decls:
                    visitor.dfs(node)
                for node in c.comb:
                    visitor.dfs(node)

   
            manager.curr_module = manager.names_list[0]
            # makes assumption top level module is first in line
            # ! no longer path code as in bit string, but indices

            
            self.check_state(manager, state)

            curr_path = total_paths[i]
            modules_seen = 0
            for module_name in curr_path:
                manager.curr_module = manager.names_list[modules_seen]
                manager.cycle = 0
                for complete_single_cycle_path in curr_path[module_name]:
                    #for cfg_path in complete_single_cycle_path:
                    for cfg_idx, cfg_path in enumerate(complete_single_cycle_path):
                        directions = cfgs_by_module[module_name][cfg_idx].compute_direction(cfg_path)
                        #directions = cfgs_by_module[module_name][complete_single_cycle_path.index(cfg_path)].compute_direction(cfg_path)
                        k: int = 0
                        for basic_block_idx in cfg_path:
                            if basic_block_idx < 0: 
                                # dummy node
                                continue
                            else:
                                direction = directions[k]
                                k += 1
                                basic_block = cfgs_by_module[module_name][cfg_idx].basic_block_list[basic_block_idx]
                                #basic_block = cfgs_by_module[module_name][complete_single_cycle_path.index(cfg_path)].basic_block_list[basic_block_idx]
                                for stmt in basic_block:
                                    # print(f"updating curr mod {manager.curr_module}")
                                    #self.check_state(manager, state)
                                    visitor.visit_stmt(manager, state, stmt, modules_dict, direction)
                                    #self.search_strategy.visit_stmt(manager, state, stmt, modules_dict, direction)
                    # only do once, and the last CFG 
                    #for node in cfgs_by_module[module_name][complete_single_cycle_path.index(cfg_path)].comb:
                        #self.search_strategy.visit_stmt(manager, state, node, modules_dict, None)  
                    manager.cycle += 1
                modules_seen += 1
            manager.cycle = 0
            self.done = True
            self.check_state(manager, state)
            self.done = False

            manager.curr_level = 0
            for module_name in manager.instances_seen:
                manager.instances_seen[module_name] = 0
                manager.instances_loc[module_name] = ""
            if self.debug:
                print("------------------------")
            if (manager.assertion_violation):
                print("Assertion violation")
                #manager.assertion_violation = False
                counterexample = {}
                symbols_to_values = {}
                solver_start = time.process_time()
                if self.solve_pc(state.pc):
                    solver_end = time.process_time()
                    manager.solver_time += solver_end - solver_start
                    solved_model = state.pc.model()
                    decls =  solved_model.decls()
                    for item in decls:
                        symbols_to_values[item.name()] = solved_model[item]

                    # plug in phase
                    for module in state.store:
                        for signal in state.store[module]:
                            for symbol in symbols_to_values:
                                if state.store[module][signal] == symbol:
                                    counterexample[signal] = symbols_to_values[symbol]

                    print(counterexample)
                else:
                    print("UNSAT")
                return
            
            state.pc.reset()

            for module in manager.dependencies:
                module = {}
                
            
            manager.ignore = False
            manager.abandon = False
            manager.reg_writes.clear()
            for name in manager.names_list:
                state.store[name] = {}
            manager.path_count += 1
        print(f"Branch points explored: {manager.branch_count}")
        print(f"Paths explored: {manager.path_count}")
        self.module_depth -= 1

    def check_state(self, manager, state):
        """Checks the status of the execution and displays the state."""
        if self.done and manager.debug and not manager.is_child and not manager.init_run_flag and not manager.ignore and not manager.abandon:
            print(f"Cycle {manager.cycle} final state:")
            print(state.store)
    
            print(f"Cycle {manager.cycle} final path condition:")
            print(state.pc)
        elif self.done and not manager.is_child and manager.assertion_violation and not manager.ignore and not manager.abandon:
            print(f"Cycle {manager.cycle} initial state:")
            print(manager.initial_store)

            print(f"Cycle {manager.cycle} final state:")
            print(state.store)
    
            print(f"Cycle {manager.cycle} final path condition:")
            print(state.pc)
        elif manager.debug and not manager.is_child and not manager.init_run_flag and not manager.ignore:
            print("Initial state:")
            print(state.store)
                
