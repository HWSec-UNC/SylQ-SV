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
from strategies.dfs import DepthFirst
import sys
from copy import deepcopy
import pyslang as ps
from helpers.slang_helpers import get_module_name, init_state

CONDITIONALS = (
    ps.ConditionalStatementSyntax,
    ps.CaseStatementSyntax,
    ps.ForeachLoopStatementSyntax,
    ps.ForLoopStatementSyntax,
    ps.LoopStatementSyntax,
    ps.DoWhileStatementSyntax
)
class ExecutionEngine:
    module_depth: int = 0
    search_strategy = DepthFirst()
    debug: bool = False
    done: bool = False

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
        """Solve path condition."""
        result = str(s.check())
        if str(result) == "sat":
            model = s.model()
            return True
        else:
            return False

    def count_conditionals_2(self, m:ExecutionManager, items) -> int:
        """Rewrite to actually return an int."""
        stmts = items
        if isinstance(items, Block):
            stmts = items.statements
            items.cname = "Block"

        if hasattr(stmts, '__iter__'):
            for item in stmts:
                if isinstance(item, CONDITIONALS):
                    if isinstance(item, IfStatement):
                        return self.count_conditionals_2(m, item.true_statement) + self.count_conditionals_2(m, item.false_statement)  + 1
                    elif isinstance(items, CaseStatement):
                        return self.count_conditionals_2(m, items.caselist) + 1
                    elif isinstance(items, ForStatement):
                        return self.count_conditionals_2(m, items.statement) + 1
                if isinstance(item, Block):
                    return self.count_conditionals_2(m, item.items)
                elif isinstance(item, Always):
                   return self.count_conditionals_2(m, item.statement)             
                elif isinstance(item, Initial):
                    return self.count_conditionals_2(m, item.statement)
        elif items != None:
            if isinstance(items, IfStatement):
                return  ( self.count_conditionals_2(m, items.true_statement) + 
                self.count_conditionals_2(m, items.false_statement)) + 1
            if isinstance(items, CaseStatement):
                return self.count_conditionals_2(m, items.caselist) + len(items.caselist)
            if isinstance(items, ForStatement):
                return self.count_conditionals_2(m, items.statement) + 1
        return 0

    def count_conditionals(self, m: ExecutionManager, items):
        """Identify control flow structures to count total number of paths."""
        stmts = items
        if isinstance(items, Block):
            stmts = items.statements
            items.cname = "Block"
        if hasattr(stmts, '__iter__'):
            for item in stmts:
                if isinstance(item, CONDITIONALS):
                    if isinstance(item, IfStatement):
                        m.num_paths *= 2
                        self.count_conditionals(m, item.true_statement)
                        self.count_conditionals(m, item.false_statement)
                    elif isinstance(item, CaseStatement):
                        for case in item.caselist:
                            m.num_paths *= 2
                            self.count_conditionals(m, case.statement)
                    elif isinstance(item, ForStatement):
                        m.num_paths *= 2
                        self.count_conditionals(m, item.statement) 
                if isinstance(item, Block):
                    self.count_conditionals(m, item.items)
                elif isinstance(item, Always):
                    self.count_conditionals(m, item.statement)             
                elif isinstance(item, Initial):
                    self.count_conditionals(m, item.statement)
                elif isinstance(item, Case):
                    self.count_conditionals(m, item.statement)
        elif items != None:
            if isinstance(items, IfStatement):
                m.num_paths *= 2
                self.count_conditionals(m, items.true_statement)
                self.count_conditionals(m, items.false_statement)
            if isinstance(items, CaseStatement):
                for case in items.caselist:
                    m.num_paths *= 2
                    self.count_conditionals(m, case.statement)
            if isinstance(items, ForStatement):
                m.num_paths *= 2
                self.count_conditionals(m, items.statement) 

    def lhs_signals(self, m: ExecutionManager, items):
        """Take stock of which signals are written to in which always blocks for COI analysis."""
        stmts = items
        if isinstance(items, Block):
            stmts = items.statements
            items.cname = "Block"
        if hasattr(stmts, '__iter__'):
            for item in stmts:
                if isinstance(item, IfStatement) or isinstance(item, CaseStatement):
                    if isinstance(item, IfStatement):
                        self.lhs_signals(m, item.true_statement)
                        self.lhs_signals(m, item.false_statement)
                    if isinstance(item, CaseStatement):
                        for case in item.caselist:
                            self.lhs_signals(m, case.statement)
                if isinstance(item, Block):
                    self.lhs_signals(m, item.items)
                elif isinstance(item, Always):
                    m.curr_always = item
                    m.always_writes[item] = []
                    self.lhs_signals(m, item.statement)             
                elif isinstance(item, Initial):
                    self.lhs_signals(m, item.statement)
                elif isinstance(item, Case):
                    self.lhs_signals(m, item.statement)
                elif isinstance(item, Assign):
                    if isinstance(item.left.var, Partselect):
                        if m.curr_always is not None and item.left.var.var.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(item.left.var.var.name)
                    elif isinstance(item.left.var, Pointer):
                        if m.curr_always is not None and item.left.var.var.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(item.left.var.ptr)
                    elif isinstance(item.left.var, Concat) and m.curr_always is not None:
                        for sub_item in item.left.var.list:
                            m.always_writes[m.curr_always].append(sub_item.name)
                    elif m.curr_always is not None and item.left.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(item.left.var.name)
                elif isinstance(item, NonblockingSubstitution):
                    if isinstance(item.left.var, Partselect):
                        if m.curr_always is not None and item.left.var.var.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(item.left.var.var.name)
                    elif isinstance(item.left.var, Concat):
                        for sub_item in item.left.var.list:
                            if isinstance(sub_item, Partselect):
                                if m.curr_always is not None and sub_item.var.name not in m.always_writes[m.curr_always]:
                                    m.always_writes[m.curr_always].append(sub_item.var.name)
                            elif isinstance(sub_item, Pointer):
                                if m.curr_always is not None and sub_item.var.name not in m.always_writes[m.curr_always]:
                                    m.always_writes[m.curr_always].append(sub_item.var.name)
                            else:
                                m.always_writes[m.curr_always].append(sub_item.name)
                    elif isinstance(item.left.var, Pointer):
                        if m.curr_always is not None and item.left.var.var.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(item.left.var.var.name)
                    elif m.curr_always is not None and item.left.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(item.left.var.name)
                elif isinstance(item, BlockingSubstitution):
                    if isinstance(item.left.var, Partselect):
                        if m.curr_always is not None and item.left.var.var.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(item.left.var.var.name)
                    elif isinstance(item.left.var, Pointer):
                        if m.curr_always is not None and item.left.var.var.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(item.left.var.var.name)
                    elif m.curr_always is not None and item.left.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(item.left.var.name)
        elif items != None:
            if isinstance(items, IfStatement):
                self.lhs_signals(m, items.true_statement)
                self.lhs_signals(m, items.false_statement)
            if isinstance(items, CaseStatement):
                for case in items.caselist:
                    self.lhs_signals(m, case.statement)
            elif isinstance(items, Assign):
                if m.curr_always is not None and items.left.var.name not in m.always_writes[m.curr_always]:
                    m.always_writes[m.curr_always].append(items.left.var.name)
            elif isinstance(items, NonblockingSubstitution):
                if isinstance(items.left.var, Concat):
                    for sub_item in items.left.var.list:
                        if sub_item.name not in m.always_writes[m.curr_always]:
                            m.always_writes[m.curr_always].append(sub_item.name)
                elif isinstance(items.left.var, Partselect):
                    if m.curr_always is not None and items.left.var.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(item.left.var.var.name)
                elif isinstance(items.left.var, Pointer):
                    if m.curr_always is not None and items.left.var.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(items.left.var.var.name)
                elif m.curr_always is not None and items.left.var.name not in m.always_writes[m.curr_always]:
                    m.always_writes[m.curr_always].append(items.left.var.name)
            elif isinstance(items, BlockingSubstitution):
                if isinstance(items.left.var, Pointer):
                    if m.curr_always is not None and items.left.var.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(items.left.var.var.name)
                elif isinstance(items.left.var, Partselect):
                    if m.curr_always is not None and items.left.var.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(item.left.var.var.name)
                else:
                    if m.curr_always is not None and items.left.var.name not in m.always_writes[m.curr_always]:
                        m.always_writes[m.curr_always].append(items.left.var.name)



    def get_assertions(self, m: ExecutionManager, items):
        """Traverse the AST and get the assertion violating conditions."""
        stmts = items
        if isinstance(items, Block):
            stmts = items.statements
            items.cname = "Block"
        if hasattr(stmts, '__iter__'):
            for item in stmts:
                if isinstance(item, IfStatement) or isinstance(item, CaseStatement):
                    if isinstance(item, IfStatement):
                        # starting to check for the assertions
                        if isinstance(item.true_statement, Block):
                            if isinstance(item.true_statement.statements[0], SingleStatement):
                                if isinstance(item.true_statement.statements[0].statement, SystemCall) and "ASSERTION" in item.true_statement.statements[0].statement.args[0].value:
                                    m.assertions.append(item.cond)
                                    #print("assertion found")
                            else:     
                                self.get_assertions(m, item.true_statement)
                            #self.get_assertions(m, item.false_statement)
                    if isinstance(item, CaseStatement):
                        for case in item.caselist:
                            self.get_assertions(m, case.statement)
                elif isinstance(item, Block):
                    self.get_assertions(m, item.items)
                elif isinstance(item, Always):
                    self.get_assertions(m, item.statement)             
                elif isinstance(item, Initial):
                    self.get_assertions(m, item.statement)
                elif isinstance(item, Case):
                    self.get_assertions(m, item.statement)
        elif items != None:
            if isinstance(items, IfStatement):
                self.get_assertions(m, items.true_statement)
                self.get_assertions(m, items.false_statement)
            if isinstance(items, CaseStatement):
                for case in items.caselist:
                    self.get_assertions(m, case.statement)

    def map_assertions_signals(self, m: ExecutionManager):
        """Map the assertions to a list of relevant signals."""
        signals = []
        for assertion in m.assertions:
            # TODO write function to exhaustively get all the signals from assertions
            # this is just grabbing the left most
            if isinstance(assertion.right, IntConst):
                ...
            elif isinstance(assertion.right.left, Identifier):
                signals.append(assertion.right.left.name)
        return signals

    def assertions_always_intersect(self, m: ExecutionManager):
        """Get the always blocks that have the signals relevant to the assertions."""
        signals_of_interest = self.map_assertions_signals(m)
        blocks_of_interest = []
        for block in m.always_writes:
            for signal in signals_of_interest:
                if signal in m.always_writes[block]:
                    blocks_of_interest.append(block)
        m.blocks_of_interest = blocks_of_interest


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

    def module_count(self, m: ExecutionManager, items) -> None:
        """Traverse a top level module and count up the instances of each type of module."""
        if isinstance(items, Block):
            items = items.statements
        if hasattr(items, '__iter__'):
            for item in items:
                if isinstance(item, InstanceList):
                    self.module_count(m, item.instances)
                elif isinstance(item, Instance):
                    if item.module in m.instance_count:
                        m.instance_count[item.module] += 1
                        ...
                    else:
                        m.instance_count[item.module] = 1
                if isinstance(item, Block):
                    self.module_count(m, item.items)
                elif isinstance(item, Always):
                    self.module_count(m, item.statement)             
                elif isinstance(item, Initial):
                    self.module_count(m, item.statement)
        elif items != None:
                if isinstance(items, InstanceList):
                    if items.module in m.instance_count:
                        m.instance_count[items.module] += 1
                    else:
                        m.instance_count[items.module] = 1
                    self.module_count(m, items.instances)



    def init_run(self, m: ExecutionManager, module) -> None:
        """Initalize run."""
        m.init_run_flag = True
        # come back to this stuff 
        # TODO change to members and redo 
        # self.count_conditionals(m, module.items)
        # self.lhs_signals(m, module.items)
        # self.get_assertions(m, module.items)
        m.init_run_flag = False
        #self.module_count(m, module.items)


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

    def piece_wise_execute(self, ast, manager: Optional[ExecutionManager], modules) -> None:
        """Drives symbolic execution piecewise when number of paths is too large not to breakup. 
        We break it up to avoid the memory blow up."""
        self.module_depth += 1
        manager.piece_wise = True
        state: SymbolicState = SymbolicState()
        if manager is None:
            manager: ExecutionManager = ExecutionManager()
            manager.debugging = False
        modules_dict = {}
        for module in modules:
            modules_dict[module.name] = module
            manager.seen_mod[module.name] = {}
            sub_manager = ExecutionManager()
            manager.names_list.append(module.name)
            self.init_run(sub_manager, module)
            self.module_count(manager, module.items)
            if module.name in manager.instance_count:
                manager.instances_seen[module.name] = 0
                manager.instances_loc[module.name] = ""
                num_instances = manager.instance_count[module.name]
                for i in range(num_instances):
                    instance_name = f"{module.name}_{i}"
                    manager.names_list.append(instance_name)
                    manager.child_path_codes[instance_name] = to_binary(0)
                    manager.child_num_paths[instance_name] = sub_manager.num_paths
                    manager.config[instance_name] = to_binary(0)
                    state.store[instance_name] = {}
                    manager.dependencies[instance_name] = {}
                    manager.intermodule_dependencies[instance_name] = {}
                    manager.cond_assigns[instance_name] = {}
                manager.names_list.remove(module.name)
            else:
                manager.child_path_codes[module.name] = to_binary(0)
                manager.child_num_paths[module.name] = sub_manager.num_paths
                manager.config[module.name] = to_binary(0)
                state.store[module.name] = {}
                manager.dependencies[module.name] = {}
                instance_name = module.name
                manager.intermodule_dependencies[instance_name] = {}
                manager.cond_assigns[module.name] = {}

        total_paths = sum(manager.child_num_paths.values())
        #print(total_paths)
        manager.piece_wise = True
        #TODO: things piecewise, say 10,000 at a time.
        for i in range(0, total_paths, 10):
            manager.child_range = range(i*10, i*10+10)
            self.populate_child_paths(manager)
            if len(modules) >= 1:
                self.populate_seen_mod(manager)
                #manager.opt_1 = True
            else:
                manager.opt_1 = False
            manager.modules = modules_dict
            paths = list(product(*manager.child_path_codes.values()))
            #print(f" Upper bound on num paths {len(paths)}")
            self.init_run(manager, ast)
            manager.seen = {}
            for name in manager.names_list:
                manager.seen[name] = []
            manager.curr_module = manager.names_list[0]

            stride_length = len(manager.names_list)
            # for each combinatoin of multicycle paths
            for i in range(len(paths)):
                manager.cycle = 0

                for j in range(0, len(paths[i])):
                    for name in manager.names_list:
                        manager.config[name] = paths[i][j]

                manager.path_code = paths[i][0]
                manager.prev_store = state.store
                init_state(state, manager.prev_store, ast)
                self.search_strategy.visit_module(manager, state, ast, modules_dict)
                manager.cycle += 1
                manager.curr_level = 0
                if self.check_dup(manager):
                # #if False:
                    if self.debug:
                        print("----------------------")
                    ...
                else:
                    if self.debug:
                        print("------------------------")
                    ...
                    #print(f"{ast.name} Path {i}")
                manager.seen[ast.name].append(manager.path_code)
                if (manager.assertion_violation):
                    print("Assertion violation")
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
                for module in manager.dependencies:
                    module = {}
                for module in manager.intermodule_dependencies:
                    module = {}
                state.pc.reset()

                manager.ignore = False
                manager.abandon = False
                manager.reg_writes.clear()
                for name in manager.names_list:
                    state.store[name] = {}

            #manager.path_code = to_binary(0)
            #print(f" finishing {ast.name}")
            self.module_depth -= 1

    def multicycle_helper(self, ast, modules_dict, paths,  s: SymbolicState, manager: ExecutionManager, num_cycles: int) -> None:
        """Recursive Helper to resolve multi cycle execution."""
        #TODO: Add in the merging state element to this helper function
        for a in range(num_cycles):
            for i in range(len(paths)):
                for j in range(len(paths[i])):
                    manager.config[manager.names_list[j]] = paths[i][j]

    def execute_sv(self, visitor, modules, manager: Optional[ExecutionManager], num_cycles: int) -> None:
        """Drives symbolic execution for SystemVerilog designs."""
        gc.collect()
        print(f"Executing for {num_cycles} clock cycles")
        self.module_depth += 1
        state: SymbolicState = SymbolicState()
        if manager is None:
            manager: ExecutionManager = ExecutionManager()
            manager.cache = self.cache
            manager.sv = True
            manager.debugging = False
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
                self.init_run(sub_manager, module)
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




            #NEVER ENTER THE FOLLOWING IF, IF-ELSE
            if total_paths > 100:
                start = time.process_time()
                self.piece_wise_execute(ast, manager, modules)
                end = time.process_time()
                print(f"Elapsed time {end - start}")
                print(f"Solver time {manager.solver_time}")
                sys.exit()
            self.populate_child_paths(manager)
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
        self.assertions_always_intersect(manager)

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
        single_paths_by_module = {}
        total_paths_by_module = {}
        for module_name in cfgs_by_module:
            print(f"Module {module_name} has {len(cfgs_by_module[module_name])} always blocks")
            single_paths_by_module[module_name] = list(product(*mapped_paths[module_name].values()))
            total_paths_by_module[module_name] = list(tuple(product(single_paths_by_module[module_name], repeat=int(num_cycles))))
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

    #@profile     
    def execute(self, ast, modules, manager: Optional[ExecutionManager], directives, num_cycles: int) -> None:
        """Drives symbolic execution."""
        gc.collect()
        print(f"Executing for {num_cycles} clock cycles")
        self.module_depth += 1
        state: SymbolicState = SymbolicState()
        if manager is None:
            manager: ExecutionManager = ExecutionManager()
            manager.debugging = False
            modules_dict = {}
            # a dictionary keyed by module name, that gives the list of cfgs
            cfgs_by_module = {}
            cfg_count_by_module = {}
            for module in modules:
                modules_dict[module.name] = module
                always_blocks_by_module = {module.name: []}
                manager.seen_mod[module.name] = {}
                cfgs_by_module[module.name] = []
                sub_manager = ExecutionManager()
                self.init_run(sub_manager, module)
                self.module_count(manager, module.items) 
                if module.name in manager.instance_count:
                    manager.instances_seen[module.name] = 0
                    manager.instances_loc[module.name] = ""
                    num_instances = manager.instance_count[module.name]
                    cfgs_by_module.pop(module.name, None)
                    for i in range(num_instances):
                        instance_name = f"{module.name}_{i}"
                        manager.names_list.append(instance_name)
                        cfgs_by_module[instance_name] = []
                        # build X CFGx for the particular module 
                        cfg = CFG()
                        cfg.reset()
                        cfg.get_always(manager, state, module.items)
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
                            cfg.reset()
                            #print(cfg.paths)


                        state.store[instance_name] = {}
                        manager.dependencies[instance_name] = {}
                        manager.intermodule_dependencies[instance_name] = {}
                        manager.cond_assigns[instance_name] = {}
                else: 
                    manager.names_list.append(module.name)
                    # build X CFGx for the particular module 
                    cfg = CFG()
                    cfg.all_nodes = []
                    #cfg.partition_points = []
                    cfg.get_always(manager, state, ast.items)
                    cfg_count = len(cfg.always_blocks)
                    always_blocks_by_module[module.name] = deepcopy(cfg.always_blocks)
                    for k in range(cfg_count):
                        cfg.basic_blocks(manager, state, always_blocks_by_module[module.name][k])
                        cfg.partition()
                        # print(cfg.partition_points)
                        # print(len(cfg.basic_block_list))
                        # print(cfg.edgelist)
                        cfg.build_cfg(manager, state)
                        #print(cfg.cfg_edges)
                        cfg.module_name = ast.name
                        cfgs_by_module[module.name].append(deepcopy(cfg))
                        cfg.reset()
                        #print(cfg.paths)

                    state.store[module.name] = {}
                    manager.dependencies[module.name] = {}
                    manager.intermodule_dependencies[module.name] = {}
                    manager.cond_assigns[module.name] = {}
            total_paths = 1
            for x in manager.child_num_paths.values():
                total_paths *= x

            # have do do things piece wise
            manager.debug = self.debug
            if total_paths > 100:
                start = time.process_time()
                self.piece_wise_execute(ast, manager, modules)
                end = time.process_time()
                print(f"Elapsed time {end - start}")
                print(f"Solver time {manager.solver_time}")
                sys.exit()
            self.populate_child_paths(manager)
            if len(modules) > 1:
                self.populate_seen_mod(manager)
                #manager.opt_1 = True
            else:
                manager.opt_1 = False
            manager.modules = modules_dict

            mapped_paths = {}
            #print(total_paths)

        if self.debug:
            manager.debug = True
        self.assertions_always_intersect(manager)

        manager.seen = {}
        for name in manager.names_list:
            manager.seen[name] = []

            # each module has a mapping table of cfg idx to path list
            mapped_paths[name] = {}
        manager.curr_module = manager.names_list[0]

        # index into cfgs list
        curr_cfg = 0
        """for module_name in cfgs_by_module:
            for cfg in cfgs_by_module[module_name]:
                mapped_paths[module_name][curr_cfg] = cfg.paths
                curr_cfg += 1
            curr_cfg = 0"""
        for module_name, cfg_list in cfgs_by_module.items():
            for i, cfg in enumerate(cfg_list):
                mapped_paths[module_name][i] = cfg.paths

        print(mapped_paths)

        #stride_length = cfg_count
        single_paths_by_module = {}
        total_paths_by_module = {}
        for module_name in cfgs_by_module:
            single_paths_by_module[module_name] = list(product(*mapped_paths[module_name].values()))
            total_paths_by_module[module_name] = list(tuple(product(single_paths_by_module[module_name], repeat=int(num_cycles))))
        #print(f"tp {total_paths_by_module}")
        keys, values = zip(*total_paths_by_module.items())
        total_paths = [dict(zip(keys, path)) for path in product(*values)]
        #print(total_paths)
        
        #single_paths = list(product(*mapped_paths[manager.curr_module].values()))
        #total_paths = list(tuple(product(single_paths, repeat=int(num_cycles))))

        # for each combinatoin of multicycle paths

        for i in range(len(total_paths)):
            manager.prev_store = state.store
            init_state(state, manager.prev_store, ast)
            # initalize inputs with symbols for all submodules too
            for module_name in manager.names_list:
                manager.curr_module = module_name
                # actually want to terminate this part after the decl and comb part
                self.search_strategy.visit_module(manager, state, ast, modules_dict)
                
            for cfg_idx in range(cfg_count):
                for node in cfgs_by_module[manager.curr_module][cfg_idx].decls:
                    self.search_strategy.visit_stmt(manager, state, node, modules_dict, None)
                for node in cfgs_by_module[manager.curr_module][cfg_idx].comb:
                    self.search_strategy.visit_stmt(manager, state, node, modules_dict, None) 

   
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
                                    self.search_strategy.visit_stmt(manager, state, stmt, modules_dict, direction)
                                            # only do once, and the last CFG 
                    for node in cfgs_by_module[module_name][cfg_count-1].comb:
                        self.search_strategy.visit_stmt(manager, state, node, modules_dict, None)  
                        print(state.store)
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

        self.module_depth -= 1


    def execute_child(self, ast, state: SymbolicState, manager: Optional[ExecutionManager]) -> None:
        """Drives symbolic execution of child modules."""
        # different manager
        # same state
        # dont call pc solve
        manager_sub = ExecutionManager()
        manager_sub.is_child = True
        manager_sub.curr_module = ast.name
        self.init_run(manager_sub, ast)

        manager_sub.path_code = manager.config[ast.name]
        manager_sub.seen = manager.seen

        # mark this exploration of the submodule as seen and store the state so we don't have to explore it again.
        if manager.seen_mod[ast.name][manager_sub.path_code] == {}:
            manager.seen_mod[ast.name][manager_sub.path_code] = state.store
        else:
            ...
            #print("already seen this")
        # i'm pretty sure we only ever want to do 1 loop here
        for i in range(1):
        #for i in range(manager_sub.num_paths):
            manager_sub.path_code = manager.config[ast.name]

            self.search_strategy.visit_module(manager_sub, state, ast, manager.modules)
            if (manager.assertion_violation):
                print("Assertion violation")
                manager.assertion_violation = False
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
            manager.curr_level = 0
            #state.pc.reset()
        #manager.path_code = to_binary(0)
        if manager_sub.ignore:
            manager.ignore = True
        self.module_depth -= 1
        #manager.is_child = False


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
                
