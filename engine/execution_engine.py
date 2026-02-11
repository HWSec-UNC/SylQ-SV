# Main execution engine that orchestrates symbolic execution of SystemVerilog designs

from z3 import Solver, ExprRef
from z3 import z3util
from .execution_manager import ExecutionManager
from .symbolic_state import SymbolicState
from .cfg import CFG
from typing import Optional, Generator
import time
import gc
from itertools import product
from helpers.utils import to_binary
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
    module_depth: int = 0  # Tracks current module nesting depth during execution
    debug: bool = False    # Boolean flag to enable debug output
    done: bool = False     # Boolean flag indicating if execution is complete
    timeout: bool = False  # Set to True by main.py timeout handler to request early stop

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

    def explore_block(self, visitor, manager: ExecutionManager, state_template: SymbolicState,
                      module_name: str, cfg: CFG, modules_dict: dict) -> Generator[dict, None, None]:
        """Explore all paths through a single always block independently (streaming generator).
        Yields {'pc': list of z3 constraints, 'store': {signal: expr}} per feasible path."""
        prev_curr_module = manager.curr_module
        manager.curr_module = module_name
        num_paths = cfg.get_path_count()
        print("explore_block: {} ({} paths)".format(module_name, num_paths), flush=True)
        try:
            for path_idx, path in enumerate(cfg.get_paths()):
                if path_idx == 0 or (path_idx + 1) % 100 == 0 or path_idx == num_paths - 1:
                    print("  path {}/{}".format(path_idx + 1, num_paths), flush=True)
                manager.ignore = False
                manager.abandon = False
                path_state = state_template.fresh_for_block(module_name,
                    state_template.snapshot(module_name))
                directions = cfg.compute_direction(path)
                k = 0
                for bb_idx in path:
                    if bb_idx < 0:
                        continue
                    direction = directions[k] if k < len(directions) else 1
                    k += 1
                    basic_block = cfg.basic_block_list[bb_idx]
                    for stmt in basic_block:
                        # Expose current PC for _cache_key slicing (Paper §4.2.2)
                        manager._pc_ref = path_state.pc
                        try:
                            visitor.visit_stmt(manager, path_state, stmt, modules_dict, direction)
                        finally:
                            manager._pc_ref = None
                try:
                    constraints = list(path_state.pc.assertions())
                except Exception:
                    constraints = []
                try:
                    path_state.pc.set("timeout", 10000)
                except Exception:
                    pass
                if not constraints or str(path_state.pc.check()) == "sat":
                    yield {
                        "pc": constraints,
                        "store": dict(path_state.store.get(module_name, {}))
                    }
        finally:
            manager.curr_module = prev_curr_module
            if hasattr(manager, '_pc_ref'):
                manager._pc_ref = None

    def _collect_assertions(self, ast, module_name, assertions_list):
        """Recursively walk a pyslang AST and collect SVA assertion nodes.

        Populates assertions_list with dicts containing:
          - 'node': the raw AST node
          - 'module': the module name the assertion belongs to
          - 'source': source location string for reporting
          - 'kind': 'immediate' or 'concurrent'
        The z3 expression ('z3_expr') is filled in later during symbolic evaluation.
        """
        if ast is None:
            return

        # Handle iterable containers (e.g., module members)
        if isinstance(ast, (list, tuple)):
            for item in ast:
                self._collect_assertions(item, module_name, assertions_list)
            return

        cname = ast.__class__.__name__ if hasattr(ast, '__class__') else ''

        # Check for immediate assertion statements: assert(expr)
        if cname in ('ImmediateAssertStatementSyntax', 'ImmediateAssertionMemberSyntax',
                      'ImmediateAssumeStatementSyntax', 'ImmediateCoverStatementSyntax'):
            source = str(getattr(ast, 'sourceRange', getattr(ast, 'toString',
                         lambda: '<unknown>'))() if callable(
                         getattr(ast, 'toString', None)) else getattr(ast, 'sourceRange', '<unknown>'))
            assertions_list.append({
                'node': ast,
                'module': module_name,
                'source': source,
                'kind': 'immediate',
                'z3_expr': None,  # filled during explore_block or cross-module check
            })
            return

        # Check for concurrent assertion statements: assert property (...)
        if cname in ('AssertPropertyStatementSyntax', 'ConcurrentAssertionMemberSyntax',
                      'AssumePropertyStatementSyntax', 'CoverPropertyStatementSyntax',
                      'ExpectPropertyStatementSyntax'):
            source = str(getattr(ast, 'sourceRange', getattr(ast, 'toString',
                         lambda: '<unknown>'))() if callable(
                         getattr(ast, 'toString', None)) else getattr(ast, 'sourceRange', '<unknown>'))
            assertions_list.append({
                'node': ast,
                'module': module_name,
                'source': source,
                'kind': 'concurrent',
                'z3_expr': None,
            })
            return

        # Recurse into known container types
        if isinstance(ast, ps.ModuleDeclarationSyntax):
            for mem in ast.members:
                self._collect_assertions(mem, module_name, assertions_list)
            return

        if hasattr(ast, 'statement'):
            self._collect_assertions(getattr(ast, 'statement'), module_name, assertions_list)
        if hasattr(ast, 'items'):
            items = getattr(ast, 'items')
            if hasattr(items, '__iter__'):
                for item in items:
                    self._collect_assertions(item, module_name, assertions_list)
        if hasattr(ast, 'members'):
            members = getattr(ast, 'members')
            if hasattr(members, '__iter__') and not isinstance(ast, ps.ModuleDeclarationSyntax):
                for mem in members:
                    self._collect_assertions(mem, module_name, assertions_list)
        if hasattr(ast, 'body'):
            self._collect_assertions(getattr(ast, 'body'), module_name, assertions_list)

    def _eval_assertion_expr(self, assertion_info, visitor, manager, state, modules_dict):
        """Attempt to evaluate the assertion node's condition/expression into a Z3 expression.

        Tries multiple attribute names to locate the assertion condition in the
        pyslang AST node, then uses the visitor's expr_to_z3 to convert it.
        Sets assertion_info['z3_expr'] if successful.
        """
        node = assertion_info['node']
        expr_node = None

        # Try various attribute names for the assertion expression
        for attr in ('cond', 'expr', 'condition', 'expression'):
            if hasattr(node, attr):
                expr_node = getattr(node, attr)
                if expr_node is not None:
                    break

        # For concurrent assertions, try the property spec
        if expr_node is None:
            for attr in ('propertySpec', 'property', 'spec'):
                if hasattr(node, attr):
                    prop = getattr(node, attr)
                    if prop is not None:
                        # Property spec may have an .expr attribute
                        for pattr in ('expr', 'expression', 'cond'):
                            if hasattr(prop, pattr):
                                expr_node = getattr(prop, pattr)
                                if expr_node is not None:
                                    break
                        if expr_node is not None:
                            break

        # For nodes that wrap a statement containing the assertion
        if expr_node is None and hasattr(node, 'statement'):
            stmt = node.statement
            for attr in ('cond', 'expr', 'condition', 'expression'):
                if hasattr(stmt, attr):
                    expr_node = getattr(stmt, attr)
                    if expr_node is not None:
                        break

        if expr_node is None:
            return  # Could not find assertion expression

        try:
            visitor.visit_expr(manager, state, expr_node)
            z3_expr = visitor.expr_to_z3(manager, state, expr_node)
            if z3_expr is not None:
                assertion_info['z3_expr'] = z3_expr
        except Exception as e:
            import logging
            logging.debug(f"Failed to evaluate assertion expression: {e}")

    def _vars_in_pcs(self, pc_list):
        """Return set of variable names (str) appearing in the given list of Z3 constraints."""
        out = set()
        for c in pc_list:
            try:
                for v in z3util.get_vars(c):
                    out.add(str(v))
            except Exception:
                pass
        return out

    def merge_block_results_streaming(self, block_result_lists: list, module_name: str = "",
                                      manager=None):
        """Combine per-block results: Cartesian product filtered by SAT (streaming generator).
        If two blocks' PCs share no symbolic variables, skip the solver (paper §4.3).
        Merge-query caching uses a separate QU from path exploration (paper §4.3).
        Yields merged {'pc': list of constraints, 'store': dict} one at a time."""
        if not block_result_lists:
            yield {"pc": [], "store": {}}
            return

        # Import merge-query caching utilities
        from .query_normalization import normalize_query_list

        combo_count = 0
        yielded_count = 0
        for combo in product(*block_result_lists):
            # Global timeout hook: if main.py marked the engine as timed out,
            # stop generating further merged combinations.
            if getattr(self, "timeout", False):
                break
            combo_count += 1
            if combo_count % 100000 == 0:
                print(f"    merge {module_name}: checked {combo_count} combos, yielded {yielded_count} feasible", flush=True)
            
            all_pcs = []
            all_stores = {}
            var_sets = []
            for r in combo:
                all_pcs.extend(r["pc"])
                all_stores.update(r["store"])
                var_sets.append(self._vars_in_pcs(r["pc"]))
            need_solver = False
            for i in range(len(var_sets)):
                for j in range(i + 1, len(var_sets)):
                    if var_sets[i] & var_sets[j]:
                        need_solver = True
                        break
                if need_solver:
                    break
            if need_solver and all_pcs:
                # --- Merge-query caching (Paper §4.3) ---
                merge_cache_key = None
                if manager and manager.cache:
                    try:
                        merge_cache_key = "merge:" + normalize_query_list(all_pcs)
                        cached = manager.cache.get(merge_cache_key)
                        if cached is not None:
                            if cached.decode() != "sat":
                                continue
                            else:
                                yielded_count += 1
                                yield {"pc": all_pcs, "store": all_stores}
                                continue
                    except Exception:
                        merge_cache_key = None

                # Register merge constraints in separate QU
                if manager and manager.qu_merge is not None:
                    for c in all_pcs:
                        manager.qu_merge.register_constraint(c)

                s = Solver()
                try:
                    s.set("timeout", 10000)
                except Exception:
                    pass
                for c in all_pcs:
                    s.add(c)
                result_str = str(s.check())
                # Cache the merge result
                if merge_cache_key and manager and manager.cache:
                    try:
                        manager.cache.set(merge_cache_key, result_str)
                    except Exception:
                        pass
                if result_str != "sat":
                    continue
            yielded_count += 1
            yield {"pc": all_pcs, "store": all_stores}

    def execute_sv(self, visitor, modules, manager: Optional[ExecutionManager], num_cycles: int) -> None:
        """Main entry point for PySlang execution
        Drives symbolic execution for SystemVerilog designs."""
        gc.collect()
        print(f"Executing for {num_cycles} clock cycles")
        self.module_depth += 1
        state: SymbolicState = SymbolicState()
        
        if manager is not None:
            # Only manager=None is supported: we build modules_dict, cfgs_by_module, etc. here.
            raise ValueError("execute_sv requires manager=None; the engine creates the manager internally.")
        manager = ExecutionManager()
        if hasattr(self, "cache"):
            manager.cache = self.cache
        manager.sv = True
        # Store manager ref so main.py timeout handler can access stats
        self._last_manager = manager
        # Initialize Quick-Union for query slicing (Paper §4.2.2)
        from .query_slicing import QuickUnion
        manager.qu_path = QuickUnion()
        manager.qu_merge = QuickUnion()

        modules_dict = {}
        # a dictionary keyed by module name, that gives the list of cfgs
        cfgs_by_module = {}
        for module in modules:
            sv_module_name = get_module_name(module)
            modules_dict[sv_module_name] = module
            always_blocks_by_module = {sv_module_name: []}
            manager.seen_mod[sv_module_name] = {}
            cfgs_by_module[sv_module_name] = []
    
            self.module_count_sv(manager, module) 
            if sv_module_name in manager.instance_count:
                print(f"Module {sv_module_name} has {manager.instance_count[sv_module_name]} instances")
                manager.instances_seen[sv_module_name] = 0
                manager.instances_loc[sv_module_name] = ""
                num_instances = manager.instance_count[sv_module_name]
                cfgs_by_module.pop(sv_module_name, None)
                for i in range(num_instances):
                    instance_name = f"{sv_module_name}_{i}"
                    manager.names_list.append(instance_name)
                    cfgs_by_module[instance_name] = []
    
                     # 1) discover always blocks once
                    probe = CFG()
                    probe.get_always_sv(manager, state, module)
    
                    # 2) build a fresh CFG per sequential always block (SV walker)
                    for ab in probe.always_blocks:
                        ab_body = getattr(ab, "statement", getattr(ab, "members", ab))
                        c = CFG()
                        c.module_name = instance_name
                        c.is_combinational = False
                        c.basic_blocks_sv(manager, state, ab_body)
                        c.partition()
                        c.build_cfg(manager, state)
                        cfgs_by_module[instance_name].append(c)
    
                    # 3) build CFGs for always_comb blocks (Paper §4.4)
                    for ab in probe.always_comb_blocks:
                        ab_body = getattr(ab, "statement", getattr(ab, "members", ab))
                        c = CFG()
                        c.module_name = instance_name
                        c.is_combinational = True
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
    
                # discover always blocks once
                probe = CFG()
                probe.get_always_sv(manager, state, module)
                always_blocks_by_module[sv_module_name] = probe.always_blocks
    
                # fresh CFG per sequential always block (SV walker)
                cfgs_by_module[sv_module_name] = []
                for ab in always_blocks_by_module[sv_module_name]:
                    ab_body = getattr(ab, "statement", getattr(ab, "members", ab))
                    c = CFG()
                    c.module_name = sv_module_name
                    c.is_combinational = False
                    c.basic_blocks_sv(manager, state, ab_body)
                    c.partition()
                    c.build_cfg(manager, state)
                    cfgs_by_module[sv_module_name].append(c)
    
                # Build CFGs for always_comb blocks (Paper §4.4)
                for ab in probe.always_comb_blocks:
                    ab_body = getattr(ab, "statement", getattr(ab, "members", ab))
                    c = CFG()
                    c.module_name = sv_module_name
                    c.is_combinational = True
                    c.basic_blocks_sv(manager, state, ab_body)
                    c.partition()
                    c.build_cfg(manager, state)
                    cfgs_by_module[sv_module_name].append(c)
    
    
                state.store[sv_module_name] = {}
                manager.dependencies[sv_module_name] = {}
                manager.intermodule_dependencies[sv_module_name] = {}
                manager.cond_assigns[sv_module_name] = {}
        # total_paths = 1
        # for x in manager.child_num_paths.values():
        #     total_paths *= x
    
        # have do do things piece wise
        manager.debug = self.debug
    
    
        if len(modules) > 1:
            self.populate_seen_mod(manager)
            #manager.opt_1 = True
        else:
            manager.opt_1 = False
        manager.modules = modules_dict


        print("Here", flush=True)
        print("Branch points explored: {} (phase: initial)".format(manager.branch_count), flush=True)
        if self.debug:
            manager.debug = True
        # NOTE: assertions_always_intersect() was removed - it depended on PyVerilog functions
        # This function call has been commented out. If assertion intersection logic is needed,
        # it should be reimplemented using PySlang AST types.
        # self.assertions_always_intersect(manager) # Where is this function defined?

        manager.seen = {}
        for name in manager.names_list:
            manager.seen[name] = []
        manager.curr_module = manager.names_list[0]

        keys = list(cfgs_by_module.keys())
        if not keys or not manager.names_list:
            print("No modules or no CFGs; skipping path exploration.")
            self.module_depth -= 1
            return

        first_module = modules_dict[manager.names_list[0]] if manager.names_list else None

        # --- Piecewise composition (sole execution mode) ---
        manager.prev_store = state.store
        print("Phase: init_state and module traversal", flush=True)
        init_state(state, manager.prev_store, first_module, visitor)
        for module_name in manager.names_list:
            print("Phase: traversing module {}".format(module_name), flush=True)
            manager.curr_module = module_name
            visitor.dfs(modules_dict[module_name])
        base_store = {}
        # --- always_comb look-up table (Paper §4.4) ---
        # On first pass, evaluate decls and comb nodes; store results in a look-up
        # table so they can be reused at end of cycle without re-walking the AST.
        comb_lookup = {}  # module_name -> snapshot of combinational logic results
        for module_name in keys:
            print("Phase: base_store for module {}".format(module_name), flush=True)
            manager.curr_module = module_name
            for c in cfgs_by_module[module_name]:
                for node in c.decls:
                    visitor.dfs(node)
                for node in c.comb:
                    visitor.dfs(node)
            base_store[module_name] = state.snapshot(module_name)
            # Capture the combinational-logic result for this module
            comb_lookup[module_name] = state.snapshot(module_name)

        # Count sequential vs combinational CFGs for reporting
        n_seq = sum(1 for m in keys for c in cfgs_by_module[m] if not c.is_combinational)
        n_comb = sum(1 for m in keys for c in cfgs_by_module[m] if c.is_combinational)
        if n_comb > 0:
            print(f"  always_comb optimization: {n_comb} combinational CFGs, {n_seq} sequential CFGs", flush=True)
            print(f"  Look-up table built for {len(comb_lookup)} module(s)", flush=True)

        # --- Phase: Collect SVA assertions from all modules ---
        print("Phase: collecting SVA assertions", flush=True)
        manager.assertions = []
        for module_name in manager.names_list:
            module_ast = modules_dict.get(module_name, None)
            if module_ast is None:
                # For instance names like "mod_0", try the base module name
                base_name = module_name.rsplit('_', 1)[0] if '_' in module_name else module_name
                module_ast = modules_dict.get(base_name, None)
            if module_ast is not None:
                self._collect_assertions(module_ast, module_name, manager.assertions)
        # Evaluate assertion expressions symbolically using the base store
        for assertion_info in manager.assertions:
            amod = assertion_info['module']
            manager.curr_module = amod
            self._eval_assertion_expr(assertion_info, visitor, manager, state, modules_dict)
        n_total = len(manager.assertions)
        n_with_z3 = sum(1 for a in manager.assertions if a.get('z3_expr') is not None)
        print(f"  Found {n_total} assertion(s), {n_with_z3} with Z3 expressions", flush=True)

        print("Phase: explore_block (per-module)", flush=True)
        block_results = {}
        for module_name in keys:
            state_template = state.fresh_for_block(module_name, base_store[module_name])
            # List of generators (one per block); no per-block materialization
            block_results[module_name] = [
                self.explore_block(visitor, manager, state_template, module_name, cfg, modules_dict)
                for cfg in cfgs_by_module[module_name]
            ]

        print("Phase: merge_block_results (per-module, streaming)", flush=True)
        # No per-module materialization: merge is a generator; cross-module product uses factories
        def make_merge_gen(module_name):
            """Return a callable that yields a fresh merge generator for this module."""
            return lambda: self.merge_block_results_streaming(
                block_results[module_name], module_name, manager=manager)

        for idx, module_name in enumerate(keys):
            n_blocks = len(block_results[module_name])
            print(f"  merge {module_name} ({idx+1}/{len(keys)}): {n_blocks} blocks (streaming)", flush=True)

        print("Phase: cross-module path iteration (streaming)", flush=True)
        # Per-module: product of num_cycles fresh merge runs (no materialization)
        per_module_cycle_iter = [
            product(*(make_merge_gen(m)() for _ in range(int(num_cycles))))
            for m in keys
        ]
        path_combo_gen = product(*per_module_cycle_iter)
        path_combo_iteration = 0

        for path_combo in path_combo_gen:
            # Stop iterating cross-module combinations if a timeout was requested.
            if getattr(self, "timeout", False):
                break
            path_combo_iteration += 1
            if path_combo_iteration <= 5 or path_combo_iteration % 1000000 == 0:
                print("  path_combo iteration: {} (feasible so far: {})".format(
                    path_combo_iteration, manager.path_count), flush=True)
            curr_path = {keys[i]: path_combo[i] for i in range(len(keys))}

            all_pcs = []
            all_stores = {}
            var_sets = []
            for module_name in keys:
                for cycle_result in curr_path[module_name]:
                    all_pcs.extend(cycle_result["pc"])
                    var_sets.append(self._vars_in_pcs(cycle_result["pc"]))
                    for sig, expr in cycle_result["store"].items():
                        all_stores[f"{module_name}.{sig}"] = expr
            need_solver = False
            for i in range(len(var_sets)):
                for j in range(i + 1, len(var_sets)):
                    if var_sets[i] & var_sets[j]:
                        need_solver = True
                        break
                if need_solver:
                    break
            if need_solver and all_pcs:
                s = Solver()
                try:
                    s.set("timeout", 10000)
                except Exception:
                    pass
                for c in all_pcs:
                    s.add(c)
                if str(s.check()) != "sat":
                    continue
            manager.path_count += 1
            if manager.path_count <= 5 or manager.path_count % 10000 == 0:
                print("  path_combo: {} paths".format(manager.path_count), flush=True)

            # --- Assertion checking on this feasible path ---
            violation = self._check_assertions_on_path(
                manager, visitor, all_pcs, all_stores, modules_dict)
            if violation:
                print("Phase: cross-module path iteration complete (violation found).", flush=True)
                print("Branch points explored: {}".format(manager.branch_count), flush=True)
                print("Paths explored (feasible): {}".format(manager.path_count), flush=True)
                print("Paths explored: {}".format(manager.path_count), flush=True)
                self.module_depth -= 1
                return

        print("Phase: cross-module path iteration complete.", flush=True)
        print("Branch points explored: {}".format(manager.branch_count), flush=True)
        print("Paths explored (feasible): {}".format(manager.path_count), flush=True)
        print("Paths explored: {}".format(manager.path_count), flush=True)
        self.module_depth -= 1

    def _check_assertions_on_path(self, manager, visitor, all_pcs, all_stores, modules_dict):
        """Check all collected SVA assertions against a feasible merged path.

        For each assertion in manager.assertions, negate the assertion expression
        and check if (PC AND NOT(assertion)) is SAT.  If SAT, the assertion is
        violated: extract and print a counterexample and return True.
        Returns False if no assertion is violated on this path.
        """
        if not manager.assertions:
            return False

        for assertion_info in manager.assertions:
            assertion_z3 = assertion_info.get("z3_expr")
            if assertion_z3 is None:
                continue
            # Build a solver with the full path condition
            s = Solver()
            try:
                s.set("timeout", 10000)
            except Exception:
                pass
            for c in all_pcs:
                s.add(c)
            # Negate the assertion: if SAT, the assertion can be violated
            from z3 import Not, is_bool
            if not is_bool(assertion_z3):
                continue
            s.push()
            s.add(Not(assertion_z3))
            result = str(s.check())
            if result == "sat":
                # Assertion violation found -- extract counterexample
                model = s.model()
                symbols_to_values = {}
                for d in model.decls():
                    symbols_to_values[d.name()] = model[d]
                counterexample = {}
                for qualified_signal, expr in all_stores.items():
                    expr_str = str(expr)
                    if expr_str in symbols_to_values:
                        counterexample[qualified_signal] = symbols_to_values[expr_str]
                source = assertion_info.get("source", "<unknown>")
                module = assertion_info.get("module", "<unknown>")
                print(f"\n=== ASSERTION VIOLATION FOUND ===", flush=True)
                print(f"  Module: {module}", flush=True)
                print(f"  Source: {source}", flush=True)
                print(f"  Counterexample: {counterexample}", flush=True)
                print(f"  Solver time: {manager.solver_time:.4f}s", flush=True)
                print(f"=================================\n", flush=True)
                manager.assertion_violation = True
                return True
            s.pop()
        return False

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
                
