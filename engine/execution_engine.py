# Main execution engine that orchestrates symbolic execution of SystemVerilog designs

from z3 import Solver, ExprRef
from z3 import z3util
from .execution_manager import ExecutionManager
from .symbolic_state import SymbolicState
from .cfg import CFG
from .dfs_iterator import DFSMergeIterator, DFSCrossModuleIterator, partition_blocks, LazyProduct
from typing import Optional, List
from functools import reduce
from operator import mul
import time
import gc
from helpers.utils import to_binary
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

    def collect_all_instances(self, instance: ps.Symbol, out: list) -> None:
        """Recursively collect this Instance symbol and all nested sub-instances depth-first."""
        out.append(instance)
        body = getattr(instance, 'body', getattr(instance, 'instanceBody', None))
        if body is None:
            return
        for member in body:
            if member.kind == ps.SymbolKind.Instance:
                self.collect_all_instances(member, out)



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
                      module_name: str, cfg: CFG, modules_dict: dict) -> List[dict]:
        """Explore all paths through a single always block (Paper §3.3). Returns a list of
        {'pc': list of z3 constraints, 'store': {signal: expr}} per feasible path."""
        prev_curr_module = manager.curr_module
        manager.curr_module = module_name
        num_paths = cfg.get_path_count()
        print("explore_block: {} ({} paths)".format(module_name, num_paths), flush=True)
        results = []
        try:
            for path_idx, path in enumerate(cfg.get_paths()):
                if path_idx == 0 or (path_idx + 1) % 100 == 0 or path_idx == num_paths - 1:
                    print("  path {}/{}".format(path_idx + 1, num_paths), flush=True)
                manager.ignore = False
                manager.abandon = False
                path_state = state_template.fresh_for_block(module_name,
                    state_template.snapshot(module_name))

                self._execute_cfg_path(visitor, manager, path_state, module_name,
                                       cfg, path, modules_dict)

                try:
                    constraints = list(path_state.pc.assertions())
                except Exception:
                    constraints = []
                try:
                    path_state.pc.set("timeout", 10000)
                except Exception:
                    pass
                if not constraints or str(path_state.pc.check()) == "sat":
                    results.append({
                        "pc": constraints,
                        "store": dict(path_state.store.get(module_name, {}))
                    })
        finally:
            manager.curr_module = prev_curr_module
            if hasattr(manager, '_pc_ref'):
                manager._pc_ref = None
        return results

    def _execute_cfg_path(self, visitor, manager, path_state, module_name, cfg, path, modules_dict):
        """Execute a single CFG path: add edge constraints to pc and run assignment statements."""

        prev_bb_idx = None
        for bb_idx in path:
            if bb_idx == -1:
                prev_bb_idx = bb_idx
                continue
            if bb_idx == -2:
                # Exit node: handle trailing condition from previous condition BB
                if prev_bb_idx is not None and prev_bb_idx >= 0:
                    edge_data = cfg.graph.get_edge_data(prev_bb_idx, -2)
                    if edge_data:
                        self._assert_edge_condition(
                            edge_data, cfg, prev_bb_idx, path_state, manager)
                break

            # Add constraint from the incoming edge
            if prev_bb_idx is not None:
                edge_data = cfg.graph.get_edge_data(prev_bb_idx, bb_idx)
                if edge_data:
                    self._assert_edge_condition(
                        edge_data, cfg, prev_bb_idx, path_state, manager)

            # Execute statements in this BB (skip bare condition Expression nodes)
            basic_block = cfg.basic_block_list[bb_idx]
            for stmt in basic_block:
                if isinstance(stmt, ps.Expression):
                    continue
                manager._pc_ref = path_state.pc
                try:
                    visitor.visit_stmt(manager, path_state, stmt, modules_dict, 1)
                finally:
                    manager._pc_ref = None

            prev_bb_idx = bb_idx

    @staticmethod
    def _to_bool(z3_expr):
        """Coerce a Z3 expression to BoolRef. BitVec values become (val != 0)."""
        from z3 import BoolRef, BitVecRef, BitVecVal, ArithRef, IntVal
        if isinstance(z3_expr, BoolRef):
            return z3_expr
        if isinstance(z3_expr, BitVecRef):
            return z3_expr != BitVecVal(0, z3_expr.size())
        if isinstance(z3_expr, ArithRef):
            return z3_expr != IntVal(0)
        return z3_expr != 0

    def _assert_edge_condition(self, edge_data, cfg, source_bb_idx, path_state, manager):
        """Add a Z3 constraint to path_state.pc based on a CFG edge condition."""
        import z3
        from z3 import Not, Or, And
        from helpers.rvalue_to_z3 import semantic_expr_to_z3
        from engine.basic_block_visitor import CaseLabel, DefaultLabel

        cond = edge_data.get('condition')
        if cond is None:
            return

        if source_bb_idx < 0:
            return
        source_bb = cfg.basic_block_list[source_bb_idx]
        cond_expr_node = None
        for node in source_bb:
            if isinstance(node, ps.Expression):
                cond_expr_node = node
                break
        if cond_expr_node is None:
            return

        module = manager.curr_module
        store = path_state.store.get(module, {})
        cond_z3 = semantic_expr_to_z3(cond_expr_node, store, module)
        if cond_z3 is None:
            return

        from z3 import BoolRef, BitVecRef, ArithRef
        if not isinstance(cond_z3, (BoolRef, BitVecRef, ArithRef)):
            return

        path_state.assertion_counter += 1
        tag = "cfg_p{}".format(path_state.assertion_counter)

        try:
            if cond == 'true':
                path_state.pc.assert_and_track(self._to_bool(cond_z3), tag)
            elif cond == 'false':
                path_state.pc.assert_and_track(Not(self._to_bool(cond_z3)), tag)
            elif isinstance(cond, CaseLabel):
                item_z3s = []
                for item_expr in cond:
                    item_z3 = semantic_expr_to_z3(item_expr, store, module)
                    if item_z3 is not None and isinstance(item_z3, (BoolRef, BitVecRef, ArithRef)):
                        if isinstance(cond_z3, BitVecRef) and isinstance(item_z3, BitVecRef):
                            if cond_z3.size() != item_z3.size():
                                tgt = max(cond_z3.size(), item_z3.size())
                                c = cond_z3 if cond_z3.size() == tgt else z3.ZeroExt(tgt - cond_z3.size(), cond_z3)
                                i = item_z3 if item_z3.size() == tgt else z3.ZeroExt(tgt - item_z3.size(), item_z3)
                                item_z3s.append(c == i)
                                continue
                        item_z3s.append(cond_z3 == item_z3)
                if item_z3s:
                    constraint = item_z3s[0] if len(item_z3s) == 1 else Or(*item_z3s)
                    path_state.pc.assert_and_track(constraint, tag)
            elif isinstance(cond, DefaultLabel):
                neg_z3s = []
                for item_expr in cond.get('default_from', []):
                    item_z3 = semantic_expr_to_z3(item_expr, store, module)
                    if item_z3 is not None and isinstance(item_z3, (BoolRef, BitVecRef, ArithRef)):
                        if isinstance(cond_z3, BitVecRef) and isinstance(item_z3, BitVecRef):
                            if cond_z3.size() != item_z3.size():
                                tgt = max(cond_z3.size(), item_z3.size())
                                c = cond_z3 if cond_z3.size() == tgt else z3.ZeroExt(tgt - cond_z3.size(), cond_z3)
                                i = item_z3 if item_z3.size() == tgt else z3.ZeroExt(tgt - item_z3.size(), item_z3)
                                neg_z3s.append(c != i)
                                continue
                        neg_z3s.append(cond_z3 != item_z3)
                    pass
                if neg_z3s:
                    path_state.pc.assert_and_track(And(*neg_z3s), tag)
        except Exception:
            pass

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
        if cname in ('ImmediateAssertStatementSyntax', 'ImmediateAssertionStatementSyntax',
                      'ImmediateAssertionMemberSyntax',
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

    def _collect_procedural_assertions(self, always_blocks, module_name, assertions_list):
        """Walk the semantic statement trees of always blocks to find immediate assertions.

        PySlang represents ``assert(expr)`` inside procedural blocks as
        ``ImmediateAssertionStatement`` (StatementKind.ImmediateAssertion).
        The module-level syntax walker in ``_collect_assertions`` misses these
        because they live inside procedural block bodies, not at module scope.
        """
        found = []

        def _visitor(node):
            if getattr(node, 'kind', None) == ps.StatementKind.ImmediateAssertion:
                cond = getattr(node, 'cond', None)
                source_range = getattr(node, 'sourceRange', None)
                source = str(source_range) if source_range else '<unknown>'
                found.append({
                    'node': node,
                    'module': module_name,
                    'source': source,
                    'kind': 'immediate',
                    'cond_expr': cond,
                    'z3_expr': None,
                })

        for ab in always_blocks:
            ab_body = getattr(ab, 'body', getattr(ab, 'statement', None))
            if ab_body is not None:
                ab_body.visit(_visitor)

        assertions_list.extend(found)

    def _eval_assertion_expr(self, assertion_info, visitor, manager, state, modules_dict):
        """Attempt to evaluate the assertion node's condition/expression into a Z3 expression.

        Tries multiple attribute names to locate the assertion condition in the
        pyslang AST node, then uses the visitor's expr_to_z3 to convert it.
        Sets assertion_info['z3_expr'] if successful.
        """
        node = assertion_info['node']
        expr_node = assertion_info.get('cond_expr')

        if expr_node is None:
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

    def merge_block_results(self, block_result_lists: List[list], module_name: str = "",
                            manager=None):
        """Piecewise composition merge step (Paper §3.3, §4.3).
        
        Combines path fragments from always blocks via SMT while preserving completeness.
        
        Pipeline:
        1. Partition blocks into connected components (union-find on shared variables)
        2. Within each component: DFS merge with early pruning, slice → normalize → cache
        3. Across components: LazyProduct (no materialization, lazy Cartesian product)
        
        Returns a LazyProduct (iterable, supports len()) or a plain list."""
        if not block_result_lists:
            return [{"pc": [], "store": {}}]

        groups = partition_blocks(block_result_lists)
        n_groups = len(groups)
        
        if n_groups > 1:
            total_product = 1
            for bl in block_result_lists:
                total_product *= max(len(bl), 1)
            print(f"    merge {module_name}: partitioned {len(block_result_lists)} blocks "
                  f"into {n_groups} independent groups (full product would be {total_product:,})",
                  flush=True)
        
        component_results: List[List[dict]] = []
        for group_idx, block_indices in enumerate(groups):
            group_block_lists = [block_result_lists[i] for i in block_indices]
            sizes = [max(len(bl), 1) for bl in group_block_lists]
            group_naive = reduce(mul, sizes, 1)
            if len(group_block_lists) > 1:
                print(
                    f"    merge {module_name} group {group_idx}: block indices {block_indices}, "
                    f"per-block outcome counts {sizes}, naive Cartesian size {group_naive:,}",
                    flush=True,
                )
            
            if len(group_block_lists) == 1:
                component_results.append(group_block_lists[0])
            else:
                dfs_iter = DFSMergeIterator(
                    block_result_lists=group_block_lists,
                    module_name=f"{module_name}_g{group_idx}",
                    manager=manager,
                    enable_early_pruning=True,
                    enable_caching=True,
                )
                group_results = []
                for idx, merged in enumerate(dfs_iter):
                    if idx > 0 and idx % 100000 == 0:
                        stats = dfs_iter.get_stats()
                        print(f"    merge {module_name} group {group_idx}: yielded {idx}, "
                              f"checked {stats['combos_checked']}, "
                              f"pruned {stats['combos_pruned']}", flush=True)
                    group_results.append(merged)
                
                stats = dfs_iter.get_stats()
                if stats['combos_pruned'] > 0:
                    print(f"    merge {module_name} group {group_idx}: DFS pruned "
                          f"{stats['combos_pruned']} (cache hits: {stats['cache_hits']})",
                          flush=True)
                component_results.append(group_results)
        
        if n_groups == 1:
            return component_results[0]
        
        lazy = LazyProduct(component_results)
        print(f"    merge {module_name}: LazyProduct of {n_groups} groups, "
              f"sizes={lazy.component_sizes}, stored={lazy.total_stored()}, "
              f"logical_size={lazy.logical_size:,}", flush=True)
        return lazy

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
        cfgs_by_module = {}
        always_blocks_by_module = {}

        all_instances = []
        for top in modules:
            self.collect_all_instances(top, all_instances)

        for instance in all_instances:
            instance_name = instance.name
            body = getattr(instance, 'body', getattr(instance, 'instanceBody', None))

            modules_dict[instance_name] = instance
            manager.names_list.append(instance_name)
            manager.seen_mod[instance_name] = {}
            cfgs_by_module[instance_name] = []

            probe = CFG()
            probe.get_always_sv(body)
            always_blocks_by_module[instance_name] = (
                list(probe.always_blocks) + list(probe.always_comb_blocks)
            )

            for ab in probe.always_blocks:
                ab_body = getattr(ab, 'body', getattr(ab, 'statement', None))
                c = CFG()
                c._instance_body = probe._instance_body
                c.module_name = instance_name
                c.is_combinational = False
                c.build_cfg(ab_body)
                cfgs_by_module[instance_name].append(c)

            for ab in probe.always_comb_blocks:
                ab_body = getattr(ab, 'body', getattr(ab, 'statement', None))
                c = CFG()
                c._instance_body = probe._instance_body
                c.module_name = instance_name
                c.is_combinational = True
                c.build_cfg(ab_body)
                cfgs_by_module[instance_name].append(c)

            # Copy decls/comb from probe to the first CFG so downstream
            # base_store initialisation picks them up.
            if cfgs_by_module[instance_name]:
                cfgs_by_module[instance_name][0].decls = probe.decls
                cfgs_by_module[instance_name][0].comb = probe.comb

            state.store[instance_name] = {}
            manager.dependencies[instance_name] = {}
            manager.intermodule_dependencies[instance_name] = {}
            manager.cond_assigns[instance_name] = {}

        manager.debug = self.debug

        if len(all_instances) > 1:
            self.populate_seen_mod(manager)
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
                    if hasattr(node, 'name') and node.name not in state.store[module_name]:
                        state.store[module_name][node.name] = node.name
                for node in c.comb:
                    visitor.dfs(node)
            n_signals = len(state.store[module_name])
            if n_signals:
                print("  {} signal(s) initialized in store".format(n_signals), flush=True)
            base_store[module_name] = state.snapshot(module_name)
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
            # 1) Module-level syntax walk (concurrent assertions, SVA properties)
            module_sym = modules_dict.get(module_name, None)
            if module_sym is None:
                base_name = module_name.rsplit('_', 1)[0] if '_' in module_name else module_name
                module_sym = modules_dict.get(base_name, None)
            if module_sym is not None:
                module_ast = getattr(module_sym, 'syntax', module_sym)
                self._collect_assertions(module_ast, module_name, manager.assertions)
            # 2) Procedural-block walk (immediate assertions inside always blocks)
            ab_list = always_blocks_by_module.get(module_name, [])
            if ab_list:
                self._collect_procedural_assertions(ab_list, module_name, manager.assertions)

        for assertion_info in manager.assertions:
            amod = assertion_info['module']
            manager.curr_module = amod
            self._eval_assertion_expr(assertion_info, visitor, manager, state, modules_dict)
        n_total = len(manager.assertions)
        n_with_z3 = sum(1 for a in manager.assertions if a.get('z3_expr') is not None)
        n_immediate = sum(1 for a in manager.assertions if a.get('kind') == 'immediate')
        n_concurrent = n_total - n_immediate
        print(f"  Found {n_total} assertion(s) ({n_immediate} immediate, {n_concurrent} concurrent), "
              f"{n_with_z3} with Z3 expressions", flush=True)

        print("Phase: explore_block (per-module)", flush=True)
        # Per paper §3.3: explore full path tree per always block; materialize list per block
        block_results = {}
        for module_name in keys:
            state_template = state.fresh_for_block(module_name, base_store[module_name])
            block_results[module_name] = [
                self.explore_block(visitor, manager, state_template, module_name, cfg, modules_dict)
                for cfg in cfgs_by_module[module_name]
            ]

        print("Phase: merge_block_results (per-module, Paper §4.3)", flush=True)
        merged_by_module = {}
        for idx, module_name in enumerate(keys):
            n_blocks = len(block_results[module_name])
            print(f"  merge {module_name} ({idx+1}/{len(keys)}): {n_blocks} blocks", flush=True)
            merged_by_module[module_name] = self.merge_block_results(
                block_results[module_name], module_name, manager=manager)
            merged = merged_by_module[module_name]
            if isinstance(merged, LazyProduct):
                print(f"    -> {merged.logical_size:,} feasible merged paths "
                      f"(lazy: {merged.n_components} groups, {merged.total_stored()} stored)",
                      flush=True)
            else:
                print(f"    -> {len(merged)} feasible merged paths", flush=True)

        valid_assertions = [a for a in manager.assertions
                            if a.get("z3_expr") is not None]
        if not valid_assertions:
            print("Phase: cross-module path iteration — SKIPPED (no assertions to check)",
                  flush=True)
            print(f"  {len(manager.assertions)} assertion(s) collected, "
                  f"0 with valid Z3 expressions", flush=True)
            print(f"  Per-module merged results available for "
                  f"{len(merged_by_module)} modules.", flush=True)
            total_logical = 1
            for m, res in merged_by_module.items():
                count = res.logical_size if isinstance(res, LazyProduct) else len(res)
                total_logical *= max(count, 1)
            print(f"  Cross-module product would have been {total_logical:,} paths — "
                  f"all skipped.", flush=True)
            print("Branch points explored: {}".format(manager.branch_count), flush=True)
            print("Paths explored (feasible): 0 (no assertions)", flush=True)
            self.module_depth -= 1
            return

        print("Phase: cross-module path iteration (DFS)", flush=True)
        print(f"  Checking {len(valid_assertions)} assertion(s)", flush=True)
        dfs_xmod = DFSCrossModuleIterator(
            per_module_results=merged_by_module,
            num_cycles=int(num_cycles),
            manager=manager,
            enable_early_pruning=True,
            enable_caching=True,
        )
        
        path_combo_iteration = 0
        for path_combo, all_pcs, all_stores in dfs_xmod:
            if getattr(self, "timeout", False):
                break
            path_combo_iteration += 1
            if path_combo_iteration <= 5 or path_combo_iteration % 1000000 == 0:
                stats = dfs_xmod.get_stats()
                print("  path_combo iteration: {} (feasible so far: {}, pruned: {})".format(
                    path_combo_iteration, manager.path_count, stats['combos_pruned']), flush=True)
            
            manager.path_count += 1
            if manager.path_count <= 5 or manager.path_count % 10000 == 0:
                print("  path_combo: {} paths".format(manager.path_count), flush=True)

            violation = self._check_assertions_on_path(
                manager, visitor, all_pcs, all_stores, modules_dict)
            if violation:
                stats = dfs_xmod.get_stats()
                print("Phase: cross-module path iteration complete (violation found).", flush=True)
                print("  DFS stats: checked {}, pruned {}, cache hits {}".format(
                    stats['combos_checked'], stats['combos_pruned'], stats['cache_hits']), flush=True)
                print("Branch points explored: {}".format(manager.branch_count), flush=True)
                print("Paths explored (feasible): {}".format(manager.path_count), flush=True)
                self.module_depth -= 1
                return
        
        stats = dfs_xmod.get_stats()
        print("Phase: cross-module path iteration complete.", flush=True)
        print("  DFS stats: checked {}, pruned {}, cache hits {}".format(
            stats['combos_checked'], stats['combos_pruned'], stats['cache_hits']), flush=True)
        print("Branch points explored: {}".format(manager.branch_count), flush=True)
        print("Paths explored (feasible): {}".format(manager.path_count), flush=True)
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
                
