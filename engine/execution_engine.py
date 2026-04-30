# Main execution engine that orchestrates symbolic execution of SystemVerilog designs

from z3 import Solver, ExprRef
from z3 import z3util
from .execution_manager import ExecutionManager
from .symbolic_state import SymbolicState
from .cfg import CFG
from .combinational import build_comb_metadata, evaluate_dirty_comb
from .dfs_iterator import (
    DFSCrossModuleIterator,
    LazyProduct,
    ReplayableMergeResults,
    partition_blocks,
)
from typing import Optional, List
import os
import time
import gc
from math import prod
from helpers.utils import to_binary
import pyslang.syntax as ps_stx
import pyslang.ast as ps_ast
from helpers.slang_helpers import get_module_name, init_state

# Tuple of PySlang AST node types that represent conditional/loop statements
CONDITIONALS = (
    ps_stx.ConditionalStatementSyntax,
    ps_stx.CaseStatementSyntax,
    ps_stx.ForeachLoopStatementSyntax,
    ps_stx.ForLoopStatementSyntax,
    ps_stx.LoopStatementSyntax,
    ps_stx.DoWhileStatementSyntax
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

    def collect_all_instances(self, instance: ps_ast.Symbol, out: list) -> None:
        """Recursively collect this Instance symbol and all nested sub-instances depth-first."""
        out.append(instance)
        body = getattr(instance, 'body', getattr(instance, 'instanceBody', None))
        if body is None:
            return
        for member in body:
            if member.kind == ps_ast.SymbolKind.Instance:
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
                if self.debug and (
                    path_idx == 0 or (path_idx + 1) % 100 == 0 or path_idx == num_paths - 1
                ):
                    print("  path {}/{}".format(path_idx + 1, num_paths), flush=True)
                manager.ignore = False
                manager.abandon = False
                path_state = state_template.fresh_for_block(module_name,
                    state_template.snapshot(module_name))

                # base_store already has CA outputs baked in (one-shot pass in
                # execute_sv). Starts clean — only writes during the path dirty
                # their dependents, and evaluate_dirty_comb re-runs just those.

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
                if isinstance(stmt, ps_ast.Expression):
                    continue
                manager._pc_ref = path_state.pc
                try:
                    visitor.visit_stmt(manager, path_state, stmt, modules_dict, 1)
                finally:
                    manager._pc_ref = None

            prev_bb_idx = bb_idx

        # End-of-path NBA commit, then re-evaluate every continuous assign whose
        # RHS depends on a signal that was written during the path (paper §4.4).
        path_state.flush_pending_nba(module_name, manager)
        evaluate_dirty_comb(path_state, module_name, manager)

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
        from helpers.rvalue_to_z3 import semantic_expr_to_z3, case_statement_arm_matches_z3
        from engine.basic_block_visitor import CaseLabel, DefaultLabel

        cond = edge_data.get('condition')
        if cond is None:
            return

        if source_bb_idx < 0:
            return

        cond_expr_node = None

        # Exact guard node carried by CFG edge metadata.
        guard_node_idx = edge_data.get('guard_node_idx')
        if isinstance(guard_node_idx, int) and 0 <= guard_node_idx < len(cfg.all_nodes):
            candidate = cfg.all_nodes[guard_node_idx]
            if isinstance(candidate, ps_ast.Expression):
                cond_expr_node = candidate

        # Fallback: for older CFGs / missing metadata
        if cond_expr_node is None:
            source_bb = cfg.basic_block_list[source_bb_idx]
            for node in source_bb:
                if isinstance(node, ps_ast.Expression):
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

        # Temp debug: map cfg_pN -> edge/source/condition for one module
        if manager.curr_module == "or1200_dpram_256x32":
            try:
                print(f"[CFG_ASSERT] {tag} src_bb={source_bb_idx} guard_idx={guard_node_idx} cond={cond} cond_z3={cond_z3}", flush=True)
            except Exception:
                pass
        
        try:
            if cond == 'true':
                path_state.pc.assert_and_track(self._to_bool(cond_z3), tag)
            elif cond == 'false':
                path_state.pc.assert_and_track(Not(self._to_bool(cond_z3)), tag)
            elif isinstance(cond, CaseLabel):
                item_z3s = []
                case_kind = getattr(cond, "case_kind", "") or ""
                for item_expr in cond:
                    if isinstance(cond_z3, BitVecRef):
                        arm = case_statement_arm_matches_z3(
                            cond_z3, item_expr, store, module, case_kind
                        )
                        if arm is not None:
                            item_z3s.append(arm)
                            continue
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
                case_kind = getattr(cond, "case_kind", "") or ""
                for item_expr in cond.get('default_from', []):
                    if isinstance(cond_z3, BitVecRef):
                        arm = case_statement_arm_matches_z3(
                            cond_z3, item_expr, store, module, case_kind
                        )
                        if arm is not None:
                            neg_z3s.append(Not(arm))
                            continue
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

    @staticmethod
    def _format_source_range(sr) -> str:
        """Human-readable location for pyslang SourceRange or string."""
        if sr is None:
            return "<unknown>"
        if isinstance(sr, str):
            return sr
        try:
            start = getattr(sr, "start", None)
            if start is not None:
                line = getattr(start, "line", None)
                col = getattr(start, "column", None)
                buf = getattr(start, "buffer", None)
                path = ""
                if buf is not None:
                    path = getattr(buf, "name", None) or getattr(buf, "file", None) or ""
                    if path is not None and not isinstance(path, str):
                        path = str(path)
                if line is not None:
                    loc = f"{path}:{line}" if path else f"line {line}"
                    if col is not None:
                        loc += f":{col}"
                    return loc
        except Exception:
            pass
        try:
            return str(sr)
        except Exception:
            return repr(sr)

    def _collect_assertions(self, ast, module_name, assertions_list):
        """Recursively walk a pyslang AST and collect SVA assertion nodes.

        Populates assertions_list with dicts containing:
          - 'node': the raw AST node
          - 'module': the module name the assertion belongs to
          - 'source': source location string for reporting
          - 'source_range': raw SourceRange when available (better diagnostics)
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
            sr = getattr(ast, "sourceRange", None)
            if callable(getattr(ast, "toString", None)) and sr is None:
                try:
                    sr = ast.toString()
                except Exception:
                    sr = None
            source = self._format_source_range(sr)
            assertions_list.append({
                'node': ast,
                'module': module_name,
                'source': source,
                'source_range': sr,
                'kind': 'immediate',
                'z3_expr': None,  # filled during explore_block or cross-module check
            })
            return

        # Check for concurrent assertion statements: assert property (...)
        if cname in ('AssertPropertyStatementSyntax', 'ConcurrentAssertionMemberSyntax',
                      'AssumePropertyStatementSyntax', 'CoverPropertyStatementSyntax',
                      'ExpectPropertyStatementSyntax'):
            sr = getattr(ast, "sourceRange", None)
            source = self._format_source_range(sr)
            assertions_list.append({
                'node': ast,
                'module': module_name,
                'source': source,
                'source_range': sr,
                'kind': 'concurrent',
                'z3_expr': None,
            })
            return

        # Recurse into known container types
        if isinstance(ast, ps_stx.ModuleDeclarationSyntax):
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
            if hasattr(members, '__iter__') and not isinstance(ast, ps_stx.ModuleDeclarationSyntax):
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
            if getattr(node, 'kind', None) == ps_ast.StatementKind.ImmediateAssertion:
                cond = getattr(node, 'cond', None)
                source_range = getattr(node, 'sourceRange', None)
                source = ExecutionEngine._format_source_range(source_range)
                found.append({
                    'node': node,
                    'module': module_name,
                    'source': source,
                    'source_range': source_range,
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
            logging.warning(
                "Assertion Z3 eval failed module=%s source=%s: %s",
                assertion_info.get("module"),
                assertion_info.get("source"),
                e,
            )

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
        2. Within each component: ReplayableMergeResults streams DFSMergeIterator (not materialized)
        3. Across components: LazyProduct nests iteration (lazy Cartesian product)
        
        Returns a LazyProduct, ReplayableMergeResults, or plain list (all iterable); merge
        is not materialized until consumers iterate (DFS streaming)."""
        if not block_result_lists:
            return [{"pc": [], "store": {}}]

        groups = partition_blocks(block_result_lists)
        n_groups = len(groups)

        component_results: List = []
        for group_idx, block_indices in enumerate(groups):
            group_block_lists = [block_result_lists[i] for i in block_indices]

            if len(group_block_lists) == 1:
                component_results.append(group_block_lists[0])
            else:
                component_results.append(
                    ReplayableMergeResults(
                        block_result_lists=group_block_lists,
                        module_name=f"{module_name}_g{group_idx}",
                        manager=manager,
                    )
                )
        
        if n_groups == 1:
            return component_results[0]

        return LazyProduct(
            component_results,
            manager=manager,
            solver_timeout_ms=10000,
        )

    def execute_sv(
        self,
        visitor,
        modules,
        manager: Optional[ExecutionManager],
        num_cycles: int,
        *,
        max_cross_module_paths: Optional[int] = None,
    ) -> None:
        """Main entry point for PySlang execution
        Drives symbolic execution for SystemVerilog designs."""
        gc.collect()
        print(f"Executing for {num_cycles} clock cycle(s)", flush=True)
        self.module_depth += 1
        state: SymbolicState = SymbolicState()

        state.pending_nba = {}

        if manager is not None:
            # Only manager=None is supported: we build modules_dict, cfgs_by_module, etc. here.
            raise ValueError("execute_sv requires manager=None; the engine creates the manager internally.")
        manager = ExecutionManager()
        if hasattr(self, "cache"):
            manager.cache = self.cache
        manager.sv = True
        manager.max_cross_module_paths = max_cross_module_paths
        manager.cross_module_stopped_reason = ""
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
            print("No modules or no CFGs; skipping path exploration.", flush=True)
            self.module_depth -= 1
            return

        first_module = modules_dict[manager.names_list[0]] if manager.names_list else None

        # --- Piecewise composition (sole execution mode) ---
        manager.prev_store = state.store
        print("Phase: init_state", flush=True)
        init_state(state, manager.prev_store, first_module, visitor)
        for module_name in manager.names_list:
            manager.curr_module = module_name
            visitor.dfs(modules_dict[module_name])
        print(f"Phase: hierarchy traversal — {len(manager.names_list)} module(s)", flush=True)
        base_store = {}
        # --- always_comb look-up table (Paper §4.4) ---
        # On first pass, evaluate decls and comb nodes; store results in a look-up
        # table so they can be reused at end of cycle without re-walking the AST.
        comb_lookup = {}  # module_name -> snapshot of combinational logic results
        for module_name in keys:
            manager.curr_module = module_name
            for c in cfgs_by_module[module_name]:
                for node in c.decls:
                    visitor.dfs(node)
                    if hasattr(node, 'name') and node.name not in state.store[module_name]:
                        state.store[module_name][node.name] = node.name
                for node in c.comb:
                    visitor.dfs(node)
            base_store[module_name] = state.snapshot(module_name)
            comb_lookup[module_name] = state.snapshot(module_name)

        print(f"Phase: base_store — {len(keys)} module(s)", flush=True)

        # --- Continuous-assign dependency map (paper §4.4) ---
        # Per module: list of ContinuousAssign nodes, {idx→lhs}, {rhs_signal→[idx,…]}.
        # Writes consult comb_deps to OR dirty bits; evaluate_dirty_comb drains them
        # at end of each path.
        # TODO Param look at this section down to line # 746
        manager.comb_assigns = {}
        manager.comb_lhs = {}
        manager.comb_deps = {}
        for module_name in keys:
            cfgs = cfgs_by_module[module_name]
            comb_nodes = cfgs[0].comb if cfgs else []
            assigns, lhs, deps = build_comb_metadata(comb_nodes)
            manager.comb_assigns[module_name] = assigns
            manager.comb_lhs[module_name] = lhs
            manager.comb_deps[module_name] = deps
        total_ca = sum(len(v) for v in manager.comb_assigns.values())
        if total_ca > 0:
            print(
                f"  continuous-assign dep map: {total_ca} CAs across {len(keys)} module(s)",
                flush=True,
            )

        # One-shot: evaluate every CA against each module's base_store so path
        # starts inherit CA outputs baked in. Without this, each path would have
        # to re-evaluate every CA itself (paths × CAs Z3 builds per module).
        from types import SimpleNamespace
        for module_name in keys:
            n_ca = len(manager.comb_assigns[module_name])
            if n_ca == 0:
                continue
            seed = SimpleNamespace(
                store={module_name: base_store[module_name]},
                dirty={module_name: (1 << n_ca) - 1}, # Marks all CAs as dirty so they get evaluated
            )
            evaluate_dirty_comb(seed, module_name, manager)

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

        print("Phase: explore_block", flush=True)
        # Per paper §3.3: explore full path tree per always block; materialize list per block.
        #
        # RTL timestep (ideal): one shared register snapshot for the posedge; every
        # always_ff/always active region runs (reads see pre-NBA values); all ``<=`` are
        # collected; one NBA commit updates every driven reg before the next timestep.
        #
        # Current model: each CFG path ends with ``flush_pending_nba`` (see
        # ``SymbolicState``) so ``<=`` within one always block matches single-block NBA;
        # different always blocks in the same module still start each ``explore_block``
        # from the same ``base_store`` snapshot (parallel same-cycle reads) and merge
        # later—there is no second global NBA pass across blocks here.
        #
        # CFG order: sequential (clocked) blocks before ``always_comb`` so comb reads
        # see a stable ordering when results are merged downstream.
        block_results = {}
        for module_name in keys:
            state_template = state.fresh_for_block(module_name, base_store[module_name])
            cfgs = list(cfgs_by_module[module_name])
            cfgs.sort(key=lambda c: (1 if getattr(c, "is_combinational", False) else 0,))
            block_results[module_name] = [
                self.explore_block(visitor, manager, state_template, module_name, cfg, modules_dict)
                for cfg in cfgs
            ]

        print("Phase: merge_block_results", flush=True)
        merge_prof = os.environ.get("SYLQ_MERGE_PROFILE", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if merge_prof:
            _mp_wall0 = time.monotonic()
            _mp_solver0 = manager.solver_time
            _mp_zm0 = manager.feasibility_z3_at_merge
            _mp_zlp0 = manager.feasibility_z3_at_lazy_product
            print(
                "  [merge-profile] per-module lines print after each instance; "
                "cumulative line prints after all merges (export SYLQ_MERGE_PROFILE=1).",
                flush=True,
            )
        merged_by_module = {}
        for idx, module_name in enumerate(keys):
            if merge_prof:
                _mod_wall0 = time.monotonic()
                _mod_solver0 = manager.solver_time
                _mod_zm0 = manager.feasibility_z3_at_merge
                _mod_zlp0 = manager.feasibility_z3_at_lazy_product
            n_blocks = len(block_results[module_name])
            merged_by_module[module_name] = self.merge_block_results(
                block_results[module_name], module_name, manager=manager)
            merged = merged_by_module[module_name]
            explore_sizes = [len(br) for br in block_results[module_name]]
            explorer_product = prod(explore_sizes) if explore_sizes else 1
            max_log_product = int(os.environ.get("SYLQ_MERGE_LOG_MAX_PRODUCT", "100000"))
            # merge_block_results returns a lazy iterator; optional path counting below
            # walks the full intra-module merge (Z3). Large LazyProduct×k dominates gaps
            # between the first line and the "→ N feasible merged paths" line.
            if isinstance(merged, LazyProduct):
                merge_kind = f"LazyProduct×{merged.n_components}"
            elif isinstance(merged, ReplayableMergeResults):
                merge_kind = "lazy merge"
            else:
                merge_kind = ""

            if merge_kind:
                print(
                    f"  merge {module_name} ({idx+1}/{len(keys)}): {n_blocks} blocks → {merge_kind}",
                    flush=True,
                )
                if explorer_product <= max_log_product:
                    n_paths = sum(1 for _ in merged)
                    print(
                        f"      → {n_paths} feasible merged paths",
                        flush=True,
                    )
            else:
                print(
                    f"  merge {module_name} ({idx+1}/{len(keys)}): {n_blocks} blocks → {len(merged)} paths",
                    flush=True,
                )

            if merge_prof:
                mod_wall = time.monotonic() - _mod_wall0
                mod_z3 = manager.solver_time - _mod_solver0
                mod_zm = manager.feasibility_z3_at_merge - _mod_zm0
                mod_zlp = manager.feasibility_z3_at_lazy_product - _mod_zlp0
                mod_ovh = max(0.0, mod_wall - mod_z3)
                mod_pct = 100.0 * mod_ovh / mod_wall if mod_wall > 1e-9 else 0.0
                print(
                    f"  [merge-profile] {module_name}: wall={mod_wall:.2f}s "
                    f"feasibility_Z3_wall={mod_z3:.2f}s merge_calls={mod_zm} "
                    f"lazy_product_calls={mod_zlp} non_Z3≈{mod_ovh:.2f}s ({mod_pct:.0f}%)",
                    flush=True,
                )

        if merge_prof:
            wall = time.monotonic() - _mp_wall0
            z3 = manager.solver_time - _mp_solver0
            ovh = max(0.0, wall - z3)
            zm = manager.feasibility_z3_at_merge - _mp_zm0
            zlp = manager.feasibility_z3_at_lazy_product - _mp_zlp0
            pct = 100.0 * ovh / wall if wall > 1e-9 else 0.0
            print(
                f"  [merge-profile] phase_wall={wall:.2f}s feasibility_Z3_wall={z3:.2f}s "
                f"(merge_calls={zm} lazy_product_calls={zlp}) "
                f"non_Z3_wall≈{ovh:.2f}s ({pct:.0f}% of phase; Python/keys/cache/disjoint+iter)",
                flush=True,
            )

        valid_assertions = [a for a in manager.assertions
                            if a.get("z3_expr") is not None]

        print("Phase: cross-module path iteration (DFS)", flush=True)
        if valid_assertions:
            print(f"  Checking {len(valid_assertions)} assertion(s)", flush=True)
        else:
            print("  Mode: no assertions — enumerating all feasible global path combinations "
                  "(path_count and DFS stats updated)", flush=True)
            print(f"  Per-module merged results: {len(merged_by_module)} module(s)", flush=True)
            if max_cross_module_paths is not None:
                print(f"  Stopping after {max_cross_module_paths:,} global path combination(s).",
                      flush=True)
        dfs_xmod = DFSCrossModuleIterator(
            per_module_results=merged_by_module,
            num_cycles=int(num_cycles),
            manager=manager,
            enable_early_pruning=True,
            enable_caching=True,
            structural_module_graph=getattr(manager, "structural_module_graph", None),
        )

        cross_stop: Optional[str] = None
        for path_combo, all_pcs, all_stores in dfs_xmod:
            if getattr(self, "timeout", False):
                cross_stop = "timeout"
                break
            manager.path_count += 1
            stats = dfs_xmod.get_stats()
            sparse = manager.path_count <= 3 or manager.path_count % 1_000_000 == 0
            frequent = self.debug and (
                manager.path_count <= 10 or manager.path_count % 100_000 == 0
            )
            if sparse or frequent:
                print(
                    "  cross-module: {} global path(s) completed; "
                    "outcome_pulls={} (each pull = next merged outcome for one module in this search; "
                    "not intra-module merge steps), "
                    "pruned {}, cache_hits {}".format(
                        manager.path_count,
                        stats["combos_checked"],
                        stats["combos_pruned"],
                        stats["cache_hits"],
                    ),
                    flush=True,
                )

            if valid_assertions:
                violation = self._check_assertions_on_path(
                    manager, visitor, all_pcs, all_stores, modules_dict)
                if violation:
                    stats = dfs_xmod.get_stats()
                    manager.cross_module_stopped_reason = "violation"
                    print(
                        "Phase: cross-module iteration stopped (assertion violation). "
                        "paths_checked={} ast_branch_visits={} DFS(pulls/pruned/cache_hits)={}/{}/{}".format(
                            manager.path_count,
                            manager.branch_count,
                            stats["combos_checked"],
                            stats["combos_pruned"],
                            stats["cache_hits"],
                        ),
                        flush=True,
                    )
                    self.module_depth -= 1
                    return

            if (
                max_cross_module_paths is not None
                and max_cross_module_paths > 0
                and manager.path_count >= max_cross_module_paths
            ):
                print(
                    f"Phase: cross-module iteration stopped — --max-cross-module-paths "
                    f"({max_cross_module_paths:,}) reached after {manager.path_count:,} path(s).",
                    flush=True,
                )
                cross_stop = "max_paths"
                break

        stats = dfs_xmod.get_stats()
        timed_out = getattr(self, "timeout", False)
        if cross_stop == "timeout":
            status = "interrupted (timeout)"
            manager.cross_module_stopped_reason = "timeout"
        elif cross_stop == "max_paths":
            status = "stopped (max-cross-module-paths)"
            manager.cross_module_stopped_reason = "max_paths"
        elif timed_out:
            status = "interrupted (timeout)"
            manager.cross_module_stopped_reason = "timeout"
        else:
            status = "complete (exhausted iterator)"
            manager.cross_module_stopped_reason = "complete"
        print(
            "Phase: cross-module iteration {} — paths_checked={} "
            "ast_branch_visits={} (SymbolicDFS if/case/while; 0 common: CFG branches are edge constraints) "
            "DFS(pulls/pruned/cache_hits)={}/{}/{}".format(
                status,
                manager.path_count,
                manager.branch_count,
                stats["combos_checked"],
                stats["combos_pruned"],
                stats["cache_hits"],
            ),
            flush=True,
        )
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

        n_assertions = len(manager.assertions)
        for a_idx, assertion_info in enumerate(manager.assertions):
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
            t0 = time.monotonic()
            result = str(s.check())
            dt = time.monotonic() - t0
            if manager is not None:
                manager.assertion_solver_time += dt
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
                module = assertion_info.get("module", "<unknown>")
                sr = assertion_info.get("source_range")
                source_pretty = self._format_source_range(
                    sr if sr is not None else assertion_info.get("source")
                )
                z3_s = str(assertion_z3)
                if len(z3_s) > 400:
                    z3_s = z3_s[:400] + " ... [truncated]"
                print(f"\n=== ASSERTION VIOLATION FOUND ===", flush=True)
                print(
                    f"  Assertion: #{a_idx + 1} of {n_assertions} (index in collected assertion list)",
                    flush=True,
                )
                print(f"  Module: {module}", flush=True)
                print(f"  Kind: {assertion_info.get('kind', '?')}", flush=True)
                print(f"  Source: {source_pretty}", flush=True)
                print(f"  Property (Z3, must hold; violation uses NOT this): {z3_s}", flush=True)
                print(f"  Counterexample: {counterexample}", flush=True)
                print(
                    f"  Solver time: feasibility={manager.solver_time:.4f}s, "
                    f"assertions={manager.assertion_solver_time:.4f}s "
                    f"(total {manager.solver_time + manager.assertion_solver_time:.4f}s)",
                    flush=True,
                )
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
                
