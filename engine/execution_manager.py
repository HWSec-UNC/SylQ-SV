"""The main class that controls the flow of execution. Most of the bookkeeping happens here, and 
a lot of this information will probably be useful when working in a specific search strategy."""
# Central coordinator that tracks all execution metadata, paths explored, modules processed, and optimization state.

from __future__ import annotations
from .symbolic_state import SymbolicState
from helpers.utils import init_symbol
from typing import Dict, Optional, Set
# import pkg_resources
import pyslang.syntax as ps_stx

# Using this as a reference for conditionals:
# https://sv-lang.com/structslang_1_1syntax_1_1_statement_syntax.html
CONDITIONALS = (
    ps_stx.ConditionalStatementSyntax,
    ps_stx.CaseStatementSyntax,
    ps_stx.ForeachLoopStatementSyntax,
    ps_stx.ForLoopStatementSyntax,
    ps_stx.LoopStatementSyntax,
    ps_stx.DoWhileStatementSyntax
)

class ExecutionManager:
    """The ExecutionManager class is responsible for managing the execution of the symbolic execution engine.
    It is responsible for counting the number of paths, merging states, and other bookkeeping tasks."""
    num_paths: int = 1
    curr_level: int = 0
    path_code: str = "0" * 12
    ast_str: str = ""
    abandon: bool = False
    assertion_violation: bool = False
    in_always: bool = False
    modules = {}
    dependencies = {}
    intermodule_dependencies = {}
    updates = {}
    seen = {}
    final = False
    completed = []
    is_child: bool = False
    # Map of module name to path nums for child module
    child_num_paths = {}    
    # Map of module name to path code for child module
    child_path_codes = {}
    paths = []
    config = {}
    names_list = []
    instance_count = {}
    seen_mod = {}
    opt_1: bool = False
    curr_module: str = ""
    piece_wise: bool = False
    # Piecewise composition is the sole execution mode 
    child_range: range = None
    always_writes = {}
    curr_always = None
    opt_2: bool = True
    opt_3: bool = False
    assertions = []
    blocks_of_interest = []
    init_run_flag: bool = False
    ignore = False
    branch: bool = False
    cond_assigns = {}
    cond_updates = []
    reg_writes = set()
    path = []
    cycle = 0
    prev_store = {}
    reg_decls = set()
    reg_widths = {}
    curr_case = None
    debug: bool = False
    visit_count: int = 0  # for progress indicator when not debug
    initial_store = {}
    instances_seen = {}
    instances_loc = {}
    solver_time = 0
    # Wall time in _check_assertions_on_path (PC ∧ ¬assertion); not included in solver_time above
    assertion_solver_time: float = 0
    # Optional undirected RTL adjacency (module -> neighbor modules). If None, cross-module
    # disjoint-skip is conservative (no skip across distinct modules); see feasibility_independence.
    structural_module_graph: Optional[Dict[str, Set[str]]] = None
    # Feasibility accounting (merge / LazyProduct / cross-module); see sat_check_full_pc & dfs_iterator
    # feasibility_z3_checks = merge + lazy_product + cross_module (joint full-PC SAT calls)
    feasibility_z3_checks: int = 0
    feasibility_z3_at_merge: int = 0
    feasibility_z3_at_lazy_product: int = 0
    feasibility_z3_at_cross_module: int = 0
    feasibility_disjoint_skip_merge: int = 0
    feasibility_disjoint_skip_cross: int = 0
    feasibility_pruned_merge: int = 0
    feasibility_pruned_lazy_product: int = 0
    feasibility_pruned_cross_module: int = 0
    sv = False
    cache = None
    path_count = 0
    branch_count = 0
    # Set when cross-module DFS is skipped (no assertions): product of per-module
    # merged path counts, when computable (not LazyProduct with unknown size).
    estimated_global_combinations: Optional[int] = None
    # True when no assertions and at least one module merge is lazy / unknown size,
    # so the full cross-module product cannot be multiplied without iterating.
    feasible_paths_unknown: bool = False
    # CLI: --max-cross-module-paths
    max_cross_module_paths: Optional[int] = None
    # Why cross-module DFS ended: "", "complete", "max_paths", "timeout", "violation"
    cross_module_stopped_reason: str = ""
    # Quick-Union for query slicing (Paper §4.2.2)
    # Initialized lazily; separate instances for path exploration and merge
    qu_path = None   # QuickUnion for path-exploration queries
    qu_merge = None  # QuickUnion for merge queries (separate per §4.3)
    # Continuous-assign metadata (paper §4.4). Populated by ExecutionEngine.execute_sv.
    # module_name → list[ContinuousAssign], {idx→lhs_signal}, {rhs_signal→[idx,…]}.
    # TODO: Param check
    comb_assigns: Dict[str, list] = {}
    comb_lhs: Dict[str, Dict[int, str]] = {}
    comb_deps: Dict[str, Dict[str, list]] = {}

    def feasibility_stats_line(self) -> str:
        """One-line summary of feasibility checks (Z3, disjoint skips, pruned paths)."""
        return (
            f"Feasibility: Z3_checks={self.feasibility_z3_checks} "
            f"(merge={self.feasibility_z3_at_merge} lazy_product={self.feasibility_z3_at_lazy_product} "
            f"cross_module={self.feasibility_z3_at_cross_module}), "
            f"disjoint_skip_merge={self.feasibility_disjoint_skip_merge}, "
            f"disjoint_skip_cross={self.feasibility_disjoint_skip_cross}, "
            f"pruned_merge={self.feasibility_pruned_merge}, "
            f"pruned_lazy_product={self.feasibility_pruned_lazy_product}, "
            f"pruned_cross_module={self.feasibility_pruned_cross_module}"
        )

    def merge_states(self, state: SymbolicState, store, flag, module_name=""):
        """Merges two states. The flag is for when we are just merging a particular module"""
        for key, val in state.store.items():
            if type(val) != dict:
                continue
            else:
                for key2, var in val.items():
                    if var in store.values() and (key2 in self.reg_decls or key2.startswith("clk") or key2.startswith("rst")):
                        prev_symbol = state.store[key][key2]
                        new_symbol = store[key][key2]
                        state.store[key][key2].replace(prev_symbol, new_symbol)
                    else:
                        if flag:
                            state.store[module_name][key2] = store[key][key2]
                        else:
                            state.store[key][key2] = store[key][key2]

    def init_run(self, m: ExecutionManager, module: ps_stx.ModuleDeclarationSyntax) -> None:
        """Initalize run for a module"""
        m.init_run_flag = True
        self.count_conditionals(m, module.members)
        # these are for the COI opt
        #self.lhs_signals(m, module.members)
        #self.get_assertions(m, module.members)
        m.init_run_flag = False

    def count_conditionals(self, m: "ExecutionManager", items):
        """Recursively count all conditional statements in the AST (pyslang version)"""
        stmts = items
        if isinstance(items, ps_stx.BlockStatementSyntax):
            # PySlang uses .items, not .statements for BlockStatementSyntax
            stmts = getattr(items, 'items', getattr(items, 'statements', items))
        # If stmts is iterable, traverse each statement
        if hasattr(stmts, '__iter__'):
            for item in stmts:
                self.count_conditionals(m, item)
        elif items is not None:
            # Check for each conditional type and recurse into their bodies
            if isinstance(items, ps_stx.ConditionalStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.ifTrue)
                if items.ifFalse is not None:
                    self.count_conditionals(m, items.ifFalse)
            elif isinstance(items, ps_stx.CaseStatementSyntax):
                m.num_paths += 1
                for case in items.items:
                    # Case items may have .statements or .statement attribute
                    case_body = getattr(case, 'statements', getattr(case, 'statement', None))
                    self.count_conditionals(m, case_body)
            elif isinstance(items, ps_stx.ForLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps_stx, "ForeachLoopStatementSyntax") and isinstance(items, ps_stx.ForeachLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps_stx, "WhileLoopStatementSyntax") and isinstance(items, ps_stx.WhileLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps_stx, "DoWhileLoopStatementSyntax") and isinstance(items, ps_stx.DoWhileLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps_stx, "RepeatLoopStatementSyntax") and isinstance(items, ps_stx.RepeatLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif isinstance(items, ps_stx.BlockStatementSyntax):
                # PySlang uses .items, not .statements for BlockStatementSyntax
                self.count_conditionals(m, items.items)
            elif hasattr(ps_stx, "AlwaysConstructSyntax") and isinstance(items, ps_stx.AlwaysConstructSyntax):
                self.count_conditionals(m, items.statement)
            elif hasattr(ps_stx, "InitialConstructSyntax") and isinstance(items, ps_stx.InitialConstructSyntax):
                self.count_conditionals(m, items.statement)
            elif hasattr(ps_stx, "CaseItemSyntax") and isinstance(items, ps_stx.CaseItemSyntax):
                # CaseItemSyntax may have .statements or .statement attribute
                case_body = getattr(items, 'statements', getattr(items, 'statement', None))
                self.count_conditionals(m, case_body)

    def count_conditionals_2(self, m:ExecutionManager, items) -> int:
        """(Alternative conditional counter) Rewrite to actually return an int"""
        stmts = items
        if isinstance(items, ps_stx.BlockStatementSyntax):
            # PySlang uses .items, not .statements for BlockStatementSyntax
            stmts = items.items
            # items.cname = "Block"

        if hasattr(stmts, '__iter__'):
            for item in stmts:
                if isinstance(item, CONDITIONALS):
                    if isinstance(item, ps_stx.ConditionalStatementSyntax) or isinstance(item, ps_stx.CaseStatementSyntax):
                        if isinstance(item, ps_stx.ConditionalStatementSyntax):
                            return self.count_conditionals_2(m, item.ifTrue) + self.count_conditionals_2(m, item.ifFalse)  + 1
                        if isinstance(items, ps_stx.CaseStatementSyntax):
                            return self.count_conditionals_2(m, items.items) + 1
                if isinstance(item, ps_stx.BlockStatementSyntax):
                    return self.count_conditionals_2(m, item.statements)
                elif hasattr(ps_stx, "AlwaysConstructSyntax") and isinstance(item, ps_stx.AlwaysConstructSyntax):
                    return self.count_conditionals_2(m, item.statement)             
                elif hasattr(ps_stx, "InitialConstructSyntax") and isinstance(item, ps_stx.InitialConstructSyntax):
                    return self.count_conditionals_2(m, item.statement)
        elif items is not None:
            if isinstance(items, ps_stx.ConditionalStatementSyntax):
                return  ( self.count_conditionals_2(m, items.ifTrue) + 
                self.count_conditionals_2(m, items.ifFalse)) + 1
            if isinstance(items, ps_stx.CaseStatementSyntax):
                return self.count_conditionals_2(m, items.items) + 1
        return 0

    def seen_all_cases(self, m: ExecutionManager, bit_index: int, nested_ifs: int) -> bool:
        """Checks if we've seen all the cases for this index in the bit string.
        We know there are no more nested conditionals within the block, just want to check 
        that we have seen the path where this bit was turned on but the thing to the left of it
        could vary """
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
