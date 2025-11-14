"""The main class that controls the flow of execution. Most of the bookkeeping happens here, and 
a lot of this information will probably be useful when working in a specific search strategy."""
# Central coordinator that tracks all execution metadata, paths explored, modules processed, and optimization state.

from __future__ import annotations
from .symbolic_state import SymbolicState
from helpers.utils import init_symbol
from typing import Optional
# import pkg_resources
import pyslang as ps

# Using this as a reference for conditionals:
# https://sv-lang.com/structslang_1_1syntax_1_1_statement_syntax.html
CONDITIONALS = (
    ps.ConditionalStatementSyntax,
    ps.CaseStatementSyntax,
    ps.ForeachLoopStatementSyntax,
    ps.ForLoopStatementSyntax,
    ps.LoopStatementSyntax,
    ps.DoWhileStatementSyntax
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
    initial_store = {}
    instances_seen = {}
    instances_loc = {}
    solver_time = 0
    sv = False
    cache = None
    path_count = 0
    branch_count = 0

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

    def init_run(self, m: ExecutionManager, module: ps.ModuleDeclarationSyntax) -> None:
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
        if isinstance(items, ps.BlockStatementSyntax):
            # PySlang uses .items, not .statements for BlockStatementSyntax
            stmts = getattr(items, 'items', getattr(items, 'statements', items))
        # If stmts is iterable, traverse each statement
        if hasattr(stmts, '__iter__'):
            for item in stmts:
                self.count_conditionals(m, item)
        elif items is not None:
            # Check for each conditional type and recurse into their bodies
            if isinstance(items, ps.ConditionalStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.ifTrue)
                if items.ifFalse is not None:
                    self.count_conditionals(m, items.ifFalse)
            elif isinstance(items, ps.CaseStatementSyntax):
                m.num_paths += 1
                for case in items.items:
                    # Case items may have .statements or .statement attribute
                    case_body = getattr(case, 'statements', getattr(case, 'statement', None))
                    self.count_conditionals(m, case_body)
            elif isinstance(items, ps.ForLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps, "ForeachLoopStatementSyntax") and isinstance(items, ps.ForeachLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps, "WhileLoopStatementSyntax") and isinstance(items, ps.WhileLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps, "DoWhileLoopStatementSyntax") and isinstance(items, ps.DoWhileLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif hasattr(ps, "RepeatLoopStatementSyntax") and isinstance(items, ps.RepeatLoopStatementSyntax):
                m.num_paths += 1
                self.count_conditionals(m, items.body)
            elif isinstance(items, ps.BlockStatementSyntax):
                # PySlang uses .items, not .statements for BlockStatementSyntax
                self.count_conditionals(m, items.items)
            elif hasattr(ps, "AlwaysConstructSyntax") and isinstance(items, ps.AlwaysConstructSyntax):
                self.count_conditionals(m, items.statement)
            elif hasattr(ps, "InitialConstructSyntax") and isinstance(items, ps.InitialConstructSyntax):
                self.count_conditionals(m, items.statement)
            elif hasattr(ps, "CaseItemSyntax") and isinstance(items, ps.CaseItemSyntax):
                # CaseItemSyntax may have .statements or .statement attribute
                case_body = getattr(items, 'statements', getattr(items, 'statement', None))
                self.count_conditionals(m, case_body)

    def count_conditionals_2(self, m:ExecutionManager, items) -> int:
        """(Alternative conditional counter) Rewrite to actually return an int"""
        stmts = items
        if isinstance(items, ps.BlockStatementSyntax):
            # PySlang uses .items, not .statements for BlockStatementSyntax
            stmts = items.items
            # items.cname = "Block"

        if hasattr(stmts, '__iter__'):
            for item in stmts:
                if isinstance(item, CONDITIONALS):
                    if isinstance(item, ps.ConditionalStatementSyntax) or isinstance(item, ps.CaseStatementSyntax):
                        if isinstance(item, ps.ConditionalStatementSyntax):
                            return self.count_conditionals_2(m, item.ifTrue) + self.count_conditionals_2(m, item.ifFalse)  + 1
                        if isinstance(items, ps.CaseStatementSyntax):
                            return self.count_conditionals_2(m, items.items) + 1
                if isinstance(item, ps.BlockStatementSyntax):
                    return self.count_conditionals_2(m, item.statements)
                elif hasattr(ps, "AlwaysConstructSyntax") and isinstance(item, ps.AlwaysConstructSyntax):
                    return self.count_conditionals_2(m, item.statement)             
                elif hasattr(ps, "InitialConstructSyntax") and isinstance(item, ps.InitialConstructSyntax):
                    return self.count_conditionals_2(m, item.statement)
        elif items is not None:
            if isinstance(items, ps.ConditionalStatementSyntax):
                return  ( self.count_conditionals_2(m, items.ifTrue) + 
                self.count_conditionals_2(m, items.ifFalse)) + 1
            if isinstance(items, ps.CaseStatementSyntax):
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
