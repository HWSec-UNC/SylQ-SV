"""A library of helper functions for working with the PySlang AST."""
import pyslang.ast as ps_ast
import pyslang.syntax as ps_stx
from helpers.utils import init_symbol
from engine.execution_manager import ExecutionManager
from engine.symbolic_state import SymbolicState
from helpers.rvalue_to_z3 import solve_pc
from z3 import Not, is_bool, BoolVal, ExprRef, BitVecRef, BitVecVal
from engine.query_slicing import slice_query, get_vars_from_expr
from engine.query_normalization import normalize_query


def _cache_key(manager, cond_z3, negate=False):
    """Compute the cache key for a branch condition using the
    slice -> normalize -> key pipeline (Paper §4.2.2-4.2.4).

    If the cache is not enabled, returns a simple string key.
    """
    if cond_z3 is None:
        return str(cond_z3)
    expr = Not(cond_z3) if negate else cond_z3
    # If query slicing is available, slice first
    if manager.qu_path is not None:
        try:
            branch_vars = get_vars_from_expr(expr)
            all_constraints = []
            try:
                all_constraints = list(manager._pc_ref.assertions()) if hasattr(manager, '_pc_ref') else []
            except Exception:
                pass
            if all_constraints and branch_vars:
                sliced = slice_query(manager.qu_path, all_constraints, branch_vars)
                # Include the branch condition itself for a unique key
                sliced.append(expr)
                # Normalize the sliced query
                from engine.query_normalization import normalize_query_list
                return normalize_query_list(sliced)
        except Exception:
            pass
    # Fallback: normalize just the condition
    return normalize_query(expr)


def _cache_lookup(manager, key):
    """Look up a key in the cache. Returns the decoded value or None."""
    if manager.cache and manager.cache.exists(key):
        raw = manager.cache.get(key)
        if raw is None:
            return None
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    return None


def _cache_store(manager, key, value):
    """Store a key-value pair in the cache."""
    if manager.cache:
        manager.cache.set(key, str(value))

def init_state(s: SymbolicState, prev_store, ast, symbol_visitor):
    """Give fresh symbols and merge register values in."""
    symbol_visitor.dfs(ast)
    merge_states(s, prev_store)

def merge_states(state: SymbolicState, store):
    """Merges two symbolic states"""
    for key, val in state.store.items():
        if type(val) != dict:
            continue
        else:
            for key2, var in val.items():
                if var in store.values():
                    prev_symbol = state.store[key][key2]
                    new_symbol = store[key][key2]
                    state.store[key][key2].replace(prev_symbol, new_symbol)
                else:
                    state.store[key][key2] = store[key][key2]

def get_module_name(module) -> str:
    """Extracts module name from module syntax object"""
    return module.name

class SlangSymbolVisitor:
    """Visits a Slang AST by each Symbol, counting branches and paths"""

    def __init__(self):
        self.branch_points = 0
        self.paths = 0
    
    def visit_stmt(self, stmt):
        """Visits statements, counts branches (conditionals, cases, loops)"""
        if stmt is None:
            self.paths += 1
            return

        kind = stmt.kind

        if kind == ps_ast.StatementKind.Conditional:
            self.branch_points += 1
            if stmt.conditions:
                for cond in stmt.conditions:
                    self.visit_expr(cond.expr)
            if stmt.ifTrue:
                self.visit_stmt(stmt.ifTrue)
            else:
                self.paths += 1
            if stmt.ifFalse:
                self.visit_stmt(stmt.ifFalse)
            else:
                self.paths += 1

        elif kind == ps_ast.StatementKind.Case:
            self.branch_points += 1
            self.visit_expr(stmt.expr)
            for case in stmt.cases:
                for e in case.exprs:
                    self.visit_expr(e)
                self.visit_stmt(case.stmt)

        elif kind in [ps_ast.StatementKind.WhileLoop, ps_ast.StatementKind.DoWhileLoop,
                      ps_ast.StatementKind.ForLoop, ps_ast.StatementKind.ForeverLoop,
                      ps_ast.StatementKind.RepeatLoop, ps_ast.StatementKind.ForeachLoop]:
            self.branch_points += 1
            if hasattr(stmt, 'cond'):
                self.visit_expr(stmt.cond)
            if hasattr(stmt, 'init'):
                self.visit_stmt(stmt.init)
            if hasattr(stmt, 'body'):
                self.visit_stmt(stmt.body)
            if hasattr(stmt, 'incr'):
                self.visit_stmt(stmt.incr)
            self.paths += 1  # conservative

        elif kind == ps_ast.StatementKind.List and hasattr(stmt, 'body'):
            for s in stmt.body:
                self.visit_stmt(s)

        elif kind == ps_ast.StatementKind.Block and hasattr(stmt, 'body'):
            for substmt in stmt.body:
                self.visit_stmt(substmt)

        elif kind in [ps_ast.StatementKind.Return, ps_ast.StatementKind.Break,
                      ps_ast.StatementKind.Continue, ps_ast.StatementKind.Disable,
                      ps_ast.StatementKind.ForeverLoop]:
            self.paths += 1

        elif kind == ps_ast.StatementKind.Timed and hasattr(stmt, 'stmt'):
            self.visit_stmt(stmt.stmt)

        elif kind in [ps_ast.StatementKind.ImmediateAssertion, ps_ast.StatementKind.ConcurrentAssertion,
                      ps_ast.StatementKind.Wait, ps_ast.StatementKind.WaitFork, ps_ast.StatementKind.WaitOrder,
                      ps_ast.StatementKind.RandCase, ps_ast.StatementKind.RandSequence]:
            if hasattr(stmt, 'stmt'):
                self.visit_stmt(stmt.stmt)

        elif kind in [ps_ast.StatementKind.ExpressionStatement,
                      ps_ast.StatementKind.ProceduralAssign, ps_ast.StatementKind.ProceduralDeassign,
                      ps_ast.StatementKind.DisableFork, ps_ast.StatementKind.EventTrigger,
                      ps_ast.StatementKind.VariableDeclaration, ps_ast.StatementKind.Empty]:
            pass  # no effect on path or branching

        else:
            pass  # other kinds not relevant here

    def visit_expr(self, expr):
        """Visit expressions"""
        if expr is None:
            return

        kind = expr.kind
        if kind == ps_ast.ExpressionKind.ConditionalOp:
            self.branch_points += 1
            self.visit_expr(expr.predicate)
            self.visit_expr(expr.left)
            self.visit_expr(expr.right)

        elif kind == ps_ast.ExpressionKind.BinaryOp:
            self.visit_expr(expr.left)
            self.visit_expr(expr.right)

        elif kind == ps_ast.ExpressionKind.UnaryOp:
            self.visit_expr(expr.operand)

        elif kind in [ps_ast.ExpressionKind.Assignment,
                      ps_ast.ExpressionKind.NamedValue,
                      ps_ast.ExpressionKind.ElementSelect,
                      ps_ast.ExpressionKind.RangeSelect,
                      ps_ast.ExpressionKind.MemberAccess,
                      ps_ast.ExpressionKind.Call]:
            if hasattr(expr, 'left'):
                self.visit_expr(expr.left)
            if hasattr(expr, 'right'):
                self.visit_expr(expr.right)
            if hasattr(expr, 'value'):
                self.visit_expr(expr.value)

        elif kind in [ps_ast.ExpressionKind.Concatenation, ps_ast.ExpressionKind.Replication,
                      ps_ast.ExpressionKind.SimpleAssignmentPattern,
                      ps_ast.ExpressionKind.StructuredAssignmentPattern,
                      ps_ast.ExpressionKind.ReplicatedAssignmentPattern,
                      ps_ast.ExpressionKind.List, ps_ast.ExpressionKind.Pattern,
                      ps_ast.ExpressionKind.StructurePattern]:
            for e in getattr(expr, 'elements', getattr(expr, 'operands', [])):
                if hasattr(e, 'value'):
                    self.visit_expr(e.value)
                else:
                    self.visit_expr(e)
    def _recurse_if_present(self, symbol, *attr_names):
        """Helper: for given attribute names, if present on symbol recurse into them."""
        for a in attr_names:
            if hasattr(symbol, a):
                val = getattr(symbol, a)
                if val is None:
                    continue
                if isinstance(val, (list, tuple, set)):
                    for item in val:
                        if hasattr(item, "kind"):
                            self.visit(item)
                else:
                    if hasattr(val, "kind") or isinstance(val, ps_ast.Symbol):
                        self.visit(val)
    
    def visit(self, symbol):
        """Main entry point for visiting symbols"""
        if isinstance(symbol, (list, tuple, set)):
            for s in symbol:
                self.visit(s)
            return

        if not isinstance(symbol, ps_ast.Symbol):
            # Not every AST node in Slang is a ps.Symbol, so therefore I added a traversal here which traveres through their .members attribute because they might contian Statements there such as Compilation root, Definition objects, etc.
            if hasattr(symbol, "members"):
                for m in getattr(symbol, "members"):
                    self.visit(m)
            return
        
        # if symbol.kind == ps.SymbolKind.Unknown:
        #     # unknown symbol
        #     ...
        # elif symbol.kind == ps.SymbolKind.Root:
        #     # root symbol
        #     ...
        # Root / Compilation unit: recurse into members
        if symbol.kind in (ps_ast.SymbolKind.Root, ps_ast.SymbolKind.CompilationUnit):
            self._recurse_if_present(symbol, "members", "items", "declarations")
            return
        
        elif symbol.kind == ps_ast.SymbolKind.Definition:
            # definitions can contain members, etc.
            self._recurse_if_present(symbol, "members", "declarations", "items", "body", "syntax")
            return
        
        # Procedural block: count branches by delegating to visit_stmt
        if symbol.kind == ps_ast.SymbolKind.ProceduralBlock:
            # some procedural blocks expose `.body` or `.statement`
            body = getattr(symbol, "body", getattr(symbol, "statement", None))
            if body is not None:
                try:
                    self.visit_stmt(body)
                except Exception:
                    # fall back to recursing members if visit_stmt fails
                    self._recurse_if_present(symbol, "members", "body")
            else:
                self._recurse_if_present(symbol, "members")
            return
        
        elif symbol.kind == ps_ast.SymbolKind.ContinuousAssign:
            try:
                assign = getattr(symbol, "assignment", None)
                if assign is not None:
                    self.visit_expr(assign)
            except Exception:
                pass
            self._recurse_if_present(symbol, "members", "children")
            return
        
        elif symbol.kind == ps_ast.SymbolKind.Instance:
            # instance.name is a common attribute
            try:
                instance_name = getattr(symbol, "name", None)
            except Exception:
                instance_name = None
            self._recurse_if_present(symbol, "instanceBody", "parentInstance", "members", "children")
            return
        
        # Instance Body / Instance Array: recurse into members/statements
        elif symbol.kind in (ps_ast.SymbolKind.InstanceBody, ps_ast.SymbolKind.InstanceArray):
            self._recurse_if_present(symbol, "members", "statements", "items")
            return
        

        elif symbol.kind in (ps_ast.SymbolKind.Port, ps_ast.SymbolKind.Variable, ps_ast.SymbolKind.Net, ps_ast.SymbolKind.Parameter):
            # If there is an initializer or assignment expression, visit it
            init_expr = getattr(symbol, "initializer", None) or getattr(symbol, "assignment", None)
            if init_expr is not None:
                try:
                    self.visit_expr(init_expr)
                except Exception:
                    self._recurse_if_present(init_expr, "members", "elements", "expressions")
            # recurse into members to catch nested declarations
            self._recurse_if_present(symbol, "members", "declarations", "children")
            return
        
        # Cases where the symbol was not contributing to the RTL executable code, I have implemented a transversion mechanism to register the symbol in the internal maps and advance symbol_id.:
        # Attempt to recurse known container-like attributes (members/body/statements/children)
        self._recurse_if_present(symbol,
                                "members", "body", "statement", "statements",
                                "items", "declarations", "children", "syntax")
        return
class SymbolicDFS:
    """DFS visitor for PySlang symbols, updating symbolic store and path condition."""

    def __init__(self, cycles, symbolic_store=None, path_condition=None):
        self.symbolic_store = symbolic_store if symbolic_store is not None else {}
        self.path_condition = path_condition if path_condition is not None else []
        self.visited = set()
        self.cycles = 0

    def expr_to_z3(self, m, s, expr):
        """Convert a pyslang Expression to a Z3 expression using the semantic converter."""
        from helpers.rvalue_to_z3 import semantic_expr_to_z3
        store = s.store.get(m.curr_module, {})
        return semantic_expr_to_z3(expr, store, m.curr_module)

    def dfs(self, symbol):
        """Main DFS traversal of symbols"""
        if not isinstance(symbol, ps_ast.Symbol):
            return

        if symbol is None or symbol in self.visited:
            return
        self.visited.add(symbol)

        # Update symbolic store for variables, parameters, etc.
        if hasattr(symbol, "name") and symbol.kind in (
            ps_ast.SymbolKind.Variable,
            ps_ast.SymbolKind.Parameter,
            ps_ast.SymbolKind.Port,
        ):
            self.symbolic_store[symbol.name] = symbol

        # Update path condition for conditional statements
        if symbol.kind == ps_ast.SymbolKind.ProceduralBlock and hasattr(symbol, "body"):
            self.dfs_stmt(symbol.body)
        elif symbol.kind == ps_ast.SymbolKind.ContinuousAssign and hasattr(symbol, "assignment"):
            self.dfs_expr(symbol.assignment)

        # Recursively visit children if available
        if hasattr(symbol, "members"):
            for member in symbol.members:
                self.dfs(member)
        if hasattr(symbol, "body") and symbol.kind != ps_ast.SymbolKind.ProceduralBlock:
            self.dfs(symbol.body)

    def dfs_stmt(self, stmt):
        """DFS traversal of statements"""
        if stmt is None:
            return
        if stmt.kind == ps_ast.StatementKind.ExpressionStatement:
            self.dfs_expr(stmt.expr)
        elif stmt.kind == ps_ast.StatementKind.Block:
            if hasattr(stmt, "body"):
                self.dfs_stmt(stmt.body)
        elif stmt.kind == ps_ast.StatementKind.Conditional:
            cond_expr = stmt.conditions[0].expr if stmt.conditions else None
            if cond_expr:
                self.dfs_expr(cond_expr)
                self.path_condition.append(cond_expr)
            if stmt.ifTrue:
                self.dfs_stmt(stmt.ifTrue)
            if stmt.ifFalse:
                self.dfs_stmt(stmt.ifFalse)
            if cond_expr:
                self.path_condition.pop()
        elif stmt.kind == ps_ast.StatementKind.List:
            for s in stmt.body:
                self.dfs_stmt(s)

    def dfs_expr(self, expr):
        """Lightweight expression walk used during init-phase DFS.

        Full symbolic evaluation happens later via visit_expr inside
        explore_block; here we just need to avoid crashing on
        ContinuousAssign / ExpressionStatement nodes encountered
        during base_store initialization.
        """
        pass

    def visit_expr(self, m: ExecutionManager, s: SymbolicState, expr):
        """Visits expressions"""
        if getattr(m, "debug", False):
            print(expr.__class__.__name__, flush=True)
        if expr is None:
            return

        kind = expr.kind

        if kind == ps_ast.ExpressionKind.NamedValue:
            return s.store[m.curr_module].get(expr.symbol.name, init_symbol())

        elif kind == ps_ast.ExpressionKind.BinaryOp:
            self.visit_expr(m, s, expr.left)
            self.visit_expr(m, s, expr.right)

        elif kind == ps_ast.ExpressionKind.UnaryOp:
            self.visit_expr(m, s, expr.operand)

        elif kind == ps_ast.ExpressionKind.ConditionalOp:
            self.visit_expr(m, s, expr.predicate)
            self.visit_expr(m, s, expr.left)
            self.visit_expr(m, s, expr.right)

        elif kind == ps_ast.ExpressionKind.Assignment:
            # Blocking (=) updates the store immediately; nonblocking (<=) defers
            # to end of this CFG path (see SymbolicState.flush_pending_nba).
            lhs_sym = getattr(getattr(expr, 'left', None), 'symbol', None)
            if lhs_sym is not None:
                lhs_name = lhs_sym.name
                rhs = expr.right

                is_nb = bool(getattr(expr, "isNonBlocking", False))
                if getattr(s, "pending_nba", None) is None:
                    s.pending_nba = {}
                pend = s.pending_nba.setdefault(m.curr_module, {})

                def _apply(val):
                    if is_nb:
                        pend[lhs_name] = val
                    else:
                        s.store[m.curr_module][lhs_name] = val
                        # Blocking write is visible immediately; dirty its comb dependents.
                        # NBA dirties are deferred to flush_pending_nba.
                        # TODO: Param check 
                        s.mark_dirty(m.curr_module, lhs_name, m)

                rhs_sym = getattr(rhs, 'symbol', None)
                if rhs_sym is not None:
                    rhs_val = s.store[m.curr_module].get(rhs_sym.name, init_symbol())
                    _apply(rhs_val)
                elif getattr(rhs, 'kind', None) == ps_ast.ExpressionKind.IntegerLiteral:
                    _apply(str(rhs.value))
                else:
                    try:
                        from helpers.rvalue_to_z3 import semantic_expr_to_z3
                        store = s.store.get(m.curr_module, {})
                        rhs_z3 = semantic_expr_to_z3(rhs, store, m.curr_module)
                        if rhs_z3 is not None:
                            _apply(rhs_z3)
                    except Exception:
                        pass

        elif kind == ps_stx.SyntaxKind.AssignmentExpression:
            if hasattr(expr.left, "identifier") and hasattr(expr.right, "identifier"):
                if expr.right.identifier.value in s.store[m.curr_module]:
                    s.store[m.curr_module][expr.left.identifier.value] = s.store[m.curr_module][expr.right.identifier.value]
            elif hasattr(expr.left, "identifier"):
                # Only LHS has an identifier attribute
                #  RHS is likely a literal
                if getattr(m, "debug", False):
                    print(expr.right.kind, flush=True)
                if expr.right.kind == ps_stx.SyntaxKind.ConcatenationExpression:
                    # Handle concatenation on RHS
                    parts = [str(operand.value) for operand in expr.right.expressions if hasattr(operand, "value")]
                    s.store[m.curr_module][expr.left.identifier.value] = "".join(parts)
                else:
                    s.store[m.curr_module][expr.left.identifier.value] = str(expr.right.value.value)
            else:
                # LHS or RHS doesn't have an identifier attribute-skip for now
                ...

        elif kind == ps_stx.SyntaxKind.NonblockingAssignmentExpression: 
            if expr.left.kind == ps_stx.IdentifierNameSyntax:
                if expr.left.identifier.value in s.store: 
                    s.store[m.curr_module][expr.left.identifier.value] = s.store[m.curr_module][expr.right.identifier.value]
            else:
                if expr.right.kind == ps_stx.SyntaxKind.ConcatenationExpression:
                    # Handle concatenation on RHS
                    concat_value = ""
                    for operand in expr.right.expressions:
                        if hasattr(operand, "value"):
                            concat_value += str(operand.value)
                    s.store[m.curr_module][expr.left.identifier.value] = concat_value
                else:
                    ...

        elif kind ==ps_ast.ExpressionKind.Concatenation:
            for e in expr.operands:
                self.visit_expr(m, s, e)

        elif kind == ps_ast.ExpressionKind.Call:
            for arg in expr.arguments:
                self.visit_expr(m, s, arg)

        elif kind == ps_ast.ExpressionKind.ElementSelect:
            self.visit_expr(m, s, expr.value)
            self.visit_expr(m, s, expr.selector)

        elif kind == ps_ast.ExpressionKind.RangeSelect:
            self.visit_expr(m, s, expr.value)
            self.visit_expr(m, s, expr.left)
            self.visit_expr(m, s, expr.right)

        elif kind == ps_ast.ExpressionKind.Conversion:
            # Sized literals (e.g. 5'd9) are ConversionExpression; operand holds the inner expr.
            self.visit_expr(m, s, expr.operand)

        elif kind in [ps_ast.ExpressionKind.MemberAccess, ps_ast.ExpressionKind.Streaming,
                    ps_ast.ExpressionKind.Replication, ps_ast.ExpressionKind.TaggedUnion,
                    ps_ast.ExpressionKind.CopyClass]:
            self.visit_expr(m, s, expr.value)

        elif kind in [ps_ast.ExpressionKind.SimpleAssignmentPattern]:
            for e in expr.elements:
                self.visit_expr(m, s, e)

        elif kind in [ps_ast.ExpressionKind.StructuredAssignmentPattern, ps_ast.ExpressionKind.ReplicatedAssignmentPattern]:
            for e in expr.elements:
                self.visit_expr(m, s, e.value)

        elif kind in [ps_ast.ExpressionKind.MinTypMax]:
            self.visit_expr(m, s, expr.min)
            self.visit_expr(m, s, expr.typ)
            self.visit_expr(m, s, expr.max)


        # Ignore literals and null
        elif kind in [ps_ast.ExpressionKind.IntegerLiteral, ps_ast.ExpressionKind.RealLiteral,
                    ps_ast.ExpressionKind.TimeLiteral, ps_ast.ExpressionKind.NullLiteral,
                    ps_ast.ExpressionKind.StringLiteral, ps_ast.ExpressionKind.UnbasedUnsizedIntegerLiteral,
                    ps_ast.UnboundedLiteral]:
            pass

        # Ignore misc. nodes in syntax tree 
        elif kind in [ps_stx.TokenKind.IntegerLiteral, ps_stx.SyntaxKind.IntegerVectorExpression, 
                      ps_stx.SyntaxKind.ConcatenationExpression, ps_stx.SyntaxKind.IdentifierName,
                      ps_stx.SyntaxKind.IdentifierSelectName, ps_stx.TokenKind.Comma, ps_stx.SyntaxKind.IntegerLiteralExpression]:
            pass

        else:
            if getattr(m, "debug", False):
                print(f"Unsupported Expression: {expr} of kind {kind}", flush=True)

    def _visit_case_stmt(self, m: ExecutionManager, s: SymbolicState, stmt, modules=None, direction=None):
        """Case statement handling; called by visit_stmt without reading stmt.kind."""
        if getattr(m, "debug", False):
            print("_visit_case_stmt", flush=True)
            print("case", flush=True)
        m.branch_count += 1
        # Avoid reading stmt.expr if it can hang; use getattr with None default.
        case_expr = getattr(stmt, "expr", None)
        if case_expr is None:
            return
        self.visit_expr(m, s, case_expr)

        cond_z3 = self.expr_to_z3(m, s, case_expr)

        for case in getattr(stmt, "items", getattr(stmt, "case_items", [])):
            exprs = getattr(case, "expressions", getattr(case, "exprs", []))
            for e in exprs:
                self.visit_expr(m, s, e)
                s.pc.push()
                s.assertion_counter += 1
                case_z3 = self.expr_to_z3(m, s, e)

                cond_expr = cond_z3 if isinstance(cond_z3, ExprRef) else None
                case_expr = case_z3 if isinstance(case_z3, ExprRef) else None

                if cond_expr is not None:
                    if is_bool(cond_expr):
                        match_guard = cond_expr
                        mismatch_guard = Not(cond_expr)
                    elif case_expr is not None:
                        match_guard = cond_expr == case_expr
                        mismatch_guard = cond_expr != case_expr
                    elif isinstance(cond_expr, BitVecRef):
                        zero = BitVecVal(0, cond_expr.size())
                        match_guard = cond_expr != zero
                        mismatch_guard = cond_expr == zero
                    else:
                        match_guard = BoolVal(True)
                        mismatch_guard = BoolVal(True)
                else:
                    match_guard = BoolVal(True)
                    mismatch_guard = BoolVal(True)

                guard = match_guard if direction else mismatch_guard
                if not isinstance(guard, ExprRef) or not is_bool(guard):
                    guard = BoolVal(True)

                # --- Query slicing (Paper §4.2.2) ---
                if m.qu_path is not None and isinstance(guard, ExprRef):
                    m.qu_path.register_constraint(guard)

                key = _cache_key(m, guard, negate=False)
                self.branch = bool(direction)
                cached = _cache_lookup(m, key)
                if cached is not None:
                    result = cached
                else:
                    result = str(solve_pc(s.pc))
                    _cache_store(m, key, result)
                s.pc.assert_and_track(guard, f"p{s.assertion_counter}")
                if not solve_pc(s.pc):
                    s.pc.pop()
                    _cache_store(m, key, False)
                    m.abandon = True
                    m.ignore = True
                    return

                case_body = getattr(case, "statement", getattr(case, "stmt", None))
                if case_body is None and hasattr(case, "statements"):
                    case_body = case.statements

                if case_body is None:
                    s.pc.pop()
                    continue

                if isinstance(case_body, (list, tuple)):
                    body_iter = case_body
                elif hasattr(case_body, "__iter__") and not isinstance(case_body, ps_stx.StatementSyntax):
                    body_iter = list(case_body)
                else:
                    body_iter = [case_body]

                for stmt_node in body_iter:
                    if stmt_node is None:
                        continue
                    self.visit_stmt(m, s, stmt_node, modules, direction)

                s.pc.pop()

    def visit_stmt(self, m: ExecutionManager, s: SymbolicState, stmt, modules=None, direction=None):
        """Visits statements"""
        # Progress indicator: every 10k statements print one line.
        m.visit_count = getattr(m, "visit_count", 0) + 1
        if getattr(m, "debug", False) and m.visit_count % 10000 == 0:
            print("... {} statements visited".format(m.visit_count), flush=True)

        cls_name = stmt.__class__.__name__
        # Handle case/binary by class name first - never read m.ignore or stmt.kind for these.
        if "CaseStatement" in cls_name:
            if getattr(m, "debug", False):
                print("visit:", cls_name, flush=True)
            self._visit_case_stmt(m, s, stmt, modules, direction)
            return
        if cls_name == "BinaryExpressionSyntax":
            if getattr(m, "debug", False):
                print("visit:", cls_name, flush=True)
            self.visit_expr(m, s, stmt)
            return

        # For other nodes, print with kind then check ignore (only when debug).
        if getattr(m, "debug", False):
            print(
                "visit:",
                cls_name,
                getattr(getattr(stmt, "kind", None), "name", getattr(stmt, "kind", None)),
                flush=True,
            )
        if stmt is None or m.ignore:
            return

        kind = stmt.kind

        # Expression nodes (e.g. condition expressions stored in CFG basic
        # blocks) aren't Statements -- route them through visit_expr directly.
        if isinstance(kind, ps_ast.ExpressionKind):
            self.visit_expr(m, s, stmt)
            return

        if kind == ps_ast.StatementKind.ExpressionStatement:
            self.visit_expr(m, s, stmt.expr)

        elif kind == ps_ast.StatementKind.Block and hasattr(stmt, "body"):
            for substmt in stmt.body:
                self.visit_stmt(m, s, substmt, modules, direction)

        elif kind == ps_ast.StatementKind.Conditional or isinstance(stmt, ps_stx.ConditionalStatementSyntax):
            m.branch_count += 1
            # PySlang 7.0 uses conditions list, not predicate attribute
            # Pattern matches usage in dfs_stmt() method (line 550)
            cond_expr = stmt.conditions[0].expr if (hasattr(stmt, 'conditions') and stmt.conditions) else None
            if cond_expr:
                self.visit_expr(m, s, cond_expr)
                s.pc.push()
                s.assertion_counter += 1
                cond_z3 = self.expr_to_z3(m, s, cond_expr)
                # --- Query slicing (Paper §4.2.2) ---
                if m.qu_path is not None and cond_z3 is not None:
                    m.qu_path.register_constraint(cond_z3)
                if direction:
                    key = _cache_key(m, cond_z3, negate=False)
                    self.branch = True
                    cached = _cache_lookup(m, key)
                    if cached is not None:
                        result = cached
                    else:
                        result = str(solve_pc(s.pc))
                        _cache_store(m, key, result)
                    s.pc.assert_and_track(cond_z3, f"p{s.assertion_counter}")
                else:
                    self.branch = False
                    key = _cache_key(m, cond_z3, negate=True)
                    cached = _cache_lookup(m, key)
                    if cached is not None:
                        result = cached
                    else:
                        result = str(solve_pc(s.pc))
                        _cache_store(m, key, result)
                    s.pc.assert_and_track(cond_z3, f"p{s.assertion_counter}")
                if not solve_pc(s.pc):
                    neg_key = _cache_key(m, cond_z3, negate=True)
                    _cache_store(m, neg_key, False)
                    s.pc.pop()
                    m.abandon = True
                    m.ignore = True
                    return

            # PySlang 7.0 uses ifTrue/ifFalse for ConditionalStatementSyntax
            # Pattern matches usage in dfs_stmt() method (line 554-557)
            if hasattr(stmt, 'ifTrue') and stmt.ifTrue:
                self.visit_stmt(m, s, stmt.ifTrue, modules, direction)
            if hasattr(stmt, 'ifFalse') and stmt.ifFalse:
                self.visit_stmt(m, s, stmt.ifFalse, modules, direction)

            if cond_expr:
                s.pc.pop()

        elif kind == ps_ast.StatementKind.List:
            
            for s_sub in stmt.body:
                self.visit_stmt(m, s, s_sub, modules, direction)

        elif kind == ps_ast.StatementKind.ForLoop:
            if hasattr(stmt, "init"):
                self.visit_stmt(m, s, stmt.init, modules, direction)
            if hasattr(stmt, "cond"):
                self.visit_expr(m, s, stmt.cond)
            if hasattr(stmt, "body"):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, "incr"):
                self.visit_stmt(m, s, stmt.incr, modules, direction)

        elif kind == ps_ast.StatementKind.WhileLoop:
            if getattr(m, "debug", False):
                print("whileloop", flush=True)
            m.branch_count += 1
            if hasattr(stmt, "cond"):
                self.visit_expr(m, s, stmt.cond)
                s.pc.push()
                s.assertion_counter += 1
                cond_z3 = self.expr_to_z3(m, s, stmt.cond)
                # --- Query slicing (Paper §4.2.2) ---
                if m.qu_path is not None and cond_z3 is not None:
                    m.qu_path.register_constraint(cond_z3)
                if direction:
                    key = _cache_key(m, cond_z3, negate=False)
                    self.branch = True
                    cached = _cache_lookup(m, key)
                    if cached is not None:
                        result = cached
                    else:
                        result = str(solve_pc(s.pc))
                        _cache_store(m, key, result)
                    s.pc.assert_and_track(cond_z3, f"p{s.assertion_counter}")
                else:
                    key = _cache_key(m, cond_z3, negate=True)
                    self.branch = False
                    cached = _cache_lookup(m, key)
                    if cached is not None:
                        result = cached
                    else:
                        result = str(solve_pc(s.pc))
                        _cache_store(m, key, result)
                    s.pc.assert_and_track(~cond_z3, f"p{s.assertion_counter}")
                if not solve_pc(s.pc):
                    s.pc.pop()
                    neg_key = _cache_key(m, cond_z3, negate=True)
                    _cache_store(m, neg_key, False)
                    m.abandon = True
                    m.ignore = True
                    return
            if hasattr(stmt, "body"):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, "cond"):
                s.pc.pop()

        elif kind == ps_ast.StatementKind.DoWhileLoop:
            if getattr(m, "debug", False):
                print("dowhile", flush=True)
            m.branch_count += 1
            if hasattr(stmt, "body"):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, "cond"):
                self.visit_expr(m, s, stmt.cond)

        elif kind in [ps_ast.StatementKind.ProceduralAssign]:
            self.visit_expr(m, s, stmt.left)
            self.visit_expr(m, s, stmt.right)
            if hasattr(stmt.left, 'symbol') and hasattr(stmt.right, 'symbol'):
                lhs = stmt.left.symbol.name
                rhs = stmt.right.symbol.name
                s.store[m.curr_module][lhs] = s.store[m.curr_module].get(rhs, init_symbol())
            elif hasattr(stmt.left, 'symbol'):
                lhs = stmt.left.symbol.name
                s.store[m.curr_module][lhs] = init_symbol()

        # elif kind == ps.StatementKind.ProcedureCall:
        #     self.visit_expr(m, s, stmt.expr)

        elif kind in [ps_ast.StatementKind.Block,
                    ps_ast.StatementKind.Timed]:
            self.visit_stmt(m, s, stmt.body, modules, direction)

        elif kind == ps_ast.StatementKind.ImmediateAssertion:
            # Handle immediate assertions: assert(expr)
            # Assertions are collected once in the engine's phase (_collect_assertions +
            # _eval_assertion_expr); do not append here to avoid duplicate entries per path.
            expr_node = None
            for attr in ('cond', 'expr', 'condition', 'expression'):
                if hasattr(stmt, attr):
                    expr_node = getattr(stmt, attr)
                    if expr_node is not None:
                        break
            if expr_node is not None:
                self.visit_expr(m, s, expr_node)
            # Visit the action block (pass/fail body)
            if hasattr(stmt, 'body'):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, 'ifTrue'):
                self.visit_stmt(m, s, stmt.ifTrue, modules, direction)
            if hasattr(stmt, 'elseBody'):
                self.visit_stmt(m, s, stmt.elseBody, modules, direction)

        elif kind == ps_ast.StatementKind.ConcurrentAssertion:
            # Handle concurrent assertions: assert property (...)
            # Assertions are collected once in the engine's phase; do not append here.
            expr_node = None
            for attr in ('cond', 'expr', 'condition', 'expression', 'propertySpec'):
                if hasattr(stmt, attr):
                    prop = getattr(stmt, attr)
                    if prop is not None:
                        inner = getattr(prop, 'expr', None) or getattr(prop, 'expression', None)
                        if inner is not None:
                            expr_node = inner
                        else:
                            expr_node = prop
                        break
            if expr_node is not None:
                self.visit_expr(m, s, expr_node)
            # Visit the action block
            if hasattr(stmt, 'body'):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, 'ifTrue'):
                self.visit_stmt(m, s, stmt.ifTrue, modules, direction)
            if hasattr(stmt, 'elseBody'):
                self.visit_stmt(m, s, stmt.elseBody, modules, direction)

        elif kind == ps_ast.StatementKind.Return and hasattr(stmt, "expr"):
            self.visit_expr(m, s, stmt.expr)
        
        elif kind == ps_ast.StatementKind.ExpressionStatement:
            self.visit_expr(m, s, stmt.expr)


