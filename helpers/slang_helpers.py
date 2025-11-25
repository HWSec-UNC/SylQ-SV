"""A library of helper functions for working with the PySlang AST."""
import pyslang as ps
from helpers.utils import init_symbol
from engine.execution_manager import ExecutionManager
from engine.symbolic_state import SymbolicState
from helpers.rvalue_to_z3 import solve_pc
from z3 import Not, is_bool, BoolVal, ExprRef, BitVecRef, BitVecVal

def init_state(s: SymbolicState, prev_store, ast, symbol_visitor):
    """give fresh symbols and merge register values in."""
    global_module_to_port_to_direction = dict()
    expr_symbol_visitor = ExpressionSymbolCollector()
    symbol_visitor.dfs(ast)
    params = expr_symbol_visitor.parameters
    port_list = expr_symbol_visitor.ports
    for i, token in enumerate(port_list):
        port = extract_kinds_from_descendants(token, desired_kinds=[ps.SyntaxKind.ImplicitAnsiPort])
        port_list.append(port)

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

    def __init__(self): #Post processor visitor -> doesn't depend on the num_cycles
        self.symbol_id_to_symbol = dict()
        self.sourceRange_to_symbol_id = dict()
        self.kind_to_symbol_id = dict()

        self.symbol_id = 0
        self.branch_points = 0
        self.paths = 0
    
    def visit_stmt(self, stmt):
        """Visits statements, counts branches (conditionals, cases, loops)"""
        print("visiting stmt!")
        if stmt is None:
            self.paths += 1
            return

        kind = stmt.kind

        if kind == ps.StatementKind.Conditional:
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

        elif kind == ps.StatementKind.Case:
            self.branch_points += 1
            self.visit_expr(stmt.expr)
            for case in stmt.cases:
                for e in case.exprs:
                    self.visit_expr(e)
                self.visit_stmt(case.stmt)

        elif kind in [ps.StatementKind.WhileLoop, ps.StatementKind.DoWhileLoop,
                      ps.StatementKind.ForLoop, ps.StatementKind.ForeverLoop,
                      ps.StatementKind.RepeatLoop, ps.StatementKind.ForeachLoop]:
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

        elif kind == ps.StatementKind.List and hasattr(stmt, 'body'):
            for s in stmt.body:
                self.visit_stmt(s)

        elif kind == ps.StatementKind.Block and hasattr(stmt, 'body'):
            for substmt in stmt.body:
                self.visit_stmt(substmt)

        elif kind in [ps.StatementKind.Return, ps.StatementKind.Break,
                      ps.StatementKind.Continue, ps.StatementKind.Disable,
                      ps.StatementKind.ForeverLoop]:
            self.paths += 1

        elif kind == ps.StatementKind.Timed and hasattr(stmt, 'stmt'):
            self.visit_stmt(stmt.stmt)

        elif kind in [ps.StatementKind.ImmediateAssertion, ps.StatementKind.ConcurrentAssertion,
                      ps.StatementKind.Wait, ps.StatementKind.WaitFork, ps.StatementKind.WaitOrder,
                      ps.StatementKind.RandCase, ps.StatementKind.RandSequence]:
            if hasattr(stmt, 'stmt'):
                self.visit_stmt(stmt.stmt)

        elif kind in [ps.StatementKind.ExpressionStatement,
                      ps.StatementKind.ProceduralAssign, ps.StatementKind.ProceduralDeassign,
                      ps.StatementKind.DisableFork, ps.StatementKind.EventTrigger,
                      ps.StatementKind.VariableDeclaration, ps.StatementKind.Empty]:
            pass  # no effect on path or branching

        else:
            pass  # other kinds not relevant here

    def visit_expr(self, expr):
        """Visit expressions"""
        if expr is None:
            return

        kind = expr.kind
        if kind == ps.ExpressionKind.ConditionalOp:
            self.branch_points += 1
            self.visit_expr(expr.predicate)
            self.visit_expr(expr.left)
            self.visit_expr(expr.right)

        elif kind == ps.ExpressionKind.BinaryOp:
            self.visit_expr(expr.left)
            self.visit_expr(expr.right)

        elif kind == ps.ExpressionKind.UnaryOp:
            self.visit_expr(expr.operand)

        elif kind in [ps.ExpressionKind.Assignment,
                      ps.ExpressionKind.NamedValue,
                      ps.ExpressionKind.ElementSelect,
                      ps.ExpressionKind.RangeSelect,
                      ps.ExpressionKind.MemberAccess,
                      ps.ExpressionKind.Call]:
            if hasattr(expr, 'left'):
                self.visit_expr(expr.left)
            if hasattr(expr, 'right'):
                self.visit_expr(expr.right)
            if hasattr(expr, 'value'):
                self.visit_expr(expr.value)

        elif kind in [ps.ExpressionKind.Concatenation, ps.ExpressionKind.Replication,
                      ps.ExpressionKind.SimpleAssignmentPattern,
                      ps.ExpressionKind.StructuredAssignmentPattern,
                      ps.ExpressionKind.ReplicatedAssignmentPattern,
                      ps.ExpressionKind.List, ps.ExpressionKind.Pattern,
                      ps.ExpressionKind.StructurePattern]:
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
                    if hasattr(val, "kind") or isinstance(val, ps.Symbol):
                        self.visit(val)
    
    def visit(self, symbol):
        """Main entry point for visiting symbols"""
        if isinstance(symbol, (list, tuple, set)):
            for s in symbol:
                self.visit(s)
            return

        if not isinstance(symbol, ps.Symbol):
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
        if symbol.kind in (ps.SymbolKind.Root, ps.SymbolKind.CompilationUnit):
            self._recurse_if_present(symbol, "members", "items", "declarations")
            return
        
        elif symbol.kind == ps.SymbolKind.Definition:
            # definitions can contain members, etc.
            self._recurse_if_present(symbol, "members", "declarations", "items", "body", "syntax")
            return
        
        # Procedural block: count branches by delegating to visit_stmt
        if symbol.kind == ps.SymbolKind.ProceduralBlock:
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
        
        elif symbol.kind == ps.SymbolKind.ContinuousAssign:
            try:
                assign = getattr(symbol, "assignment", None)
                if assign is not None:
                    self.visit_expr(assign)
            except Exception:
                pass
            self._recurse_if_present(symbol, "members", "children")
            return
        
        elif symbol.kind == ps.SymbolKind.Instance:
            # instance.name is a common attribute
            try:
                instance_name = getattr(symbol, "name", None)
            except Exception:
                instance_name = None
            self._recurse_if_present(symbol, "instanceBody", "parentInstance", "members", "children")
            return
        
        # Instance Body / Instance Array: recurse into members/statements
        elif symbol.kind in (ps.SymbolKind.InstanceBody, ps.SymbolKind.InstanceArray):
            self._recurse_if_present(symbol, "members", "statements", "items")
            return
        

        elif symbol.kind in (ps.SymbolKind.Port, ps.SymbolKind.Variable, ps.SymbolKind.Net, ps.SymbolKind.Parameter):
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
    if False: #left the other symbol kinds for reference
        #Types (Only provide structure information. They don't produce any RTL executable code)
        if symbol.kind == ps.SymbolKind.PredefinedIntegerType:
            # predefinedintegertype symbol
            ...
        elif symbol.kind == ps.SymbolKind.ScalarType:
            # scalartype symbol
            ...
        elif symbol.kind == ps.SymbolKind.FloatingType:
            # floatingtype symbol
            ...
        elif symbol.kind == ps.SymbolKind.EnumType:
            # enumtype symbol
            ...
        elif symbol.kind == ps.SymbolKind.EnumValue:
            # enumvalue symbol
            ...
        elif symbol.kind == ps.SymbolKind.PackedArrayType:
            # packedarraytype symbol
            ...
        elif symbol.kind == ps.SymbolKind.FixedSizeUnpackedArrayType:
            # fixedsizeunpackedarraytype symbol
            ...
        elif symbol.kind == ps.SymbolKind.DynamicArrayType:
            # dynamicarraytype symbol
            ...
        elif symbol.kind == ps.SymbolKind.DPIOpenArrayType:
            # dpiopenarraytype symbol
            ...
        elif symbol.kind == ps.SymbolKind.AssociativeArrayType:
            # associativearraytype symbol
            ...
        elif symbol.kind == ps.SymbolKind.QueueType:
            # queuetype symbol
            ...
        elif symbol.kind == ps.SymbolKind.PackedStructType:
            # packedstructtype symbol
            ...
        elif symbol.kind == ps.SymbolKind.UnpackedStructType:
            # unpackedstructtype symbol
            ...
        elif symbol.kind == ps.SymbolKind.PackedUnionType:
            # packeduniontype symbol
            ...
        elif symbol.kind == ps.SymbolKind.UnpackedUnionType:
            # unpackeduniontype symbol
            ...
    elif False:
        # ********* These are not RTL executable code *********
            if symbol.kind == ps.SymbolKind.ClassType:
            # classtype symbol
                ...
            elif symbol.kind == ps.SymbolKind.CovergroupType:
                # covergrouptype symbol
                ...
            elif symbol.kind == ps.SymbolKind.VoidType:
                # voidtype symbol
                ...
            elif symbol.kind == ps.SymbolKind.NullType:
                # nulltype symbol
                ...
            elif symbol.kind == ps.SymbolKind.CHandleType:
                # chandletype symbol
                ...
            elif symbol.kind == ps.SymbolKind.StringType:
                # stringtype symbol
                ...
            elif symbol.kind == ps.SymbolKind.UnboundedType:
                # unboundedtype symbol
                ...
            elif symbol.kind == ps.SymbolKind.TypeRefType:
                # typereftype symbol
                ...
            elif symbol.kind == ps.SymbolKind.UntypedType:
                # untypedtype symbol
                ...
            elif symbol.kind == ps.SymbolKind.SequenceType:
                # sequencetype symbol
                ...
            elif symbol.kind == ps.SymbolKind.PropertyType:
                # propertytype symbol
                ...
            elif symbol.kind == ps.SymbolKind.VirtualInterfaceType:
                # virtualinterfacetype symbol
                ...
            elif symbol.kind == ps.SymbolKind.TypeAlias:
                # typealias symbol
                ...
            elif symbol.kind == ps.SymbolKind.ErrorType:
                # errortype symbol
                ...
            elif symbol.kind == ps.SymbolKind.ForwardingTypedef:
                # forwardingtypedef symbol
                ...
            elif symbol.kind == ps.SymbolKind.NetType:
                # nettype symbol
                ...
            elif symbol.kind == ps.SymbolKind.TypeParameter:
                # typeparameter symbol
                ...
            elif symbol.kind == ps.SymbolKind.MultiPort:
                # multiport symbol
                ...
            elif symbol.kind == ps.SymbolKind.InterfacePort:
                # interfaceport symbol
                ...
            elif symbol.kind == ps.SymbolKind.Modport:
                # modport symbol
                ...
            elif symbol.kind == ps.SymbolKind.ModportPort:
                # modportport symbol
                ...
            elif symbol.kind == ps.SymbolKind.ModportClocking:
                # modportclocking symbol
                ...
            elif symbol.kind == ps.SymbolKind.Package:
                # package symbol
                ...
            elif symbol.kind == ps.SymbolKind.ExplicitImport:
                # explicitimport symbol
                ...
            elif symbol.kind == ps.SymbolKind.WildcardImport:
                # wildcardimport symbol
                ...
            elif symbol.kind == ps.SymbolKind.Attribute:
                # attribute symbol
                ...
            elif symbol.kind == ps.SymbolKind.Genvar:
                # genvar symbol
                ...
            elif symbol.kind == ps.SymbolKind.GenerateBlock:
                # generateblock symbol
                ...
            elif symbol.kind == ps.SymbolKind.GenerateBlockArray:
                # generateblockarray symbol
                ...
            elif symbol.kind == ps.SymbolKind.StatementBlock:
                # statementblock symbol
                ...
            elif symbol.kind == ps.SymbolKind.FormalArgument:
                # formalargument symbol
                ...
            elif symbol.kind == ps.SymbolKind.Field:
                # field symbol
                ...
            elif symbol.kind == ps.SymbolKind.ClassProperty:
                # classproperty symbol
                ...
            elif symbol.kind == ps.SymbolKind.Subroutine:
                # subroutine symbol
                ...
            elif symbol.kind == ps.SymbolKind.ElabSystemTask:
                # elabsystemtask symbol
                ...
            elif symbol.kind == ps.SymbolKind.GenericClassDef:
                # genericclassdef symbol
                ...
            elif symbol.kind == ps.SymbolKind.MethodPrototype:
                # methodprototype symbol
                ...
            elif symbol.kind == ps.SymbolKind.UninstantiatedDef:
                # uninstantiateddef symbol
                ...
            elif symbol.kind == ps.SymbolKind.Iterator:
                # iterator symbol
                ...
            elif symbol.kind == ps.SymbolKind.PatternVar:
                # patternvar symbol
                ...
            elif symbol.kind == ps.SymbolKind.ConstraintBlock:
                # constraintblock symbol
                ...
            elif symbol.kind == ps.SymbolKind.DefParam:
                # defparam symbol
                ...
            elif symbol.kind == ps.SymbolKind.Primitive:
                # primitive symbol
                ...
            elif symbol.kind == ps.SymbolKind.PrimitivePort:
                # primitiveport symbol
                ...
            elif symbol.kind == ps.SymbolKind.PrimitiveInstance:
                # primitiveinstance symbol
                ...
            elif symbol.kind == ps.SymbolKind.SpecifyBlock:
                # specifyblock symbol
                ...
            elif symbol.kind == ps.SymbolKind.Sequence:
                # sequence symbol
                ...
            elif symbol.kind == ps.SymbolKind.Property:
                # property symbol
                ...
            elif symbol.kind == ps.SymbolKind.AssertionPort:
                # assertionport symbol
                ...
            elif symbol.kind == ps.SymbolKind.ClockingBlock:
                # clockingblock symbol
                ...
            elif symbol.kind == ps.SymbolKind.ClockVar:
                # clockvar symbol
                ...
            elif symbol.kind == ps.SymbolKind.LocalAssertionVar:
                # localassertionvar symbol
                ...
            elif symbol.kind == ps.SymbolKind.LetDecl:
                # letdecl symbol
                ...
            elif symbol.kind == ps.SymbolKind.Checker:
                # checker symbol
                ...
            elif symbol.kind == ps.SymbolKind.CheckerInstance:
                # checkerinstance symbol
                ...
            elif symbol.kind == ps.SymbolKind.CheckerInstanceBody:
                # checkerinstancebody symbol
                ...
            elif symbol.kind == ps.SymbolKind.RandSeqProduction:
                # randseqproduction symbol
                ...
            elif symbol.kind == ps.SymbolKind.CovergroupBody:
                # covergroupbody symbol
                ...
            elif symbol.kind == ps.SymbolKind.Coverpoint:
                # coverpoint symbol
                ...
            elif symbol.kind == ps.SymbolKind.CoverCross:
                # covercross symbol
                ...
            elif symbol.kind == ps.SymbolKind.CoverCrossBody:
                # covercrossbody symbol
                ...
            elif symbol.kind == ps.SymbolKind.CoverageBin:
                # coveragebin symbol
                ...
            elif symbol.kind == ps.SymbolKind.TimingPath:
                # timingpath symbol
                ...
            elif symbol.kind == ps.SymbolKind.PulseStyle:
                # pulsestyle symbol
                ...
            elif symbol.kind == ps.SymbolKind.SystemTimingCheck:
                # systemtimingcheck symbol
                ...
            elif symbol.kind == ps.SymbolKind.AnonymousProgram:
                # anonymousprogram symbol
                ...
            elif symbol.kind == ps.SymbolKind.ConfigBlock:
                ...

class SymbolicDFS:
    """DFS visitor for PySlang symbols, updating symbolic store and path condition."""

    def __init__(self, cycles, symbolic_store=None, path_condition=None):
        self.symbolic_store = symbolic_store if symbolic_store is not None else {}
        self.path_condition = path_condition if path_condition is not None else []
        self.visited = set()
        self.cycles = 0

    def dfs(self, symbol):
        """Main DFS traversal of symbols"""
        if not isinstance(symbol, ps.Symbol):
            return

        if symbol is None or symbol in self.visited:
            return
        self.visited.add(symbol)

        # Update symbolic store for variables, parameters, etc.
        if hasattr(symbol, "name") and symbol.kind in (
            ps.SymbolKind.Variable,
            ps.SymbolKind.Parameter,
            ps.SymbolKind.Port,
        ):
            self.symbolic_store[symbol.name] = symbol

        # Update path condition for conditional statements
        if symbol.kind == ps.SymbolKind.ProceduralBlock and hasattr(symbol, "body"):
            self.dfs_stmt(symbol.body)
        elif symbol.kind == ps.SymbolKind.ContinuousAssign and hasattr(symbol, "assignment"):
            self.dfs_expr(symbol.assignment)

        # Recursively visit children if available
        if hasattr(symbol, "members"):
            for member in symbol.members:
                self.dfs(member)
        if hasattr(symbol, "body") and symbol.kind != ps.SymbolKind.ProceduralBlock:
            self.dfs(symbol.body)

    def dfs_stmt(self, stmt):
        """DFS traversal of statements"""
        if stmt is None:
            return
        if stmt.kind == ps.StatementKind.ExpressionStatement:
            self.dfs_expr(stmt.expr)
        elif stmt.kind == ps.StatementKind.Block:
            if hasattr(stmt, "body"):
                self.dfs_stmt(stmt.body)
        elif stmt.kind == ps.StatementKind.Conditional:
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
        elif stmt.kind == ps.StatementKind.List:
            for s in stmt.body:
                self.dfs_stmt(s)

    def visit_expr(self, m: ExecutionManager, s: SymbolicState, expr):
        """Visits expressions"""
        print(expr.__class__.__name__, dir(expr))
        if expr is None:
            return

        kind = expr.kind

        if kind == ps.ExpressionKind.NamedValue:
            return s.store[m.curr_module].get(expr.symbol.name, init_symbol())

        elif kind == ps.ExpressionKind.BinaryOp:
            self.visit_expr(m, s, expr.left)
            self.visit_expr(m, s, expr.right)

        elif kind == ps.ExpressionKind.UnaryOp:
            self.visit_expr(m, s, expr.operand)

        elif kind == ps.ExpressionKind.ConditionalOp:
            self.visit_expr(m, s, expr.predicate)
            self.visit_expr(m, s, expr.left)
            self.visit_expr(m, s, expr.right)

        elif kind == ps.SyntaxKind.AssignmentExpression:
            if hasattr(expr.left, "identifier") and hasattr(expr.right, "identifier"):
                if expr.right.identifier.value in s.store[m.curr_module]:
                    s.store[m.curr_module][expr.left.identifier.value] = s.store[m.curr_module][expr.right.identifier.value]
            elif hasattr(expr.left, "identifier"):
                # Only LHS has an identifier attribute
                #  RHS is likely a literal
                print(expr.right.kind)
                if expr.right.kind == ps.SyntaxKind.ConcatenationExpression:
                    # Handle concatenation on RHS
                    parts = [str(operand.value) for operand in expr.right.expressions if hasattr(operand, "value")]
                    s.store[m.curr_module][expr.left.identifier.value] = "".join(parts)
                else:
                    s.store[m.curr_module][expr.left.identifier.value] = str(expr.right.value.value)
            else:
                # LHS or RHS doesn't have an identifier attribute — skip for now
                ...

        elif kind == ps.SyntaxKind.NonblockingAssignmentExpression: 
            if expr.left.kind == ps.IdentifierNameSyntax:
                if expr.left.identifier.value in s.store: 
                    s.store[m.curr_module][expr.left.identifier.value] = s.store[m.curr_module][expr.right.identifier.value]
            else:
                if expr.right.kind == ps.SyntaxKind.ConcatenationExpression:
                    # Handle concatenation on RHS
                    concat_value = ""
                    for operand in expr.right.expressions:
                        if hasattr(operand, "value"):
                            concat_value += str(operand.value)
                    s.store[m.curr_module][expr.left.identifier.value] = concat_value
                else:
                    ...

        elif kind ==ps.ExpressionKind.Concatenation:
            for e in expr.operands:
                self.visit_expr(m, s, e)

        elif kind == ps.ExpressionKind.Call:
            for arg in expr.arguments:
                self.visit_expr(m, s, arg)

        elif kind == ps.ExpressionKind.ElementSelect:
            self.visit_expr(m, s, expr.value)
            self.visit_expr(m, s, expr.selector)

        elif kind == ps.ExpressionKind.RangeSelect:
            self.visit_expr(m, s, expr.value)
            self.visit_expr(m, s, expr.left)
            self.visit_expr(m, s, expr.right)

        elif kind in [ps.ExpressionKind.MemberAccess, ps.ExpressionKind.Streaming,
                    ps.ExpressionKind.Replication, ps.ExpressionKind.TaggedUnion,
                    ps.ExpressionKind.Conversion, ps.ExpressionKind.CopyClass,
                    ps.ExpressionKind.Streaming]:
            self.visit_expr(m, s, expr.value)

        elif kind in [ps.ExpressionKind.SimpleAssignmentPattern]:
            for e in expr.elements:
                self.visit_expr(m, s, e)

        elif kind in [ps.ExpressionKind.StructuredAssignmentPattern, ps.ExpressionKind.ReplicatedAssignmentPattern]:
            for e in expr.elements:
                self.visit_expr(m, s, e.value)

        elif kind in [ps.ExpressionKind.MinTypMax]:
            self.visit_expr(m, s, expr.min)
            self.visit_expr(m, s, expr.typ)
            self.visit_expr(m, s, expr.max)


        # Ignore literals and null
        elif kind in [ps.ExpressionKind.IntegerLiteral, ps.ExpressionKind.RealLiteral,
                    ps.ExpressionKind.TimeLiteral, ps.ExpressionKind.NullLiteral,
                    ps.ExpressionKind.StringLiteral, ps.ExpressionKind.UnbasedUnsizedIntegerLiteral,
                    ps.UnboundedLiteral]:
            pass

        # Ignore misc. nodes in syntax tree 
        elif kind in [ps.TokenKind.IntegerLiteral, ps.SyntaxKind.IntegerVectorExpression, 
                      ps.SyntaxKind.ConcatenationExpression, ps.SyntaxKind.IdentifierName,
                      ps.SyntaxKind.IdentifierSelectName, ps.TokenKind.Comma, ps.SyntaxKind.IntegerLiteralExpression]:
            pass

        else:
            print(f"Unsupported Expression: {expr} of kind {kind}")


    def visit_stmt(self, m: ExecutionManager, s: SymbolicState, stmt, modules=None, direction=None):
        """Visits statements"""
        print("visit:", stmt.__class__.__name__, getattr(getattr(stmt, "kind", None), "name", getattr(stmt, "kind", None)))
        if stmt is None or m.ignore:
            return

        kind = stmt.kind

        if kind == ps.SyntaxKind.ExpressionStatement:
            self.visit_expr(m, s, stmt.expr)

        elif kind == ps.StatementKind.Block and hasattr(stmt, "body"):
            for substmt in stmt.body:
                self.visit_stmt(m, s, substmt, modules, direction)

        elif kind == ps.StatementKind.Conditional or isinstance(stmt, ps.ConditionalStatementSyntax):
            m.branch_count += 1
            # PySlang 7.0 uses conditions list, not predicate attribute
            # Pattern matches usage in dfs_stmt() method (line 550)
            cond_expr = stmt.conditions[0].expr if (hasattr(stmt, 'conditions') and stmt.conditions) else None
            if cond_expr:
                self.visit_expr(m, s, cond_expr)
                s.pc.push()
                s.assertion_counter += 1
                cond_z3 = self.expr_to_z3(m, s, cond_expr)
                if direction:
                    key = str(cond_z3)
                    self.branch = True
                    if m.cache.exists(key):
                        result = m.cache.get(key).decode()
                    else:
                        result = str(solve_pc(s.pc))
                        m.cache.set(str(cond_z3), str(solve_pc(s.pc)))
                    s.pc.assert_and_track(cond_z3, f"p{s.assertion_counter}")
                else:
                    self.branch = False
                    key = f"~{cond_z3}"
                    if m.cache.exists(key):
                        result = m.cache.get(key).decode()
                    else:
                        result = str(solve_pc(s.pc))
                        m.cache.set(f"~{cond_z3}", str(solve_pc(s.pc)))
                    s.pc.assert_and_track(cond_z3, f"p{s.assertion_counter}")
                if not solve_pc(s.pc):
                    m.cache.set(f"~{str(cond_z3)}", False)
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

        elif kind == ps.StatementKind.List:
            
            for s_sub in stmt.body:
                self.visit_stmt(m, s, s_sub, modules, direction)

        elif kind == ps.StatementKind.ForLoop:
            if hasattr(stmt, "init"):
                self.visit_stmt(m, s, stmt.init, modules, direction)
            if hasattr(stmt, "cond"):
                self.visit_expr(m, s, stmt.cond)
            if hasattr(stmt, "body"):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, "incr"):
                self.visit_stmt(m, s, stmt.incr, modules, direction)

        elif kind == ps.StatementKind.WhileLoop:
            print("whileloop")
            m.branch_count += 1
            if hasattr(stmt, "cond"):
                self.visit_expr(m, s, stmt.cond)
                s.pc.push()
                s.assertion_counter += 1
                cond_z3 = self.expr_to_z3(m, s, stmt.cond)
                if direction:
                    key = str(cond_z3)
                    self.branch = True
                    if m.cache.exists(key):
                        result = m.cache.get(key).decode()
                    else:
                        result = str(solve_pc(s.pc))
                        m.cache.set(str(cond_z3), str(solve_pc(s.pc)))
                    s.pc.assert_and_track(cond_z3, f"p{s.assertion_counter}")
                else:
                    key = str(f"~{cond_z3}")
                    self.branch = False
                    if m.cache.exists(key):
                        result = m.cache.get(key).decode()
                    else:
                        result = str(solve_pc(s.pc))
                        m.cache.set(f"~{str(cond_z3)}", str(solve_pc(s.pc)))
                    s.pc.assert_and_track(~cond_z3, f"p{s.assertion_counter}")
                if not solve_pc(s.pc):
                    s.pc.pop()
                    m.cache.set(str(cond_z3), False)
                    m.abandon = True
                    m.ignore = True
                    return
            if hasattr(stmt, "body"):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, "cond"):
                s.pc.pop()

        elif kind == ps.StatementKind.DoWhileLoop:
            print("dowhile")
            m.branch_count += 1
            if hasattr(stmt, "body"):
                self.visit_stmt(m, s, stmt.body, modules, direction)
            if hasattr(stmt, "cond"):
                self.visit_expr(m, s, stmt.cond)

        #elif kind == ps.StatementKind.Case:
        elif stmt.__class__.__name__ == "CaseStatementSyntax":
            print("case")
            m.branch_count += 1
            self.visit_expr(m, s, stmt.expr)

            cond_z3 = self.expr_to_z3(m, s, stmt.expr)

            #for case in stmt.cases:
            for case in getattr(stmt, "items", getattr(stmt, "case_items", [])):
                exprs = getattr(case, "expressions", getattr(case, "exprs", []))
                #for e in case.exprs:
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

                    key = str(guard)
                    self.branch = bool(direction)
                    if m.cache.exists(key):
                        result = m.cache.get(key).decode()
                    else:
                        result = str(solve_pc(s.pc))
                        m.cache.set(key, result)
                    s.pc.assert_and_track(guard, f"p{s.assertion_counter}")
                    if not solve_pc(s.pc):
                        s.pc.pop()
                        if m.cache is not None:
                            m.cache.set(key, str(False))
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
                    elif hasattr(case_body, "__iter__") and not isinstance(case_body, ps.StatementSyntax):
                        body_iter = list(case_body)
                    else:
                        body_iter = [case_body]

                    for stmt_node in body_iter:
                        if stmt_node is None:
                            continue
                        self.visit_stmt(m, s, stmt_node, modules, direction)

                    s.pc.pop()

        elif kind in [ps.StatementKind.ProceduralAssign]:
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

        elif kind in [ps.StatementKind.Block,
                    ps.StatementKind.Timed]:
            self.visit_stmt(m, s, stmt.body, modules, direction)

        # elif kind in [ps.StatementKind.Assert, ps.StatementKind.Assume, ps.StatementKind.Cover]:
        #     self.visit_expr(m, s, stmt.expr)
        #     self.visit_stmt(m, s, stmt.body, modules, direction)
        #     if hasattr(stmt, "elseBody"):
        #         self.visit_stmt(m, s, stmt.elseBody, modules, direction)

        elif kind == ps.StatementKind.Return and hasattr(stmt, "expr"):
            self.visit_expr(m, s, stmt.expr)
        
        elif kind == ps.StatementKind.ExpressionStatement:
            self.visit_expr(m, s, stmt.expr)

class ExpressionSymbolCollector:
    """Visitor that traverses an expression and collects parameter and port symbols."""

    def __init__(self):
        self.parameters = set()
        self.ports = set()

    def visit(self, expr):
        if expr is None:
            return
        kind = expr.kind
        if kind == ps.ExpressionKind.NamedValue:
            symbol = getattr(expr, "symbol", None)
            if symbol is not None:
                if symbol.kind == ps.SymbolKind.Parameter:
                    self.parameters.add(symbol)
                elif symbol.kind == ps.SymbolKind.Port:
                    self.ports.add(symbol)
        elif kind == ps.ExpressionKind.BinaryOp:
            self.visit(expr.left)
            self.visit(expr.right)
        elif kind == ps.ExpressionKind.UnaryOp:
            self.visit(expr.operand)
        elif kind == ps.ExpressionKind.Assignment:
            self.visit(expr.left)
            self.visit(expr.right)
        elif kind == ps.ExpressionKind.Concatenation:
            for e in expr.operands:
                self.visit(e)
        elif kind == ps.ExpressionKind.Call:
            for arg in expr.arguments:
                self.visit(arg)
        elif kind == ps.ExpressionKind.ElementSelect:
            self.visit(expr.value)
            self.visit(expr.selector)
        elif kind == ps.ExpressionKind.RangeSelect:
            self.visit(expr.value)
            self.visit(expr.left)
            self.visit(expr.right)
        elif kind == ps.ExpressionKind.ConditionalOp:
            self.visit(expr.predicate)
            self.visit(expr.left)
            self.visit(expr.right)
        elif kind == ps.ExpressionKind.MemberAccess:
            self.visit(expr.value)
        elif kind == ps.ExpressionKind.Streaming:
            self.visit(expr.value)
        elif kind == ps.ExpressionKind.Replication:
            self.visit(expr.value)
            for e in expr.elements:
                self.visit(e)
        elif kind == ps.ExpressionKind.SimpleAssignmentPattern:
            for e in expr.elements:
                self.visit(e)
        elif kind == ps.ExpressionKind.StructuredAssignmentPattern:
            for e in expr.elements:
                self.visit(e.value)
        elif kind == ps.ExpressionKind.ReplicatedAssignmentPattern:
            self.visit(expr.value)
            for e in expr.elements:
                self.visit(e)
        # Add more cases as needed for other expression kinds

    def collect(self, expr):
        self.visit(expr)
        return list(self.parameters), list(self.ports)


# Commented out this slangNodeVisitor - unused dead code (~1200 lines)
# Was imported but never instantiated or called anywhere in the codebase.
# Only contains print statements for ig debugging PySlang node types, no actual logic.
# The codebase uses SlangSymbolVisitor and SymbolicDFS instead (symbol-based visitors).
# Wrapped in 'if False:' to make it unreachable but preserve for reference.
if False: 
    class SlangNodeVisitor:
        """Visits a Slang AST by each Node (not by symbols)"""
        visitor_for_symbol = None
        
        def __init__(self, visitor_for_symbol):
            print("building a node visitor")
            self.visitor_for_symbol = visitor_for_symbol
            self.node_id_to_node = dict()
            self.node_id_to_pid  = {0:None}
            self.node_id_to_cids = dict()
            self.node_id_to_cids = {0:None}
            self.node_id_to_name = {0:""}
            self.node_id_to_name_symbol = {0:""}
            self.node_id_to_predicates = {0:[]}

            self.kind_to_node_ids = dict()
            
            self.level = 0
            self.num_children_in_level = 1
            self.num_children_in_next_level = 0
            self.num_children_processed = 0
            self.processed_children_ids = list()

            self.node_id = 0

        def traverse_tree(self, starting_node):
            """Traverse the AST."""

            self.queue = [starting_node]

            while len(self.queue) > 0:
                curr_node = self.queue.pop(0)
                self.visit(curr_node, use_queue=True)
            
            return True

        def process_node_for_predicates(self, node):
            pid = self.node_id_to_pid[self.node_id]
            if pid == None:
                return
            p_node = self.node_id_to_node[pid]

            new_predicate = []
            if p_node.kind == ps.SyntaxKind.ConditionalStatement:
                if p_node.statement == node:
                    new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), True)]
                elif p_node.elseClause == node:
                    new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), False)]
            if p_node.kind == ps.SyntaxKind.ConditionalExpression:
                if   p_node.left  == node:
                    new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), True)]
                elif p_node.right == node:
                    new_predicate = [(self.find_corresponding_child_id(pid, p_node.predicate), False)]
            
            prev_predicates = self.node_id_to_predicates[pid]
            self.node_id_to_predicates[self.node_id] = prev_predicates+new_predicate

        def process_node_for_name(self, node):
            pid = self.node_id_to_pid[self.node_id]
            if pid == None:
                return
            p_node = self.node_id_to_node[pid]
            prev_name = self.node_id_to_name[pid]

            new_name = None
            # if node.kind == ps.SyntaxKind.CompilationUnit:
            #     new_name = "/"
            if node.kind == ps.SyntaxKind.ModuleDeclaration:
                new_name = node.header.name.value
            elif node.kind == ps.SyntaxKind.HierarchicalInstance:
                new_name = node.decl.name.value
            elif node.kind == ps.SyntaxKind.Declarator:
                new_name = node.name.value

            if new_name == None:
                self.node_id_to_name[self.node_id] = f"{prev_name}"
            else:
                self.node_id_to_name[self.node_id] = f"{prev_name}.{new_name}"

        def extract_kinds_from_descendants(self, nid, desired_kinds=[ps.TokenKind.Identifier]):
            desired_nids = list()

            queue = [(nid, [])]
            while len(queue) > 0:
                curr_nid, curr_metadata = queue.pop(0)
                curr_node = self.node_id_to_node[curr_nid]
                
                if curr_node.kind in desired_kinds:
                    desired_nids.append(curr_nid)

                for curr_cid in self.node_id_to_cids[curr_nid]:
                    new_metadata = []
                    queue.append((curr_cid, curr_metadata+new_metadata))

            return desired_nids

        def visit(self, node, use_queue=True):
            self.node_id_to_node[self.node_id] = node
            self.node_id_to_cids[self.node_id] = list()

            if node.kind == ps.SyntaxKind.Unknown:
                print("UNKNOWN NODE")
            elif node.kind == ps.SyntaxKind.SyntaxList:
                print("SYNTAX LIST")
            elif node.kind == ps.SyntaxKind.TokenList:
                print("TOKEN LIST")
            elif node.kind == ps.SyntaxKind.SeparatedList:
                print("SEPARATED LIST")
            elif node.kind == ps.SyntaxKind.AcceptOnPropertyExpr:
                print("ACCEPT ON PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ActionBlock:
                print("ACTION BLOCK")
            elif node.kind == ps.SyntaxKind.AddAssignmentExpression:
                print("ADD ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.AddExpression:
                print("ADD EXPRESSION")
            elif node.kind == ps.SyntaxKind.ProceduralBlockSyntax:
                print("ALWAYS BLOCK")
            elif node.kind == ps.SyntaxKind.AlwaysCombBlock:
                print("ALWAYS COMB BLOCK")
            elif node.kind == ps.SyntaxKind.AlwaysFFBlock:
                print("ALWAYS FF BLOCK")
            elif node.kind == ps.SyntaxKind.AlwaysLatchBlock:
                print("ALWAYS LATCH BLOCK")
            elif node.kind == ps.SyntaxKind.AndAssignmentExpression:
                print("AND ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.AndPropertyExpr:
                print("AND PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.AndSequenceExpr:
                print("AND SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.AnonymousProgram:
                print("ANONYMOUS PROGRAM")
            elif node.kind == ps.SyntaxKind.AnsiPortList:
                print("ANSI PORT LIST")
            elif node.kind == ps.SyntaxKind.AnsiUdpPortList:
                print("ANSI UDP PORT LIST")
            elif node.kind == ps.SyntaxKind.ArgumentList:
                print("ARGUMENT LIST")
            elif node.kind == ps.SyntaxKind.ArithmeticLeftShiftAssignmentExpression:
                print("ARITHMETIC LEFT SHIFT ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ArithmeticRightShiftAssignmentExpression:
                print("ARITHMETIC RIGHT SHIFT ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ArithmeticShiftLeftExpression:
                print("ARITHMETIC SHIFT LEFT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ArithmeticShiftRightExpression:
                print("ARITHMETIC SHIFT RIGHT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ArrayAndMethod:
                print("ARRAY AND METHOD")
            elif node.kind == ps.SyntaxKind.ArrayOrMethod:
                print("ARRAY OR METHOD")
            elif node.kind == ps.SyntaxKind.ArrayOrRandomizeMethodExpression:
                print("ARRAY OR RANDOMIZE METHOD EXPRESSION")
            elif node.kind == ps.SyntaxKind.ArrayUniqueMethod:
                print("ARRAY UNIQUE METHOD")
            elif node.kind == ps.SyntaxKind.ArrayXorMethod:
                print("ARRAY XOR METHOD")
            elif node.kind == ps.SyntaxKind.AscendingRangeSelect:
                print("ASCENDING RANGE SELECT")
            elif node.kind == ps.SyntaxKind.AssertPropertyStatement:
                print("ASSERT PROPERTY STATEMENT")
            elif node.kind == ps.SyntaxKind.AssertionItemPort:
                print("ASSERTION ITEM PORT")
            elif node.kind == ps.SyntaxKind.AssertionItemPortList:
                print("ASSERTION ITEM PORT LIST")
            elif node.kind == ps.SyntaxKind.AssignmentExpression:
                print("ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.AssignmentPatternExpression:
                print("ASSIGNMENT PATTERN EXPRESSION")
            elif node.kind == ps.SyntaxKind.AssignmentPatternItem:
                print("ASSIGNMENT PATTERN ITEM")
            elif node.kind == ps.SyntaxKind.AssumePropertyStatement:
                print("ASSUME PROPERTY STATEMENT")
            elif node.kind == ps.SyntaxKind.AttributeInstance:
                print("ATTRIBUTE INSTANCE")
            elif node.kind == ps.SyntaxKind.AttributeSpec:
                print("ATTRIBUTE SPEC")
            elif node.kind == ps.SyntaxKind.BadExpression:
                print("BAD EXPRESSION")
            elif node.kind == ps.SyntaxKind.BeginKeywordsDirective:
                print("BEGIN KEYWORDS DIRECTIVE")
            elif node.kind == ps.SyntaxKind.BinSelectWithFilterExpr:
                print("BIN SELECT WITH FILTER EXPR")
            elif node.kind == ps.SyntaxKind.BinaryAndExpression:
                print("BINARY AND EXPRESSION")
            elif node.kind == ps.SyntaxKind.BinaryBinsSelectExpr:
                print("BINARY BINS SELECT EXPR")
            elif node.kind == ps.SyntaxKind.BinaryBlockEventExpression:
                print("BINARY BLOCK EVENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.BinaryConditionalDirectiveExpression:
                print("BINARY CONDITIONAL DIRECTIVE EXPRESSION")
            elif node.kind == ps.SyntaxKind.BinaryEventExpression:
                print("BINARY EVENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.BinaryOrExpression:
                print("BINARY OR EXPRESSION")
            elif node.kind == ps.SyntaxKind.BinaryXnorExpression:
                print("BINARY XNOR EXPRESSION")
            elif node.kind == ps.SyntaxKind.BinaryXorExpression:
                print("BINARY XOR EXPRESSION")
            elif node.kind == ps.SyntaxKind.BindDirective:
                print("BIND DIRECTIVE")
            elif node.kind == ps.SyntaxKind.BindTargetList:
                print("BIND TARGET LIST")
            elif node.kind == ps.SyntaxKind.BinsSelectConditionExpr:
                print("BINS SELECT CONDITION EXPR")
            elif node.kind == ps.SyntaxKind.BinsSelection:
                print("BINS SELECTION")
            elif node.kind == ps.SyntaxKind.BitSelect:
                print("BIT SELECT")
            elif node.kind == ps.SyntaxKind.BitType:
                print("BIT TYPE")
            elif node.kind == ps.SyntaxKind.BlockCoverageEvent:
                print("BLOCK COVERAGE EVENT")
            elif node.kind == ps.SyntaxKind.BlockingEventTriggerStatement:
                print("BLOCKING EVENT TRIGGER STATEMENT")
            elif node.kind == ps.SyntaxKind.ByteType:
                print("BYTE TYPE")
            elif node.kind == ps.SyntaxKind.CHandleType:
                print("CHANDLE TYPE")
            elif node.kind == ps.SyntaxKind.CaseEqualityExpression:
                print("CASE EQUALITY EXPRESSION")
            elif node.kind == ps.SyntaxKind.CaseGenerate:
                print("CASE GENERATE")
            elif node.kind == ps.SyntaxKind.CaseInequalityExpression:
                print("CASE INEQUALITY EXPRESSION")
            elif node.kind == ps.SyntaxKind.CasePropertyExpr:
                print("CASE PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.CaseStatement:
                print("CASE STATEMENT")
            elif node.kind == ps.SyntaxKind.CastExpression:
                print("CAST EXPRESSION")
            elif node.kind == ps.SyntaxKind.CellConfigRule:
                print("CELL CONFIG RULE")
            elif node.kind == ps.SyntaxKind.CellDefineDirective:
                print("CELL DEFINE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.ChargeStrength:
                print("CHARGE STRENGTH")
            elif node.kind == ps.SyntaxKind.CheckerDataDeclaration:
                print("CHECKER DATA DECLARATION")
            elif node.kind == ps.SyntaxKind.CheckerDeclaration:
                print("CHECKER DECLARATION")
            elif node.kind == ps.SyntaxKind.CheckerInstanceStatement:
                print("CHECKER INSTANCE STATEMENT")
            elif node.kind == ps.SyntaxKind.CheckerInstantiation:
                print("CHECKER INSTANTIATION")
            elif node.kind == ps.SyntaxKind.ClassDeclaration:
                print("CLASS DECLARATION")
            elif node.kind == ps.SyntaxKind.ClassMethodDeclaration:
                print("CLASS METHOD DECLARATION")
            elif node.kind == ps.SyntaxKind.ClassMethodPrototype:
                print("CLASS METHOD PROTOTYPE")
            elif node.kind == ps.SyntaxKind.ClassName:
                print("CLASS NAME")
            elif node.kind == ps.SyntaxKind.ClassPropertyDeclaration:
                print("CLASS PROPERTY DECLARATION")
            elif node.kind == ps.SyntaxKind.ClassSpecifier:
                print("CLASS SPECIFIER")
            elif node.kind == ps.SyntaxKind.ClockingDeclaration:
                print("CLOCKING DECLARATION")
            elif node.kind == ps.SyntaxKind.ClockingDirection:
                print("CLOCKING DIRECTION")
            elif node.kind == ps.SyntaxKind.ClockingItem:
                print("CLOCKING ITEM")
            elif node.kind == ps.SyntaxKind.ClockingPropertyExpr:
                print("CLOCKING PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ClockingSequenceExpr:
                print("CLOCKING SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.ClockingSkew:
                print("CLOCKING SKEW")
            elif node.kind == ps.SyntaxKind.ColonExpressionClause:
                print("COLON EXPRESSION CLAUSE")
            elif node.kind == ps.SyntaxKind.CompilationUnit:
                print("COMPILATION UNIT")
            elif node.kind == ps.SyntaxKind.ConcatenationExpression:
                print("CONCATENATION EXPRESSION")
            elif node.kind == ps.SyntaxKind.ConcurrentAssertionMember:
                print("CONCURRENT ASSERTION MEMBER")
            elif node.kind == ps.SyntaxKind.ConditionalConstraint:
                print("CONDITIONAL CONSTRAINT")
            elif node.kind == ps.SyntaxKind.ConditionalExpression:
                print("CONDITIONAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.ConditionalPathDeclaration:
                print("CONDITIONAL PATH DECLARATION")
            elif node.kind == ps.SyntaxKind.ConditionalPattern:
                print("CONDITIONAL PATTERN")
            elif node.kind == ps.SyntaxKind.ConditionalPredicate:
                print("CONDITIONAL PREDICATE")
            elif node.kind == ps.SyntaxKind.ConditionalPropertyExpr:
                print("CONDITIONAL PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ConditionalStatement:
                print("CONDITIONAL STATEMENT")
            elif node.kind == ps.SyntaxKind.ConfigCellIdentifier:
                print("CONFIG CELL IDENTIFIER")
            elif node.kind == ps.SyntaxKind.ConfigDeclaration:
                print("CONFIG DECLARATION")
            elif node.kind == ps.SyntaxKind.ConfigInstanceIdentifier:
                print("CONFIG INSTANCE IDENTIFIER")
            elif node.kind == ps.SyntaxKind.ConfigLiblist:
                print("CONFIG LIBLIST")
            elif node.kind == ps.SyntaxKind.ConfigUseClause:
                print("CONFIG USE CLAUSE")
            elif node.kind == ps.SyntaxKind.ConstraintBlock:
                print("CONSTRAINT BLOCK")
            elif node.kind == ps.SyntaxKind.ConstraintDeclaration:
                print("CONSTRAINT DECLARATION")
            elif node.kind == ps.SyntaxKind.ConstraintPrototype:
                print("CONSTRAINT PROTOTYPE")
            elif node.kind == ps.SyntaxKind.ConstructorName:
                print("CONSTRUCTOR NAME")
            elif node.kind == ps.SyntaxKind.ContinuousAssign:
                print("ASSIGNMENT STATEMENT")
            elif node.kind == ps.SyntaxKind.CopyClassExpression:
                print("COPY CLASS EXPRESSION")
            elif node.kind == ps.SyntaxKind.CoverCross:
                print("COVER CROSS")
            elif node.kind == ps.SyntaxKind.CoverPropertyStatement:
                print("COVER PROPERTY STATEMENT")
            elif node.kind == ps.SyntaxKind.CoverSequenceStatement:
                print("COVER SEQUENCE STATEMENT")
            elif node.kind == ps.SyntaxKind.CoverageBins:
                print("COVERAGE BINS")
            elif node.kind == ps.SyntaxKind.CoverageBinsArraySize:
                print("COVERAGE BINS ARRAY SIZE")
            elif node.kind == ps.SyntaxKind.CoverageIffClause:
                print("COVERAGE IFF CLAUSE")
            elif node.kind == ps.SyntaxKind.CoverageOption:
                print("COVERAGE OPTION")
            elif node.kind == ps.SyntaxKind.CovergroupDeclaration:
                print("COVERGROUP DECLARATION")
            elif node.kind == ps.SyntaxKind.Coverpoint:
                print("COVERPOINT")
            elif node.kind == ps.SyntaxKind.CycleDelay:
                print("CYCLE DELAY")
            elif node.kind == ps.SyntaxKind.DPIExport:
                print("DPI EXPORT")
            elif node.kind == ps.SyntaxKind.DPIImport:
                print("DPI IMPORT")
            elif node.kind == ps.SyntaxKind.DataDeclaration:
                print("DATA DECLARATION")
            elif node.kind == ps.SyntaxKind.Declarator:
                print("DECLARATOR")
            elif node.kind == ps.SyntaxKind.DefParam:
                print("DEF PARAM")
            elif node.kind == ps.SyntaxKind.DefParamAssignment:
                print("DEF PARAM ASSIGNMENT")
            elif node.kind == ps.SyntaxKind.DefaultCaseItem:
                print("DEFAULT CASE ITEM")
            elif node.kind == ps.SyntaxKind.DefaultClockingReference:
                print("DEFAULT CLOCKING REFERENCE")
            elif node.kind == ps.SyntaxKind.DefaultConfigRule:
                print("DEFAULT CONFIG RULE")
            elif node.kind == ps.SyntaxKind.DefaultCoverageBinInitializer:
                print("DEFAULT COVERAGE BIN INITIALIZER")
            elif node.kind == ps.SyntaxKind.DefaultDecayTimeDirective:
                print("DEFAULT DECAY TIME DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DefaultDisableDeclaration:
                print("DEFAULT DISABLE DECLARATION")
            elif node.kind == ps.SyntaxKind.DefaultDistItem:
                print("DEFAULT DIST ITEM")
            elif node.kind == ps.SyntaxKind.DefaultExtendsClauseArg:
                print("DEFAULT EXTENDS CLAUSE ARG")
            elif node.kind == ps.SyntaxKind.DefaultFunctionPort:
                print("DEFAULT FUNCTION PORT")
            elif node.kind == ps.SyntaxKind.DefaultNetTypeDirective:
                print("DEFAULT NET TYPE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DefaultPatternKeyExpression:
                print("DEFAULT PATTERN KEY EXPRESSION")
            elif node.kind == ps.SyntaxKind.DefaultPropertyCaseItem:
                print("DEFAULT PROPERTY CASE ITEM")
            elif node.kind == ps.SyntaxKind.DefaultRsCaseItem:
                print("DEFAULT RS CASE ITEM")
            elif node.kind == ps.SyntaxKind.DefaultSkewItem:
                print("DEFAULT SKEW ITEM")
            elif node.kind == ps.SyntaxKind.DefaultTriregStrengthDirective:
                print("DEFAULT TRIREG STRENGTH DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DeferredAssertion:
                print("DEFERRED ASSERTION")
            elif node.kind == ps.SyntaxKind.DefineDirective:
                print("DEFINE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.Delay3:
                print("DELAY 3")
            elif node.kind == ps.SyntaxKind.DelayControl:
                print("DELAY CONTROL")
            elif node.kind == ps.SyntaxKind.DelayModeDistributedDirective:
                print("DELAY MODE DISTRIBUTED DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DelayModePathDirective:
                print("DELAY MODE PATH DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DelayModeUnitDirective:
                print("DELAY MODE UNIT DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DelayModeZeroDirective:
                print("DELAY MODE ZERO DIRECTIVE")
            elif node.kind == ps.SyntaxKind.DelayedSequenceElement:
                print("DELAYED SEQUENCE ELEMENT")
            elif node.kind == ps.SyntaxKind.DelayedSequenceExpr:
                print("DELAYED SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.DescendingRangeSelect:
                print("DESCENDING RANGE SELECT")
            elif node.kind == ps.SyntaxKind.DisableConstraint:
                print("DISABLE CONSTRAINT")
            elif node.kind == ps.SyntaxKind.DisableForkStatement:
                print("DISABLE FORK STATEMENT")
            elif node.kind == ps.SyntaxKind.DisableIff:
                print("DISABLE IFF")
            elif node.kind == ps.SyntaxKind.DisableStatement:
                print("DISABLE STATEMENT")
            elif node.kind == ps.SyntaxKind.DistConstraintList:
                print("DIST CONSTRAINT LIST")
            elif node.kind == ps.SyntaxKind.DistItem:
                print("DIST ITEM")
            elif node.kind == ps.SyntaxKind.DistWeight:
                print("DIST WEIGHT")
            elif node.kind == ps.SyntaxKind.DivideAssignmentExpression:
                print("DIVIDE ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.DivideExpression:
                print("DIVIDE EXPRESSION")
            elif node.kind == ps.SyntaxKind.DividerClause:
                print("DIVIDER CLAUSE")
            elif node.kind == ps.SyntaxKind.DoWhileStatement:
                print("DO WHILE STATEMENT")
            elif node.kind == ps.SyntaxKind.DotMemberClause:
                print("DOT MEMBER CLAUSE")
            elif node.kind == ps.SyntaxKind.DriveStrength:
                print("DRIVE STRENGTH")
            elif node.kind == ps.SyntaxKind.EdgeControlSpecifier:
                print("EDGE CONTROL SPECIFIER")
            elif node.kind == ps.SyntaxKind.EdgeDescriptor:
                print("EDGE DESCRIPTOR")
            elif node.kind == ps.SyntaxKind.EdgeSensitivePathSuffix:
                print("EDGE SENSITIVE PATH SUFFIX")
            elif node.kind == ps.SyntaxKind.ElabSystemTask:
                print("ELAB SYSTEM TASK")
            elif node.kind == ps.SyntaxKind.ElementSelect:
                print("ELEMENT SELECT")
            elif node.kind == ps.SyntaxKind.ElementSelectExpression:
                print("ELEMENT SELECT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ElsIfDirective:
                print("ELSIF DIRECTIVE")
            elif node.kind == ps.SyntaxKind.ElseClause:
                print("ELSE CLAUSE")
            elif node.kind == ps.SyntaxKind.ElseConstraintClause:
                print("ELSE CONSTRAINT CLAUSE")
            elif node.kind == ps.SyntaxKind.ElseDirective:
                print("ELSE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.ElsePropertyClause:
                print("ELSE PROPERTY CLAUSE")
            elif node.kind == ps.SyntaxKind.EmptyArgument:
                print("EMPTY ARGUMENT")
            elif node.kind == ps.SyntaxKind.EmptyIdentifierName:
                print("EMPTY IDENTIFIER NAME")
            elif node.kind == ps.SyntaxKind.EmptyMember:
                print("EMPTY MEMBER")
            elif node.kind == ps.SyntaxKind.EmptyNonAnsiPort:
                print("EMPTY NON ANSI PORT")
            elif node.kind == ps.SyntaxKind.EmptyPortConnection:
                print("EMPTY PORT CONNECTION")
            elif node.kind == ps.SyntaxKind.EmptyQueueExpression:
                print("EMPTY QUEUE EXPRESSION")
            elif node.kind == ps.SyntaxKind.EmptyStatement:
                print("EMPTY STATEMENT")
            elif node.kind == ps.SyntaxKind.EmptyTimingCheckArg:
                print("EMPTY TIMING CHECK ARG")
            elif node.kind == ps.SyntaxKind.EndCellDefineDirective:
                print("END CELL DEFINE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.EndIfDirective:
                print("END IF DIRECTIVE")
            elif node.kind == ps.SyntaxKind.EndKeywordsDirective:
                print("END KEYWORDS DIRECTIVE")
            elif node.kind == ps.SyntaxKind.EndProtectDirective:
                print("END PROTECT DIRECTIVE")
            elif node.kind == ps.SyntaxKind.EndProtectedDirective:
                print("END PROTECTED DIRECTIVE")
            elif node.kind == ps.SyntaxKind.EnumType:
                print("ENUM TYPE")
            elif node.kind == ps.SyntaxKind.EqualityExpression:
                print("EQUALITY EXPRESSION")
            elif node.kind == ps.SyntaxKind.EqualsAssertionArgClause:
                print("EQUALS ASSERTION ARG CLAUSE")
            elif node.kind == ps.SyntaxKind.EqualsTypeClause:
                print("EQUALS TYPE CLAUSE")
            elif node.kind == ps.SyntaxKind.EqualsValueClause:
                print("EQUALS VALUE CLAUSE")
            elif node.kind == ps.SyntaxKind.EventControl:
                print("EVENT CONTROL")
            elif node.kind == ps.SyntaxKind.EventControlWithExpression:
                print("EVENT CONTROL WITH EXPRESSION")
            elif node.kind == ps.SyntaxKind.EventType:
                print("EVENT TYPE")
            elif node.kind == ps.SyntaxKind.ExpectPropertyStatement:
                print("EXPECT PROPERTY STATEMENT")
            elif node.kind == ps.SyntaxKind.ExplicitAnsiPort:
                print("EXPLICIT ANSI PORT")
            elif node.kind == ps.SyntaxKind.ExplicitNonAnsiPort:
                print("EXPLICIT NON ANSI PORT")
            elif node.kind == ps.SyntaxKind.ExpressionConstraint:
                print("EXPRESSION CONSTRAINT")
            elif node.kind == ps.SyntaxKind.ExpressionCoverageBinInitializer:
                print("EXPRESSION COVERAGE BIN INITIALIZER")
            elif node.kind == ps.SyntaxKind.ExpressionOrDist:
                print("EXPRESSION OR DIST")
            elif node.kind == ps.SyntaxKind.ExpressionPattern:
                print("EXPRESSION PATTERN")
            elif node.kind == ps.SyntaxKind.ExpressionStatement:
                print("EXPRESSION STATEMENT")
            elif node.kind == ps.SyntaxKind.ExpressionTimingCheckArg:
                print("EXPRESSION TIMING CHECK ARG")
            elif node.kind == ps.SyntaxKind.ExtendsClause:
                print("EXTENDS CLAUSE")
            elif node.kind == ps.SyntaxKind.ExternInterfaceMethod:
                print("EXTERN INTERFACE METHOD")
            elif node.kind == ps.SyntaxKind.ExternModuleDecl:
                print("EXTERN MODULE DECL")
            elif node.kind == ps.SyntaxKind.ExternUdpDecl:
                print("EXTERN UDP DECL")
            elif node.kind == ps.SyntaxKind.FilePathSpec:
                print("FILE PATH SPEC")
            elif node.kind == ps.SyntaxKind.FinalBlock:
                print("FINAL BLOCK")
            elif node.kind == ps.SyntaxKind.FirstMatchSequenceExpr:
                print("FIRST MATCH SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.FollowedByPropertyExpr:
                print("FOLLOWED BY PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ForLoopStatement:
                print("FOR LOOP STATEMENT")
            elif node.kind == ps.SyntaxKind.ForVariableDeclaration:
                print("FOR VARIABLE DECLARATION")
            elif node.kind == ps.SyntaxKind.ForeachLoopList:
                print("FOREACH LOOP LIST")
            elif node.kind == ps.SyntaxKind.ForeachLoopStatement:
                print("FOREACH LOOP STATEMENT")
            elif node.kind == ps.SyntaxKind.ForeverStatement:
                print("FOREVER STATEMENT")
            elif node.kind == ps.SyntaxKind.ForwardTypeRestriction:
                print("FORWARD TYPE RESTRICTION")
            elif node.kind == ps.SyntaxKind.ForwardTypedefDeclaration:
                print("FORWARD TYPEDEF DECLARATION")
            elif node.kind == ps.SyntaxKind.FunctionDeclaration:
                print("FUNCTION DECLARATION")
            elif node.kind == ps.SyntaxKind.FunctionPort:
                print("FUNCTION PORT")
            elif node.kind == ps.SyntaxKind.FunctionPortList:
                print("FUNCTION PORT LIST")
            elif node.kind == ps.SyntaxKind.FunctionPrototype:
                print("FUNCTION PROTOTYPE")
            elif node.kind == ps.SyntaxKind.GenerateBlock:
                print("GENERATE BLOCK")
            elif node.kind == ps.SyntaxKind.GenerateRegion:
                print("GENERATE REGION")
            elif node.kind == ps.SyntaxKind.GenvarDeclaration:
                print("GENVAR DECLARATION")
            elif node.kind == ps.SyntaxKind.GreaterThanEqualExpression:
                print("GREATER THAN EQUAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.GreaterThanExpression:
                print("GREATER THAN EXPRESSION")
            elif node.kind == ps.SyntaxKind.HierarchicalInstance:
                print("HIERARCHICAL INSTANCE")
            elif node.kind == ps.SyntaxKind.HierarchyInstantiation:
                print("HIERARCHY INSTANTIATION")
            elif node.kind == ps.SyntaxKind.IdWithExprCoverageBinInitializer:
                print("ID WITH EXPR COVERAGE BIN INITIALIZER")
            elif node.kind == ps.SyntaxKind.IdentifierName:
                print("IDENTIFIER NAME")
            elif node.kind == ps.SyntaxKind.IdentifierSelectName:
                print("IDENTIFIER SELECT NAME")
            elif node.kind == ps.SyntaxKind.IfDefDirective:
                print("IFDEF DIRECTIVE")
            elif node.kind == ps.SyntaxKind.IfGenerate:
                print("IF GENERATE")
            elif node.kind == ps.SyntaxKind.IfNDefDirective:
                print("IFNDEF DIRECTIVE")
            elif node.kind == ps.SyntaxKind.IfNonePathDeclaration:
                print("IF NONE PATH DECLARATION")
            elif node.kind == ps.SyntaxKind.IffEventClause:
                print("IFF EVENT CLAUSE")
            elif node.kind == ps.SyntaxKind.IffPropertyExpr:
                print("IFF PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ImmediateAssertStatement:
                print("IMMEDIATE ASSERT STATEMENT")
            elif node.kind == ps.SyntaxKind.ImmediateAssertionMember:
                print("IMMEDIATE ASSERTION MEMBER")
            elif node.kind == ps.SyntaxKind.ImmediateAssumeStatement:
                print("IMMEDIATE ASSUME STATEMENT")
            elif node.kind == ps.SyntaxKind.ImmediateCoverStatement:
                print("IMMEDIATE COVER STATEMENT")
            elif node.kind == ps.SyntaxKind.ImplementsClause:
                print("IMPLEMENTS CLAUSE")
            elif node.kind == ps.SyntaxKind.ImplicationConstraint:
                print("IMPLICATION CONSTRAINT")
            elif node.kind == ps.SyntaxKind.ImplicationPropertyExpr:
                print("IMPLICATION PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ImplicitAnsiPort:
                print("IMPLICIT ANSI PORT")
            elif node.kind == ps.SyntaxKind.ImplicitEventControl:
                print("IMPLICIT EVENT CONTROL")
            elif node.kind == ps.SyntaxKind.ImplicitNonAnsiPort:
                print("IMPLICIT NON ANSI PORT")
            elif node.kind == ps.SyntaxKind.ImplicitType:
                print("IMPLICIT TYPE")
            elif node.kind == ps.SyntaxKind.ImpliesPropertyExpr:
                print("IMPLIES PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.IncludeDirective:
                print("INCLUDE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.InequalityExpression:
                print("INEQUALITY EXPRESSION")
            elif node.kind == ps.SyntaxKind.InitialBlock:
                print("INITIAL BLOCK")
            elif node.kind == ps.SyntaxKind.InsideExpression:
                print("INSIDE EXPRESSION")
            elif node.kind == ps.SyntaxKind.InstanceConfigRule:
                print("INSTANCE CONFIG RULE")
            elif node.kind == ps.SyntaxKind.InstanceName:
                print("INSTANCE NAME")
            elif node.kind == ps.SyntaxKind.IntType:
                print("INT TYPE")
            elif node.kind == ps.SyntaxKind.IntegerLiteralExpression:
                print("INTEGER LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.IntegerType:
                print("INTEGER TYPE")
            elif node.kind == ps.SyntaxKind.IntegerVectorExpression:
                print("INTEGER VECTOR EXPRESSION")
            elif node.kind == ps.SyntaxKind.InterfaceDeclaration:
                print("INTERFACE DECLARATION")
            elif node.kind == ps.SyntaxKind.InterfaceHeader:
                print("INTERFACE HEADER")
            elif node.kind == ps.SyntaxKind.InterfacePortHeader:
                print("INTERFACE PORT HEADER")
            elif node.kind == ps.SyntaxKind.IntersectClause:
                print("INTERSECT CLAUSE")
            elif node.kind == ps.SyntaxKind.IntersectSequenceExpr:
                print("INTERSECT SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.InvocationExpression:
                print("INVOCATION EXPRESSION")
            elif node.kind == ps.SyntaxKind.JumpStatement:
                print("JUMP STATEMENT")
            elif node.kind == ps.SyntaxKind.LessThanEqualExpression:
                print("LESS THAN EQUAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.LessThanExpression:
                print("LESS THAN EXPRESSION")
            elif node.kind == ps.SyntaxKind.LetDeclaration:
                print("LET DECLARATION")
            elif node.kind == ps.SyntaxKind.LibraryDeclaration:
                print("LIBRARY DECLARATION")
            elif node.kind == ps.SyntaxKind.LibraryIncDirClause:
                print("LIBRARY INC DIR CLAUSE")
            elif node.kind == ps.SyntaxKind.LibraryIncludeStatement:
                print("LIBRARY INCLUDE STATEMENT")
            elif node.kind == ps.SyntaxKind.LibraryMap:
                print("LIBRARY MAP")
            elif node.kind == ps.SyntaxKind.LineDirective:
                print("LINE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.LocalScope:
                print("LOCAL SCOPE")
            elif node.kind == ps.SyntaxKind.LocalVariableDeclaration:
                print("LOCAL VARIABLE DECLARATION")
            elif node.kind == ps.SyntaxKind.LogicType:
                print("LOGIC TYPE")
            elif node.kind == ps.SyntaxKind.LogicalAndExpression:
                print("LOGICAL AND EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalEquivalenceExpression:
                print("LOGICAL EQUIVALENCE EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalImplicationExpression:
                print("LOGICAL IMPLICATION EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalLeftShiftAssignmentExpression:
                print("LOGICAL LEFT SHIFT ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalOrExpression:
                print("LOGICAL OR EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalRightShiftAssignmentExpression:
                print("LOGICAL RIGHT SHIFT ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalShiftLeftExpression:
                print("LOGICAL SHIFT LEFT EXPRESSION")
            elif node.kind == ps.SyntaxKind.LogicalShiftRightExpression:
                print("LOGICAL SHIFT RIGHT EXPRESSION")
            elif node.kind == ps.SyntaxKind.LongIntType:
                print("LONG INT TYPE")
            elif node.kind == ps.SyntaxKind.LoopConstraint:
                print("LOOP CONSTRAINT")
            elif node.kind == ps.SyntaxKind.LoopGenerate:
                print("LOOP GENERATE")
            elif node.kind == ps.SyntaxKind.LoopStatement:
                print("LOOP STATEMENT")
            elif node.kind == ps.SyntaxKind.MacroActualArgument:
                print("MACRO ACTUAL ARGUMENT")
            elif node.kind == ps.SyntaxKind.MacroActualArgumentList:
                print("MACRO ACTUAL ARGUMENT LIST")
            elif node.kind == ps.SyntaxKind.MacroArgumentDefault:
                print("MACRO ARGUMENT DEFAULT")
            elif node.kind == ps.SyntaxKind.MacroFormalArgument:
                print("MACRO FORMAL ARGUMENT")
            elif node.kind == ps.SyntaxKind.MacroFormalArgumentList:
                print("MACRO FORMAL ARGUMENT LIST")
            elif node.kind == ps.SyntaxKind.MacroUsage:
                print("MACRO USAGE")
            elif node.kind == ps.SyntaxKind.MatchesClause:
                print("MATCHES CLAUSE")
            elif node.kind == ps.SyntaxKind.MemberAccessExpression:
                print("MEMBER ACCESS EXPRESSION")
            elif node.kind == ps.SyntaxKind.MinTypMaxExpression:
                print("MIN TYP MAX EXPRESSION")
            elif node.kind == ps.SyntaxKind.ModAssignmentExpression:
                print("MOD ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ModExpression:
                print("MOD EXPRESSION")
            elif node.kind == ps.SyntaxKind.ModportClockingPort:
                print("MODPORT CLOCKING PORT")
            elif node.kind == ps.SyntaxKind.ModportDeclaration:
                print("MODPORT DECLARATION")
            elif node.kind == ps.SyntaxKind.ModportExplicitPort:
                print("MODPORT EXPLICIT PORT")
            elif node.kind == ps.SyntaxKind.ModportItem:
                print("MODPORT ITEM")
            elif node.kind == ps.SyntaxKind.ModportNamedPort:
                print("MODPORT NAMED PORT")
            elif node.kind == ps.SyntaxKind.ModportSimplePortList:
                print("MODPORT SIMPLE PORT LIST")
            elif node.kind == ps.SyntaxKind.ModportSubroutinePort:
                print("MODPORT SUBROUTINE PORT")
            elif node.kind == ps.SyntaxKind.ModportSubroutinePortList:
                print("MODPORT SUBROUTINE PORT LIST")
            elif node.kind == ps.SyntaxKind.ModuleDeclaration:
                print("MODULE DECLARATION")
            elif node.kind == ps.SyntaxKind.ModuleHeader:
                print("MODULE HEADER")
            elif node.kind == ps.SyntaxKind.MultipleConcatenationExpression:
                print("MULTIPLE CONCATENATION EXPRESSION")
            elif node.kind == ps.SyntaxKind.MultiplyAssignmentExpression:
                print("MULTIPLY ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.MultiplyExpression:
                print("MULTIPLY EXPRESSION")
            elif node.kind == ps.SyntaxKind.NameValuePragmaExpression:
                print("NAME VALUE PRAGMA EXPRESSION")
            elif node.kind == ps.SyntaxKind.NamedArgument:
                print("NAMED ARGUMENT")
            elif node.kind == ps.SyntaxKind.NamedBlockClause:
                print("NAMED BLOCK CLAUSE")
            elif node.kind == ps.SyntaxKind.NamedConditionalDirectiveExpression:
                print("NAMED CONDITIONAL DIRECTIVE EXPRESSION")
            elif node.kind == ps.SyntaxKind.NamedLabel:
                print("NAMED LABEL")
            elif node.kind == ps.SyntaxKind.NamedParamAssignment:
                print("NAMED PARAM ASSIGNMENT")
            elif node.kind == ps.SyntaxKind.NamedPortConnection:
                print("NAMED PORT CONNECTION")
            elif node.kind == ps.SyntaxKind.NamedStructurePatternMember:
                print("NAMED STRUCTURE PATTERN MEMBER")
            elif node.kind == ps.SyntaxKind.NamedType:
                print("NAMED TYPE")
            elif node.kind == ps.SyntaxKind.NetAlias:
                print("NET ALIAS")
            elif node.kind == ps.SyntaxKind.NetDeclaration:
                print("NET DECLARATION")
            elif node.kind == ps.SyntaxKind.NetPortHeader:
                print("NET PORT HEADER")
            elif node.kind == ps.SyntaxKind.NetTypeDeclaration:
                print("NET TYPE DECLARATION")
            elif node.kind == ps.SyntaxKind.NewArrayExpression:
                print("NEW ARRAY EXPRESSION")
            elif node.kind == ps.SyntaxKind.NewClassExpression:
                print("NEW CLASS EXPRESSION")
            elif node.kind == ps.SyntaxKind.NoUnconnectedDriveDirective:
                print("NO UNCONNECTED DRIVE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.NonAnsiPortList:
                print("NON ANSI PORT LIST")
            elif node.kind == ps.SyntaxKind.NonAnsiUdpPortList:
                print("NON ANSI UDP PORT LIST")
            elif node.kind == ps.SyntaxKind.NonblockingAssignmentExpression:
                print("NONBLOCKING ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.NonblockingEventTriggerStatement:
                print("NONBLOCKING EVENT TRIGGER STATEMENT")
            elif node.kind == ps.SyntaxKind.NullLiteralExpression:
                print("NULL LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.NumberPragmaExpression:
                print("NUMBER PRAGMA EXPRESSION")
            elif node.kind == ps.SyntaxKind.OneStepDelay:
                print("ONE STEP DELAY")
            elif node.kind == ps.SyntaxKind.OrAssignmentExpression:
                print("OR ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.OrPropertyExpr:
                print("OR PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.OrSequenceExpr:
                print("OR SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.OrderedArgument:
                print("ORDERED ARGUMENT")
            elif node.kind == ps.SyntaxKind.OrderedParamAssignment:
                print("ORDERED PARAM ASSIGNMENT")
            elif node.kind == ps.SyntaxKind.OrderedPortConnection:
                print("ORDERED PORT CONNECTION")
            elif node.kind == ps.SyntaxKind.OrderedStructurePatternMember:
                print("ORDERED STRUCTURE PATTERN MEMBER")
            elif node.kind == ps.SyntaxKind.PackageDeclaration:
                print("PACKAGE DECLARATION")
            elif node.kind == ps.SyntaxKind.PackageExportAllDeclaration:
                print("PACKAGE EXPORT ALL DECLARATION")
            elif node.kind == ps.SyntaxKind.PackageExportDeclaration:
                print("PACKAGE EXPORT DECLARATION")
            elif node.kind == ps.SyntaxKind.PackageHeader:
                print("PACKAGE HEADER")
            elif node.kind == ps.SyntaxKind.PackageImportDeclaration:
                print("PACKAGE IMPORT DECLARATION")
            elif node.kind == ps.SyntaxKind.PackageImportItem:
                print("PACKAGE IMPORT ITEM")
            elif node.kind == ps.SyntaxKind.ParallelBlockStatement:
                print("PARALLEL BLOCK STATEMENT")
            elif node.kind == ps.SyntaxKind.ParameterDeclaration:
                print("PARAMETER DECLARATION")
            elif node.kind == ps.SyntaxKind.ParameterDeclarationStatement:
                print("PARAMETER DECLARATION STATEMENT")
            elif node.kind == ps.SyntaxKind.ParameterPortList:
                print("PARAMETER PORT LIST")
            elif node.kind == ps.SyntaxKind.ParameterValueAssignment:
                print("PARAMETER VALUE ASSIGNMENT")
            elif node.kind == ps.SyntaxKind.ParenExpressionList:
                print("PAREN EXPRESSION LIST")
            elif node.kind == ps.SyntaxKind.ParenPragmaExpression:
                print("PAREN PRAGMA EXPRESSION")
            elif node.kind == ps.SyntaxKind.ParenthesizedBinsSelectExpr:
                print("PARENTHESIZED BINS SELECT EXPR")
            elif node.kind == ps.SyntaxKind.ParenthesizedConditionalDirectiveExpression:
                print("PARENTHESIZED CONDITIONAL DIRECTIVE EXPRESSION")
            elif node.kind == ps.SyntaxKind.ParenthesizedEventExpression:
                print("PARENTHESIZED EVENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.ParenthesizedExpression:
                print("PARENTHESIZED EXPRESSION")
            elif node.kind == ps.SyntaxKind.ParenthesizedPattern:
                print("PARENTHESIZED PATTERN")
            elif node.kind == ps.SyntaxKind.ParenthesizedPropertyExpr:
                print("PARENTHESIZED PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ParenthesizedSequenceExpr:
                print("PARENTHESIZED SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.PathDeclaration:
                print("PATH DECLARATION")
            elif node.kind == ps.SyntaxKind.PathDescription:
                print("PATH DESCRIPTION")
            elif node.kind == ps.SyntaxKind.PatternCaseItem:
                print("PATTERN CASE ITEM")
            elif node.kind == ps.SyntaxKind.PortConcatenation:
                print("PORT CONCATENATION")
            elif node.kind == ps.SyntaxKind.PortDeclaration:
                print("PORT DECLARATION")
            elif node.kind == ps.SyntaxKind.PortReference:
                print("PORT REFERENCE")
            elif node.kind == ps.SyntaxKind.PostdecrementExpression:
                print("POSTDECREMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.PostincrementExpression:
                print("POSTINCREMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.PowerExpression:
                print("POWER EXPRESSION")
            elif node.kind == ps.SyntaxKind.PragmaDirective:
                print("PRAGMA DIRECTIVE")
            elif node.kind == ps.SyntaxKind.PrimaryBlockEventExpression:
                print("PRIMARY BLOCK EVENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.PrimitiveInstantiation:
                print("PRIMITIVE INSTANTIATION")
            elif node.kind == ps.SyntaxKind.ProceduralAssignStatement:
                print("PROCEDURAL ASSIGN STATEMENT")
            elif node.kind == ps.SyntaxKind.ProceduralDeassignStatement:
                print("PROCEDURAL DEASSIGN STATEMENT")
            elif node.kind == ps.SyntaxKind.ProceduralForceStatement:
                print("PROCEDURAL FORCE STATEMENT")
            elif node.kind == ps.SyntaxKind.ProceduralReleaseStatement:
                print("PROCEDURAL RELEASE STATEMENT")
            elif node.kind == ps.SyntaxKind.Production:
                print("PRODUCTION")
            elif node.kind == ps.SyntaxKind.ProgramDeclaration:
                print("PROGRAM DECLARATION")
            elif node.kind == ps.SyntaxKind.ProgramHeader:
                print("PROGRAM HEADER")
            elif node.kind == ps.SyntaxKind.PropertyDeclaration:
                print("PROPERTY DECLARATION")
            elif node.kind == ps.SyntaxKind.PropertySpec:
                print("PROPERTY SPEC")
            elif node.kind == ps.SyntaxKind.PropertyType:
                print("PROPERTY TYPE")
            elif node.kind == ps.SyntaxKind.ProtectDirective:
                print("PROTECT DIRECTIVE")
            elif node.kind == ps.SyntaxKind.ProtectedDirective:
                print("PROTECTED DIRECTIVE")
            elif node.kind == ps.SyntaxKind.PullStrength:
                print("PULL STRENGTH")
            elif node.kind == ps.SyntaxKind.PulseStyleDeclaration:
                print("PULSE STYLE DECLARATION")
            elif node.kind == ps.SyntaxKind.QueueDimensionSpecifier:
                print("QUEUE DIMENSION SPECIFIER")
            elif node.kind == ps.SyntaxKind.RandCaseItem:
                print("RAND CASE ITEM")
            elif node.kind == ps.SyntaxKind.RandCaseStatement:
                print("RAND CASE STATEMENT")
            elif node.kind == ps.SyntaxKind.RandJoinClause:
                print("RAND JOIN CLAUSE")
            elif node.kind == ps.SyntaxKind.RandSequenceStatement:
                print("RAND SEQUENCE STATEMENT")
            elif node.kind == ps.SyntaxKind.RangeCoverageBinInitializer:
                print("RANGE COVERAGE BIN INITIALIZER")
            elif node.kind == ps.SyntaxKind.RangeDimensionSpecifier:
                print("RANGE DIMENSION SPECIFIER")
            elif node.kind == ps.SyntaxKind.RangeList:
                print("RANGE LIST")
            elif node.kind == ps.SyntaxKind.RealLiteralExpression:
                print("REAL LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.RealTimeType:
                print("REAL TIME TYPE")
            elif node.kind == ps.SyntaxKind.RealType:
                print("REAL TYPE")
            elif node.kind == ps.SyntaxKind.RegType:
                print("REG TYPE")
            elif node.kind == ps.SyntaxKind.RepeatedEventControl:
                print("REPEATED EVENT CONTROL")
            elif node.kind == ps.SyntaxKind.ReplicatedAssignmentPattern:
                print("REPLICATED ASSIGNMENT PATTERN")
            elif node.kind == ps.SyntaxKind.ResetAllDirective:
                print("RESET ALL DIRECTIVE")
            elif node.kind == ps.SyntaxKind.RestrictPropertyStatement:
                print("RESTRICT PROPERTY STATEMENT")
            elif node.kind == ps.SyntaxKind.ReturnStatement:
                print("RETURN STATEMENT")
            elif node.kind == ps.SyntaxKind.RootScope:
                print("ROOT SCOPE")
            elif node.kind == ps.SyntaxKind.RsCase:
                print("RS CASE")
            elif node.kind == ps.SyntaxKind.RsCodeBlock:
                print("RS CODE BLOCK")
            elif node.kind == ps.SyntaxKind.RsElseClause:
                print("RS ELSE CLAUSE")
            elif node.kind == ps.SyntaxKind.RsIfElse:
                print("RS IF ELSE")
            elif node.kind == ps.SyntaxKind.RsProdItem:
                print("RS PROD ITEM")
            elif node.kind == ps.SyntaxKind.RsRepeat:
                print("RS REPEAT")
            elif node.kind == ps.SyntaxKind.RsRule:
                print("RS RULE")
            elif node.kind == ps.SyntaxKind.RsWeightClause:
                print("RS WEIGHT CLAUSE")
            elif node.kind == ps.SyntaxKind.SUntilPropertyExpr:
                print("S UNTIL PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.SUntilWithPropertyExpr:
                print("S UNTIL WITH PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.ScopedName:
                print("SCOPED NAME")
            elif node.kind == ps.SyntaxKind.SequenceDeclaration:
                print("SEQUENCE DECLARATION")
            elif node.kind == ps.SyntaxKind.SequenceMatchList:
                print("SEQUENCE MATCH LIST")
            elif node.kind == ps.SyntaxKind.SequenceRepetition:
                print("SEQUENCE REPETITION")
            elif node.kind == ps.SyntaxKind.SequenceType:
                print("SEQUENCE TYPE")
            elif node.kind == ps.SyntaxKind.SequentialBlockStatement:
                print("SEQUENTIAL BLOCK STATEMENT")
            elif node.kind == ps.SyntaxKind.ShortIntType:
                print("SHORT INT TYPE")
            elif node.kind == ps.SyntaxKind.ShortRealType:
                print("SHORT REAL TYPE")
            elif node.kind == ps.SyntaxKind.SignalEventExpression:
                print("SIGNAL EVENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.SignedCastExpression:
                print("SIGNED CAST EXPRESSION")
            elif node.kind == ps.SyntaxKind.SimpleAssignmentPattern:
                print("SIMPLE ASSIGNMENT PATTERN")
            elif node.kind == ps.SyntaxKind.SimpleBinsSelectExpr:
                print("SIMPLE BINS SELECT EXPR")
            elif node.kind == ps.SyntaxKind.SimplePathSuffix:
                print("SIMPLE PATH SUFFIX")
            elif node.kind == ps.SyntaxKind.SimplePragmaExpression:
                print("SIMPLE PRAGMA EXPRESSION")
            elif node.kind == ps.SyntaxKind.SimplePropertyExpr:
                print("SIMPLE PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.SimpleRangeSelect:
                print("SIMPLE RANGE SELECT")
            elif node.kind == ps.SyntaxKind.SimpleSequenceExpr:
                print("SIMPLE SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.SolveBeforeConstraint:
                print("SOLVE BEFORE CONSTRAINT")
            elif node.kind == ps.SyntaxKind.SpecifyBlock:
                print("SPECIFY BLOCK")
            elif node.kind == ps.SyntaxKind.SpecparamDeclaration:
                print("SPECPARAM DECLARATION")
            elif node.kind == ps.SyntaxKind.SpecparamDeclarator:
                print("SPECPARAM DECLARATOR")
            elif node.kind == ps.SyntaxKind.StandardCaseItem:
                print("STANDARD CASE ITEM")
            elif node.kind == ps.SyntaxKind.StandardPropertyCaseItem:
                print("STANDARD PROPERTY CASE ITEM")
            elif node.kind == ps.SyntaxKind.StandardRsCaseItem:
                print("STANDARD RS CASE ITEM")
            elif node.kind == ps.SyntaxKind.StreamExpression:
                print("STREAM EXPRESSION")
            elif node.kind == ps.SyntaxKind.StreamExpressionWithRange:
                print("STREAM EXPRESSION WITH RANGE")
            elif node.kind == ps.SyntaxKind.StreamingConcatenationExpression:
                print("STREAMING CONCATENATION EXPRESSION")
            elif node.kind == ps.SyntaxKind.StringLiteralExpression:
                print("STRING LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.StringType:
                print("STRING TYPE")
            elif node.kind == ps.SyntaxKind.StrongWeakPropertyExpr:
                print("STRONG WEAK PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.StructType:
                print("STRUCT TYPE")
            elif node.kind == ps.SyntaxKind.StructUnionMember:
                print("STRUCT UNION MEMBER")
            elif node.kind == ps.SyntaxKind.StructurePattern:
                print("STRUCTURE PATTERN")
            elif node.kind == ps.SyntaxKind.StructuredAssignmentPattern:
                print("STRUCTURED ASSIGNMENT PATTERN")
            elif node.kind == ps.SyntaxKind.SubtractAssignmentExpression:
                print("SUBTRACT ASSIGNMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.SubtractExpression:
                print("SUBTRACT EXPRESSION")
            elif node.kind == ps.SyntaxKind.SuperHandle:
                print("SUPER HANDLE")
            elif node.kind == ps.SyntaxKind.SuperNewDefaultedArgsExpression:
                print("SUPER NEW DEFAULTED ARGS EXPRESSION")
            elif node.kind == ps.SyntaxKind.SystemName:
                print("SYSTEM NAME")
            elif node.kind == ps.SyntaxKind.SystemTimingCheck:
                print("SYSTEM TIMING CHECK")
            elif node.kind == ps.SyntaxKind.TaggedPattern:
                print("TAGGED PATTERN")
            elif node.kind == ps.SyntaxKind.TaggedUnionExpression:
                print("TAGGED UNION EXPRESSION")
            elif node.kind == ps.SyntaxKind.TaskDeclaration:
                print("TASK DECLARATION")
            elif node.kind == ps.SyntaxKind.ThisHandle:
                print("THIS HANDLE")
            elif node.kind == ps.SyntaxKind.ThroughoutSequenceExpr:
                print("THROUGHOUT SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.TimeLiteralExpression:
                print("TIME LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.TimeScaleDirective:
                print("TIME SCALE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.TimeType:
                print("TIME TYPE")
            elif node.kind == ps.SyntaxKind.TimeUnitsDeclaration:
                print("TIME UNITS DECLARATION")
            elif node.kind == ps.SyntaxKind.TimingCheckEventArg:
                print("TIMING CHECK EVENT ARG")
            elif node.kind == ps.SyntaxKind.TimingCheckEventCondition:
                print("TIMING CHECK EVENT CONDITION")
            elif node.kind == ps.SyntaxKind.TimingControlExpression:
                print("TIMING CONTROL EXPRESSION")
            elif node.kind == ps.SyntaxKind.TimingControlStatement:
                print("TIMING CONTROL STATEMENT")
            elif node.kind == ps.SyntaxKind.TransListCoverageBinInitializer:
                print("TRANS LIST COVERAGE BIN INITIALIZER")
            elif node.kind == ps.SyntaxKind.TransRange:
                print("TRANS RANGE")
            elif node.kind == ps.SyntaxKind.TransRepeatRange:
                print("TRANS REPEAT RANGE")
            elif node.kind == ps.SyntaxKind.TransSet:
                print("TRANS SET")
            elif node.kind == ps.SyntaxKind.TypeAssignment:
                print("TYPE ASSIGNMENT")
            elif node.kind == ps.SyntaxKind.TypeParameterDeclaration:
                print("TYPE PARAMETER DECLARATION")
            elif node.kind == ps.SyntaxKind.TypeReference:
                print("TYPE REFERENCE")
            elif node.kind == ps.SyntaxKind.TypedefDeclaration:
                print("TYPEDEF DECLARATION")
            elif node.kind == ps.SyntaxKind.UdpBody:
                print("UDP BODY")
            elif node.kind == ps.SyntaxKind.UdpDeclaration:
                print("UDP DECLARATION")
            elif node.kind == ps.SyntaxKind.UdpEdgeField:
                print("UDP EDGE FIELD")
            elif node.kind == ps.SyntaxKind.UdpEntry:
                print("UDP ENTRY")
            elif node.kind == ps.SyntaxKind.UdpInitialStmt:
                print("UDP INITIAL STMT")
            elif node.kind == ps.SyntaxKind.UdpInputPortDecl:
                print("UDP INPUT PORT DECL")
            elif node.kind == ps.SyntaxKind.UdpOutputPortDecl:
                print("UDP OUTPUT PORT DECL")
            elif node.kind == ps.SyntaxKind.UdpSimpleField:
                print("UDP SIMPLE FIELD")
            elif node.kind == ps.SyntaxKind.UnaryBinsSelectExpr:
                print("UNARY BINS SELECT EXPR")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseAndExpression:
                print("UNARY BITWISE AND EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseNandExpression:
                print("UNARY BITWISE NAND EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseNorExpression:
                print("UNARY BITWISE NOR EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseNotExpression:
                print("UNARY BITWISE NOT EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseOrExpression:
                print("UNARY BITWISE OR EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseXnorExpression:
                print("UNARY BITWISE XNOR EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryBitwiseXorExpression:
                print("UNARY BITWISE XOR EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryConditionalDirectiveExpression:
                print("UNARY CONDITIONAL DIRECTIVE EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryLogicalNotExpression:
                print("UNARY LOGICAL NOT EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryMinusExpression:
                print("UNARY MINUS EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryPlusExpression:
                print("UNARY PLUS EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryPredecrementExpression:
                print("UNARY PREDECREMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryPreincrementExpression:
                print("UNARY PREINCREMENT EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnaryPropertyExpr:
                print("UNARY PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.UnarySelectPropertyExpr:
                print("UNARY SELECT PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.UnbasedUnsizedLiteralExpression:
                print("UNBASED UNSIZED LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.UnconnectedDriveDirective:
                print("UNCONNECTED DRIVE DIRECTIVE")
            elif node.kind == ps.SyntaxKind.UndefDirective:
                print("UNDEF DIRECTIVE")
            elif node.kind == ps.SyntaxKind.UndefineAllDirective:
                print("UNDEFINE ALL DIRECTIVE")
            elif node.kind == ps.SyntaxKind.UnionType:
                print("UNION TYPE")
            elif node.kind == ps.SyntaxKind.UniquenessConstraint:
                print("UNIQUENESS CONSTRAINT")
            elif node.kind == ps.SyntaxKind.UnitScope:
                print("UNIT SCOPE")
            elif node.kind == ps.SyntaxKind.UntilPropertyExpr:
                print("UNTIL PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.UntilWithPropertyExpr:
                print("UNTIL WITH PROPERTY EXPR")
            elif node.kind == ps.SyntaxKind.Untyped:
                print("UNTYPED")
            elif node.kind == ps.SyntaxKind.UserDefinedNetDeclaration:
                print("USER DEFINED NET DECLARATION")
            elif node.kind == ps.SyntaxKind.ValueRangeExpression:
                print("VALUE RANGE EXPRESSION")
            elif node.kind == ps.SyntaxKind.VariableDimension:
                print("VARIABLE DIMENSION")
            elif node.kind == ps.SyntaxKind.VariablePattern:
                print("VARIABLE PATTERN")
            elif node.kind == ps.SyntaxKind.VariablePortHeader:
                print("VARIABLE PORT HEADER")
            elif node.kind == ps.SyntaxKind.VirtualInterfaceType:
                print("VIRTUAL INTERFACE TYPE")
            elif node.kind == ps.SyntaxKind.VoidCastedCallStatement:
                print("VOID CASTED CALL STATEMENT")
            elif node.kind == ps.SyntaxKind.VoidType:
                print("VOID TYPE")
            elif node.kind == ps.SyntaxKind.WaitForkStatement:
                print("WAIT FORK STATEMENT")
            elif node.kind == ps.SyntaxKind.WaitOrderStatement:
                print("WAIT ORDER STATEMENT")
            elif node.kind == ps.SyntaxKind.WaitStatement:
                print("WAIT STATEMENT")
            elif node.kind == ps.SyntaxKind.WildcardDimensionSpecifier:
                print("WILDCARD DIMENSION SPECIFIER")
            elif node.kind == ps.SyntaxKind.WildcardEqualityExpression:
                print("WILDCARD EQUALITY EXPRESSION")
            elif node.kind == ps.SyntaxKind.WildcardInequalityExpression:
                print("WILDCARD INEQUALITY EXPRESSION")
            elif node.kind == ps.SyntaxKind.WildcardLiteralExpression:
                print("WILDCARD LITERAL EXPRESSION")
            elif node.kind == ps.SyntaxKind.WildcardPattern:
                print("WILDCARD PATTERN")
            elif node.kind == ps.SyntaxKind.WildcardPortConnection:
                print("WILDCARD PORT CONNECTION")
            elif node.kind == ps.SyntaxKind.WildcardPortList:
                print("WILDCARD PORT LIST")
            elif node.kind == ps.SyntaxKind.WildcardUdpPortList:
                print("WILDCARD UDP PORT LIST")
            elif node.kind == ps.SyntaxKind.WithClause:
                print("WITH CLAUSE")
            elif node.kind == ps.SyntaxKind.WithFunctionClause:
                print("WITH FUNCTION CLAUSE")
            elif node.kind == ps.SyntaxKind.WithFunctionSample:
                print("WITH FUNCTION SAMPLE")
            elif node.kind == ps.SyntaxKind.WithinSequenceExpr:
                print("WITHIN SEQUENCE EXPR")
            elif node.kind == ps.SyntaxKind.XorAssignmentExpression:
                print("XOR ASSIGNMENT EXPRESSION")



        #self.process_node_for_predicates(node)
        self.process_node_for_name(node)
        try:
            self.kind_to_node_ids[node.kind].append(self.node_id)
        except KeyError:
            self.kind_to_node_ids[node.kind] = [self.node_id]

        self.num_children_processed += 1
        self.processed_children_ids.append(self.node_id)
        try:
            for i in range(len(node)):
                self.num_children_in_next_level += 1
                child_id = self.node_id + (self.num_children_in_level - self.num_children_processed) + self.num_children_in_next_level
                self.node_id_to_pid[child_id] = self.node_id
                self.node_id_to_cids[self.node_id].append(child_id)

                if use_queue:
                    self.queue.append(node.__getitem__(i))
        except TypeError:
            # This exception is required because, unlike SyntaxNode, Token
            # objects do not have a len() function (i.e., getChildCount)
            pass
        

        if self.num_children_processed == self.num_children_in_level:
            self.level += 1
            self.num_children_in_level = self.num_children_in_next_level
            self.num_children_in_next_level = 0
            self.num_children_processed = 0
            self.processed_children_ids = list()

        self.node_id += 1