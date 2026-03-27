import pyslang.ast as ps_ast
from helpers.visitor_helpers import handles, build_lookup_table


def _expr_to_label(expr):
    """Extract a human-readable source string from a pyslang Expression."""
    syntax = getattr(expr, 'syntax', None)
    if syntax is not None:
        text = str(syntax).strip()
        if text:
            return text
    operand = getattr(expr, 'operand', None)
    if operand is not None:
        return _expr_to_label(operand)
    return str(expr)


class CaseLabel(list):
    """A list of case-item expressions that also carries a readable label.

    Behaves identically to a plain list (so downstream code that iterates
    over expressions, checks ``isinstance(..., list)``, etc. keeps working)
    but ``str()`` / ``repr()`` return the Verilog source text.
    """
    def __init__(self, exprs, label=None):
        super().__init__(exprs)
        if label is not None:
            self._label = label
        else:
            self._label = ", ".join(_expr_to_label(e) for e in exprs) or "?"

    def __str__(self):
        return self._label

    def __repr__(self):
        return self._label


class DefaultLabel(dict):
    """A dict carrying the ``default_from`` expression list with a readable label."""
    def __init__(self, default_from_exprs):
        labels = ", ".join(_expr_to_label(e) for e in default_from_exprs) or "?"
        super().__init__(default_from=default_from_exprs)
        self._label = f"default (not {labels})"

    def __str__(self):
        return self._label

    def __repr__(self):
        return self._label


class BasicBlockVisitor:
    """
    Visitor to partition a Statement tree (from a ProceduralBlock symbol) into
    basic blocks and identify branch/partition points. Uses StatementKind for
    dispatch via a lookup table populated by the @handles decorator.
    """
    def __init__(self, cfg_manager):
        self.cfg = cfg_manager
        self.edge_stack = []
        self._scope = None  # Set by CFG.build_cfg before visiting
        self.lookup_table = build_lookup_table(self)

    ### STATEMENT HANDLERS ###

    @handles(ps_ast.StatementKind.Conditional)
    def handle_conditional(self, node):
        """Handle if/else and else-if chains with partition points."""
        cond_expr = node.conditions[0].expr if node.conditions else node
        parent_idx = self._add_node(cond_expr)
        self.cfg.partition_points.add(parent_idx)

        # if branch
        then_start_idx = self.cfg.curr_idx
        self.cfg.partition_points.add(then_start_idx)
        if node.ifTrue is not None:
            node.ifTrue.visit(lookup_table=self.lookup_table)
        self.cfg.edgelist.append((parent_idx, then_start_idx, "true"))

        # Preserve true-branch dangling edges so they reconnect after the
        # entire if/else (independent sequential conditionals).
        true_branch_dangling = list(self.edge_stack)

        # else / else-if branch
        if_false = node.ifFalse
        if if_false is not None:
            self.edge_stack.clear()
            if if_false.kind == ps_ast.StatementKind.Conditional:
                # else-if:
                self.cfg.edgelist.append((parent_idx, self.cfg.curr_idx, "false"))
                if_false.visit(lookup_table=self.lookup_table)
            else:
                #  else
                else_start_idx = self.cfg.curr_idx
                self.cfg.partition_points.add(else_start_idx)
                if_false.visit(lookup_table=self.lookup_table)
                self.cfg.edgelist.append((parent_idx, else_start_idx, "false"))
            # Restore both branches' dangling edges so the next statement
            # after the if/else connects to both.
            self.edge_stack.extend(true_branch_dangling)
        else:
            self.edge_stack.append((parent_idx, "false"))

        return ps_ast.VisitAction.Skip

    @handles(ps_ast.StatementKind.Case)
    def handle_case(self, node):
        """Handle case/casez/casex statement with default support."""
        parent_idx = self._add_node(node.expr)
        self.cfg.partition_points.add(parent_idx)

        case_cond = getattr(node, 'condition', None)
        case_kind_str = ""
        if case_cond == ps_ast.CaseStatementCondition.WildcardJustZ:
            case_kind_str = "casez"
        elif case_cond == ps_ast.CaseStatementCondition.WildcardXOrZ:
            case_kind_str = "casex"
        elif case_cond == ps_ast.CaseStatementCondition.Inside:
            case_kind_str = "case inside"

        all_standard_exprs = []
        persisted_dangling_edge_stack = []

        for item in node.items:
            case_start_idx = self.cfg.curr_idx
            self.cfg.partition_points.add(case_start_idx)

            branch_exprs = CaseLabel(item.expressions)
            branch_exprs.case_kind = case_kind_str
            all_standard_exprs.extend(branch_exprs)

            item.stmt.visit(lookup_table=self.lookup_table)
            self.cfg.edgelist.append((parent_idx, case_start_idx, branch_exprs))

            persisted_dangling_edge_stack.extend(self.edge_stack)
            self.edge_stack.clear()

        # Process default case item
        if node.defaultCase is not None:
            case_start_idx = self.cfg.curr_idx
            self.cfg.partition_points.add(case_start_idx)

            node.defaultCase.visit(lookup_table=self.lookup_table)
            self.cfg.edgelist.append((parent_idx, case_start_idx, DefaultLabel(all_standard_exprs)))

            persisted_dangling_edge_stack.extend(self.edge_stack)
            self.edge_stack.clear()

        self.edge_stack = persisted_dangling_edge_stack
        return ps_ast.VisitAction.Skip

    @handles(ps_ast.StatementKind.ForLoop)
    def handle_for_loop(self, node):
        """Handle for-loop by statically unrolling it N times."""
        if self._scope is None:
            raise ValueError("No scope set; cannot unroll loop")

        ast_ctx = ps_ast.ASTContext(self._scope, ps_ast.LookupLocation.max)
        eval_ctx = ps_ast.EvalContext(ast_ctx)

        # emit initializers and get initial loop variable value
        if node.loopVars:
            lv = node.loopVars[0]
            init_val = int(lv.initializer.eval(eval_ctx).value)
            idx = self._add_node(lv)
            self.edge_stack.append((idx, None))
        else:
            for init_stmt in node.initializers:
                init_stmt.visit(lookup_table=self.lookup_table)
            init_val = int(node.initializers[0].right.eval(eval_ctx).value)

        # Evaluate constant bound
        stop = node.stopExpr
        if stop is None:
            raise ValueError("ForLoop has no stop expression")
        bound_val = int(stop.right.eval(eval_ctx).value)
        step_size = self._get_step_size(node, eval_ctx)

        # unroll the loop
        current_val = init_val
        unroll_count = 0
        while self._loop_guard_holds(current_val, bound_val, stop.op):
            node.body.visit(lookup_table=self.lookup_table)
            self._emit_step(node)
            current_val += step_size
            unroll_count += 1

        return ps_ast.VisitAction.Skip

    # @handles(ps.StatementKind.DoWhileLoop)
    # def handle_do_while(self, node):
    #     """Handle do-while loop."""
    #     ...

    # @handles(ps.StatementKind.ForeachLoop)
    # def handle_foreach_loop(self, node):
    #     """Handle foreach loop."""
    #     ...

    @handles(
        ps_ast.StatementKind.ExpressionStatement,
        ps_ast.StatementKind.VariableDeclaration,
        ps_ast.StatementKind.ProceduralAssign,
    )
    def handle_leaf_statement(self, node):
        """Catch-all for leaf statements (assignments, declarations, etc.)."""
        idx = self._add_node(node)
        self.edge_stack.append((idx, None))
        return ps_ast.VisitAction.Skip

    ### HELPERS ###

    def _add_node(self, node):
        """Store a node in the CFG and return its index, connecting dangling edges.

        When multiple edges converge (edge_stack has >1 entry), this node is a
        merge point and must start a new basic block.
        """
        self.cfg.all_nodes.append(node)
        idx = self.cfg.curr_idx
        self.cfg.curr_idx += 1

        if len(self.edge_stack) > 1:
            self.cfg.partition_points.add(idx)

        while self.edge_stack:
            node_idx, condition = self.edge_stack.pop()
            if node is not None:
                self.cfg.edgelist.append((node_idx, idx, condition))

        return idx

    def _get_step_size(self, node, eval_ctx) -> int:
        """Return the signed integer step size for the first step expression."""
        step = node.steps[0]
        if step.kind == ps_ast.ExpressionKind.UnaryOp:
            if step.op in (ps_ast.UnaryOperator.Postincrement, ps_ast.UnaryOperator.Preincrement):
                return 1
            if step.op in (ps_ast.UnaryOperator.Postdecrement, ps_ast.UnaryOperator.Predecrement):
                return -1
        if step.kind == ps_ast.ExpressionKind.Assignment:
            rhs = step.right
            arith_op = step.op if step.op is not None else getattr(rhs, 'op', None)
            if hasattr(rhs, 'right'):
                rhs = rhs.right
            cv = rhs.eval(eval_ctx)
            if cv is None or cv.value is None:
                raise ValueError(
                    f"Cannot evaluate step delta as constant: {str(step.syntax)!r}"
                )
            delta = int(cv.value)
            if arith_op == ps_ast.BinaryOperator.Add:
                return delta
            if arith_op == ps_ast.BinaryOperator.Subtract:
                return -delta
        raise ValueError(f"Unsupported loop step kind: {step.kind}, op: {getattr(step, 'op', '?')}")

    def _loop_guard_holds(self, curr: int, bound: int, op) -> bool:
        """Return True while the loop guard condition holds (loop should continue)."""
        if op == ps_ast.BinaryOperator.LessThan:            return curr < bound
        if op == ps_ast.BinaryOperator.LessThanEqual:       return curr <= bound
        if op == ps_ast.BinaryOperator.GreaterThan:         return curr > bound
        if op == ps_ast.BinaryOperator.GreaterThanEqual:    return curr >= bound
        raise ValueError(f"Unsupported loop guard operator: {op}")

    def _emit_step(self, node):
        if not node.steps:
            return None

        for step_expr in node.steps:
            idx = self._add_node(step_expr)
            self.edge_stack.append((idx, None))
