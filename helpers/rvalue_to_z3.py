"""Helpers for working with Z3: semantic expression conversion and solving."""

import z3
from z3 import Solver, BitVec, BitVecRef, If, BitVecVal, And, Or, Not, ULT, UGT, BoolRef
import pyslang.ast as ps_ast


SOLVE_PC_TIMEOUT_MS = 10000

def solve_pc(s: Solver) -> bool:
    """Solve path condition. Returns True iff sat; False for unsat or timeout (unknown)."""
    try:
        s.set("timeout", SOLVE_PC_TIMEOUT_MS)
    except Exception:
        pass
    result = str(s.check())
    if result == "sat":
        return True
    if result == "unknown":
        return False
    return False


# ---------------------------------------------------------------------------
# Semantic Expression → Z3 converter
# ---------------------------------------------------------------------------
# Works directly with pyslang's semantic Expression objects (the nodes stored
# in CFG basic blocks).  This replaces the old syntax-based tokenizer path
# for any call site that has semantic nodes.
# ---------------------------------------------------------------------------

def _parse_svint(sv) -> int:
    """Convert a pyslang SVInt to a Python int."""
    s = str(sv).strip()
    if not s:
        return 0
    if "'" not in s:
        return int(s)
    parts = s.split("'", 1)
    base_char = parts[1][0].lower() if parts[1] else 'd'
    digits = parts[1][1:] if len(parts[1]) > 1 else '0'
    bases = {'b': 2, 'o': 8, 'd': 10, 'h': 16}
    base = bases.get(base_char, 10)
    clean = digits.replace('_', '').replace('?', '0')
    clean = clean.replace('x', '0').replace('X', '0')
    clean = clean.replace('z', '0').replace('Z', '0')
    return int(clean, base) if clean else 0


_BINOP_MAP = {
    ps_ast.BinaryOperator.Add:                lambda a, b: a + b,
    ps_ast.BinaryOperator.Subtract:           lambda a, b: a - b,
    ps_ast.BinaryOperator.Multiply:           lambda a, b: a * b,
    ps_ast.BinaryOperator.BinaryAnd:          lambda a, b: a & b,
    ps_ast.BinaryOperator.BinaryOr:           lambda a, b: a | b,
    ps_ast.BinaryOperator.BinaryXor:          lambda a, b: a ^ b,
    ps_ast.BinaryOperator.BinaryXnor:         lambda a, b: ~(a ^ b),
    ps_ast.BinaryOperator.Equality:           lambda a, b: a == b,
    ps_ast.BinaryOperator.Inequality:         lambda a, b: a != b,
    ps_ast.BinaryOperator.CaseEquality:       lambda a, b: a == b,
    ps_ast.BinaryOperator.CaseInequality:     lambda a, b: a != b,
    ps_ast.BinaryOperator.WildcardEquality:   lambda a, b: a == b,
    ps_ast.BinaryOperator.WildcardInequality: lambda a, b: a != b,
    ps_ast.BinaryOperator.LessThan:           lambda a, b: ULT(a, b),
    ps_ast.BinaryOperator.LessThanEqual:      lambda a, b: z3.ULE(a, b),
    ps_ast.BinaryOperator.GreaterThan:        lambda a, b: UGT(a, b),
    ps_ast.BinaryOperator.GreaterThanEqual:   lambda a, b: z3.UGE(a, b),
    ps_ast.BinaryOperator.LogicalAnd:         lambda a, b: And(a != 0, b != 0)
                                                        if not isinstance(a, BoolRef)
                                                        else And(a, b if isinstance(b, BoolRef) else b != 0),
    ps_ast.BinaryOperator.LogicalOr:          lambda a, b: Or(a != 0, b != 0)
                                                        if not isinstance(a, BoolRef)
                                                        else Or(a, b if isinstance(b, BoolRef) else b != 0),
    ps_ast.BinaryOperator.LogicalShiftLeft:   lambda a, b: a << b,
    ps_ast.BinaryOperator.LogicalShiftRight:  lambda a, b: z3.LShR(a, b),
    ps_ast.BinaryOperator.ArithmeticShiftLeft:  lambda a, b: a << b,
    ps_ast.BinaryOperator.ArithmeticShiftRight: lambda a, b: a >> b,
}

_UNOP_MAP = {
    ps_ast.UnaryOperator.LogicalNot:  lambda a: a == BitVecVal(0, a.size()) if isinstance(a, BitVecRef) else Not(a),
    ps_ast.UnaryOperator.BitwiseNot:  lambda a: ~a,
    ps_ast.UnaryOperator.Plus:        lambda a: a,
    ps_ast.UnaryOperator.Minus:       lambda a: -a,
    ps_ast.UnaryOperator.BitwiseAnd:  lambda a: z3.BVRedAnd(a),
    ps_ast.UnaryOperator.BitwiseOr:   lambda a: z3.BVRedOr(a),
    ps_ast.UnaryOperator.BitwiseXor:  None,  # no single Z3 call
    ps_ast.UnaryOperator.BitwiseNand: lambda a: ~z3.BVRedAnd(a),
    ps_ast.UnaryOperator.BitwiseNor:  lambda a: ~z3.BVRedOr(a),
}


def semantic_expr_to_z3(expr, store: dict, module: str, width_hint: int = 32):
    """Convert a pyslang semantic Expression to a Z3 BitVecRef/BoolRef.

    *store* is ``state.store[module]`` — maps signal names to symbolic names
    (strings) or Z3 expressions.
    *width_hint* is the default bit-width when the expression doesn't carry one.

    Returns a Z3 expression or None on failure.
    """
    if expr is None:
        return None

    kind = expr.kind
    w = getattr(expr, 'effectiveWidth', None) or width_hint

    # --- Leaf nodes --------------------------------------------------------
    if kind == ps_ast.ExpressionKind.NamedValue:
        name = expr.symbol.name
        sym = store.get(name, name)
        if isinstance(sym, (BitVecRef, BoolRef, z3.ArithRef)):
            return sym
        sym_str = str(sym)
        if sym_str.lstrip('-').isdigit():
            return BitVecVal(int(sym_str), w)
        return BitVec(sym_str, w)

    if kind == ps_ast.ExpressionKind.IntegerLiteral:
        return BitVecVal(_parse_svint(expr.value), w)

    if kind == ps_ast.ExpressionKind.UnbasedUnsizedIntegerLiteral:
        return BitVecVal(_parse_svint(expr.value), w)

    # --- Conversion (width cast / sign cast) --------------------------------
    if kind == ps_ast.ExpressionKind.Conversion:
        inner = semantic_expr_to_z3(expr.operand, store, module, w)
        if inner is None:
            return None
        if isinstance(inner, BoolRef):
            inner = If(inner, BitVecVal(1, 1), BitVecVal(0, 1))
        iw = inner.size() if isinstance(inner, BitVecRef) else w
        if iw < w:
            return z3.ZeroExt(w - iw, inner)
        if iw > w:
            return z3.Extract(w - 1, 0, inner)
        return inner

    # --- Binary operators ---------------------------------------------------
    if kind == ps_ast.ExpressionKind.BinaryOp:
        op = expr.op
        lhs = semantic_expr_to_z3(expr.left, store, module, w)
        rhs = semantic_expr_to_z3(expr.right, store, module, w)
        if lhs is None or rhs is None:
            return None
        # Widen/narrow to match
        if isinstance(lhs, BoolRef):
            lhs = If(lhs, BitVecVal(1, w), BitVecVal(0, w))
        if isinstance(rhs, BoolRef):
            rhs = If(rhs, BitVecVal(1, w), BitVecVal(0, w))
        if isinstance(lhs, BitVecRef) and isinstance(rhs, BitVecRef):
            if lhs.size() != rhs.size():
                target = max(lhs.size(), rhs.size())
                if lhs.size() < target:
                    lhs = z3.ZeroExt(target - lhs.size(), lhs)
                if rhs.size() < target:
                    rhs = z3.ZeroExt(target - rhs.size(), rhs)
        fn = _BINOP_MAP.get(op)
        if fn is not None:
            return fn(lhs, rhs)
        return None

    # --- Unary operators ----------------------------------------------------
    if kind == ps_ast.ExpressionKind.UnaryOp:
        inner = semantic_expr_to_z3(expr.operand, store, module, w)
        if inner is None:
            return None
        fn = _UNOP_MAP.get(expr.op)
        if fn is not None:
            return fn(inner)
        return None

    # --- Range select  (e.g. id_insn[31:26]) --------------------------------
    if kind == ps_ast.ExpressionKind.RangeSelect:
        base = semantic_expr_to_z3(expr.value, store, module, width_hint)
        if base is None:
            return None
        left_expr = expr.left
        right_expr = expr.right
        try:
            hi = _parse_svint(left_expr.value) if hasattr(left_expr, 'value') else int(str(left_expr.constant))
            lo = _parse_svint(right_expr.value) if hasattr(right_expr, 'value') else int(str(right_expr.constant))
        except Exception:
            return None
        if isinstance(base, BoolRef):
            base = If(base, BitVecVal(1, width_hint), BitVecVal(0, width_hint))
        if isinstance(base, BitVecRef):
            if hi >= base.size():
                hi = base.size() - 1
            if lo < 0:
                lo = 0
            return z3.Extract(hi, lo, base)
        return None

    # --- Element select  (e.g. sig[idx]) ------------------------------------
    if kind == ps_ast.ExpressionKind.ElementSelect:
        base = semantic_expr_to_z3(expr.value, store, module, width_hint)
        bw = base.size() if isinstance(base, BitVecRef) else width_hint
        idx_expr = semantic_expr_to_z3(expr.selector, store, module, bw)
        if base is None or idx_expr is None:
            return None
        if isinstance(base, BitVecRef) and isinstance(idx_expr, BitVecRef):
            if idx_expr.size() != base.size():
                if idx_expr.size() < base.size():
                    idx_expr = z3.ZeroExt(base.size() - idx_expr.size(), idx_expr)
                else:
                    idx_expr = z3.Extract(base.size() - 1, 0, idx_expr)
            return z3.LShR(base, idx_expr) & BitVecVal(1, base.size())
        return None

    # --- Ternary  (cond ? a : b) --------------------------------------------
    if kind == ps_ast.ExpressionKind.ConditionalOp:
        pred = semantic_expr_to_z3(expr.predicate, store, module, w)
        t_val = semantic_expr_to_z3(expr.left, store, module, w)
        f_val = semantic_expr_to_z3(expr.right, store, module, w)
        if pred is None or t_val is None or f_val is None:
            return None
        if isinstance(pred, BitVecRef):
            pred = pred != BitVecVal(0, pred.size())
        return If(pred, t_val, f_val)

    # --- Concatenation  ({a, b, c}) -----------------------------------------
    if kind == ps_ast.ExpressionKind.Concatenation:
        parts = []
        for op_expr in expr.operands:
            p = semantic_expr_to_z3(op_expr, store, module, width_hint)
            if p is None:
                return None
            if isinstance(p, BoolRef):
                p = If(p, BitVecVal(1, 1), BitVecVal(0, 1))
            parts.append(p)
        if len(parts) == 1:
            return parts[0]
        return z3.Concat(*parts)

    # --- Replication  ({N{expr}}) -------------------------------------------
    if kind == ps_ast.ExpressionKind.Replication:
        count_expr = expr.count
        inner = semantic_expr_to_z3(expr.concat, store, module, width_hint)
        if inner is None:
            return None
        try:
            n = _parse_svint(count_expr.value) if hasattr(count_expr, 'value') else 1
        except Exception:
            n = 1
        if n <= 1:
            return inner
        return z3.Concat(*([inner] * n))

    # --- Fallback: try to evaluate via constant property --------------------
    try:
        cv = expr.constant
        sv = cv.integer() if hasattr(cv, 'integer') else cv
        return BitVecVal(_parse_svint(sv), w)
    except Exception:
        pass

    return None

