"""Helpers for working with Z3: semantic expression conversion and solving."""

import z3
from typing import Optional, Tuple
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


def _hex_char_value(ch: str) -> Optional[int]:
    if ch in "0123456789":
        return int(ch)
    if ch in "aA":
        return 10
    if ch in "bB":
        return 11
    if ch in "cC":
        return 12
    if ch in "dD":
        return 13
    if ch in "eE":
        return 14
    if ch in "fF":
        return 15
    return None


def _wildcard_literal_mask_and_pat(sv, casex: bool) -> Optional[Tuple[int, int, int]]:
    """Parse ``casez``/``casex`` pattern literals for masked Z3 equality.

    Supports sized **binary**, **hex**, and **oct** literals. **Decimal** (``d``)
    with wildcards returns ``None`` (caller falls back to full-vector ``==``).

    Returns ``(care_mask, pat_bits, width)`` — same semantics as the former
    binary-only helper. Hex/oct expand each digit to 4 or 3 bits (MS digit =
    MS bits of the vector).

    - **casez** (``casex=False``): per-bit DCs ``?zZ`` in ``b``; per-nibble DCs
      ``?zZ`` in ``h``/``o``; ``x``/``X`` are **cared** (2-state 0), matching
      ``_parse_svint``-style abstraction.
    - **casex** (``casex=True``): same radices; ``x``/``X`` are also DC (per bit
      in ``b``, per nibble/tribble in ``h``/``o``). Pass ``casex=True`` from
      ``case_statement_arm_matches_z3`` when ``case_kind == \"casex\"``.
    """
    s = str(sv).strip().replace("_", "")
    if "'" not in s:
        return None
    head, body = s.split("'", 1)
    if not body:
        return None
    base_char = body[0].lower()
    digits = body[1:]
    if not digits:
        return None
    if base_char == "d":
        return None

    if base_char == "b":
        return _wildcard_mask_binary(digits, head, casex)
    if base_char == "h":
        return _wildcard_mask_hex(digits, head, casex)
    if base_char == "o":
        return _wildcard_mask_oct(digits, head, casex)
    return None


def _wildcard_mask_binary(digits: str, head: str, casex: bool) -> Optional[Tuple[int, int, int]]:
    if not head:
        width = len(digits)
    else:
        try:
            width = int(head)
        except ValueError:
            return None
    if len(digits) != width:
        return None
    care = 0
    pat = 0
    for i, ch in enumerate(digits):
        bit_idx = width - 1 - i
        if casex:
            if ch in "?zZxX":
                continue
        else:
            if ch in "?zZ":
                continue
        care |= 1 << bit_idx
        if ch == "1":
            pat |= 1 << bit_idx
        elif ch == "0":
            pass
        elif not casex and ch in "xX":
            pass
        else:
            return None
    return care, pat, width


def _wildcard_mask_hex(digits: str, head: str, casex: bool) -> Optional[Tuple[int, int, int]]:
    if not head:
        width = len(digits) * 4
    else:
        try:
            width = int(head)
        except ValueError:
            return None
    if len(digits) * 4 != width:
        return None
    care = 0
    pat = 0
    bit_index = width
    for ch in digits:
        lo = bit_index - 4
        if ch in "?zZ" or (casex and ch in "xX"):
            bit_index = lo
            continue
        if not casex and ch in "xX":
            for k in range(4):
                bit_idx = lo + 3 - k
                care |= 1 << bit_idx
            bit_index = lo
            continue
        nib = _hex_char_value(ch)
        if nib is None:
            return None
        for k in range(4):
            bit_idx = lo + 3 - k
            care |= 1 << bit_idx
            if (nib >> (3 - k)) & 1:
                pat |= 1 << bit_idx
        bit_index = lo
    if bit_index != 0:
        return None
    return care, pat, width


def _wildcard_mask_oct(digits: str, head: str, casex: bool) -> Optional[Tuple[int, int, int]]:
    if not head:
        width = len(digits) * 3
    else:
        try:
            width = int(head)
        except ValueError:
            return None
    if len(digits) * 3 != width:
        return None
    care = 0
    pat = 0
    bit_index = width
    for ch in digits:
        lo = bit_index - 3
        if ch in "?zZ" or (casex and ch in "xX"):
            bit_index = lo
            continue
        if not casex and ch in "xX":
            for k in range(3):
                bit_idx = lo + 2 - k
                care |= 1 << bit_idx
            bit_index = lo
            continue
        if ch not in "01234567":
            return None
        nib = int(ch)
        for k in range(3):
            bit_idx = lo + 2 - k
            care |= 1 << bit_idx
            if (nib >> (2 - k)) & 1:
                pat |= 1 << bit_idx
        bit_index = lo
    if bit_index != 0:
        return None
    return care, pat, width


# Backwards-compatible name used in tests
def _binary_literal_mask_and_pat(sv, casex: bool) -> Optional[Tuple[int, int, int]]:
    """Deprecated alias; use :func:`_wildcard_literal_mask_and_pat`."""
    return _wildcard_literal_mask_and_pat(sv, casex)


def _bitvec_align_same_width(a: BitVecRef, b: BitVecRef) -> tuple:
    """Widen both bitvectors to ``max(size(a), size(b))`` with zero extension."""
    if a.size() == b.size():
        return a, b
    tgt = max(a.size(), b.size())
    if a.size() < tgt:
        a = z3.ZeroExt(tgt - a.size(), a)
    else:
        a = z3.Extract(tgt - 1, 0, a)
    if b.size() < tgt:
        b = z3.ZeroExt(tgt - b.size(), b)
    else:
        b = z3.Extract(tgt - 1, 0, b)
    return a, b


def _bitvec_resize_to(selector: BitVecRef, pw: int) -> BitVecRef:
    """Truncate or zero-extend ``selector`` to ``pw`` bits (Verilog-like low bits)."""
    sw = selector.size()
    if sw == pw:
        return selector
    if sw > pw:
        return z3.Extract(pw - 1, 0, selector)
    return z3.ZeroExt(pw - sw, selector)


def case_statement_arm_matches_z3(
    selector: BitVecRef,
    pattern_expr,
    store: dict,
    module: str,
    case_kind: str,
):
    """Z3 predicate: ``case`` / ``casez`` / ``casex`` arm matches (selector vs pattern).

    For ``casez`` and ``casex`` with parseable literals (sized **binary**, **hex**,
    **oct**), uses the same masked equality; ``casex`` only changes which
    characters count as don’t-cares (see ``_wildcard_literal_mask_and_pat``).
    Unsized or
    odd shapes, **decimal** patterns, or width mismatch vs ``effectiveWidth``
    fall back to full-vector ``==``. Plain ``case`` uses ``==`` after width align.

    Returns a Z3 ``BoolRef``, or ``None`` if the match cannot be expressed.
    """
    item_z3 = semantic_expr_to_z3(pattern_expr, store, module)
    if item_z3 is None:
        return None
    if not isinstance(item_z3, BitVecRef) or not isinstance(selector, BitVecRef):
        return None

    if case_kind not in ("casez", "casex"):
        a, b = _bitvec_align_same_width(selector, item_z3)
        return a == b

    if getattr(pattern_expr, "kind", None) != ps_ast.ExpressionKind.IntegerLiteral:
        a, b = _bitvec_align_same_width(selector, item_z3)
        return a == b

    parsed = _wildcard_literal_mask_and_pat(
        pattern_expr.value, casex=(case_kind == "casex")
    )
    if parsed is None:
        a, b = _bitvec_align_same_width(selector, item_z3)
        return a == b

    care, pat_bits, pw = parsed
    ew = getattr(pattern_expr, "effectiveWidth", None)
    if ew is not None:
        try:
            if int(ew) != pw:
                a, b = _bitvec_align_same_width(selector, item_z3)
                return a == b
        except (TypeError, ValueError):
            pass

    sel_n = _bitvec_resize_to(selector, pw)
    care_bv = BitVecVal(care, pw)
    pat_bv = BitVecVal(pat_bits, pw)
    return (sel_n & care_bv) == (pat_bv & care_bv)


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

