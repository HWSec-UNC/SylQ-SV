"""Query normalization for SMT cache keys (Paper §4.2.3).

Three phases are applied to each Z3 constraint before cache lookup:

  1. Propositional term normalization:
     - Concatenation normal form  (standardize bitvector ops)
     - Arithmetic normal form     (simplify arithmetic)
     Both are polynomial-time transformations.

  2. Lexicographic ordering:
     - Sort terms in conjunctions/disjunctions by canonical string ordering.

  3. Variable renaming:
     - Rename symbolic variables left-to-right as T1, T2, ...
     - Allows cache hits across different runs with different variable names.

Usage:
    key = normalize_query(z3_expr)      # single constraint
    key = normalize_query_list(z3_list) # list of constraints
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set

try:
    from z3 import (
        ExprRef, BoolRef, BitVecRef, ArithRef,
        simplify, substitute, is_bool, is_const,
        BitVec, BitVecSort, BoolVal, BitVecVal,
        And, Or, Not, is_and, is_or, is_not,
        is_true, is_false,
    )
    from z3 import z3util
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Phase 1: Propositional term normalization (concatenation + arithmetic NF)
# ---------------------------------------------------------------------------

def _simplify_expr(expr: ExprRef) -> ExprRef:
    """Apply Z3's built-in simplifier which handles:
    - Constant folding  (e.g., M & 0x0000 -> 0x0000)
    - Bitwise identity  (e.g., M | 0 -> M, M & ~0 -> M)
    - Arithmetic simplification (e.g., M + 0 -> M)
    - Boolean simplification (e.g., True & X -> X)

    This provides a polynomial-time approximation of concatenation normal
    form and arithmetic normal form as described in the paper.
    """
    if not Z3_AVAILABLE:
        return expr
    try:
        # Z3's simplify with specific options for better normalization
        return simplify(expr,
                        som=True,           # sum-of-monomials for arithmetic
                        sort_sums=True,     # canonical ordering in sums
                        pull_cheap_ite=True,
                        flat=True,          # flatten nested And/Or
                        elim_and=False,     # keep And nodes (not rewrite to Or+Not)
                        )
    except Exception:
        return expr


# ---------------------------------------------------------------------------
# Phase 2: Lexicographic ordering
# ---------------------------------------------------------------------------

def _sort_children(expr: ExprRef) -> ExprRef:
    """If *expr* is a conjunction (And) or disjunction (Or), sort its
    children by their string representation to produce a canonical order.

    Paper §4.2.3: "terms in a constraint are put in lexicographic order."
    """
    if not Z3_AVAILABLE:
        return expr
    try:
        if is_and(expr):
            children = sorted(expr.children(), key=lambda c: str(c))
            return And(*children) if len(children) > 1 else children[0]
        if is_or(expr):
            children = sorted(expr.children(), key=lambda c: str(c))
            return Or(*children) if len(children) > 1 else children[0]
    except Exception:
        pass
    return expr


def _lexicographic_normalize(expr: ExprRef) -> ExprRef:
    """Recursively apply lexicographic ordering to all And/Or sub-expressions."""
    if not Z3_AVAILABLE:
        return expr
    try:
        # Process children first (bottom-up)
        n = expr.num_args()
        if n == 0:
            return expr
        new_children = [_lexicographic_normalize(expr.arg(i)) for i in range(n)]
        # Rebuild with normalized children
        new_expr = expr.decl()(*new_children) if n > 0 else expr
        return _sort_children(new_expr)
    except Exception:
        return expr


# ---------------------------------------------------------------------------
# Phase 3: Variable renaming
# ---------------------------------------------------------------------------

def _collect_vars_ordered(expr: ExprRef) -> List[str]:
    """Collect symbolic variable names from *expr* in left-to-right (DFS) order,
    preserving first-occurrence order."""
    if not Z3_AVAILABLE:
        return []
    seen: Set[str] = set()
    ordered: List[str] = []

    def _walk(e):
        try:
            if is_const(e) and e.decl().arity() == 0:
                name = str(e)
                # Skip numeric constants and True/False
                if name not in seen and not _is_literal(name):
                    seen.add(name)
                    ordered.append(name)
            else:
                for i in range(e.num_args()):
                    _walk(e.arg(i))
        except Exception:
            pass

    _walk(expr)
    return ordered


def _is_literal(name: str) -> bool:
    """Check if a name looks like a Z3 numeric/boolean literal."""
    if name in ('True', 'False'):
        return True
    try:
        int(name)
        return True
    except ValueError:
        pass
    # Hex-like
    if name.startswith('#') or name.startswith('0x'):
        return True
    return False


def _rename_variables(expr: ExprRef, rename_map: Optional[Dict[str, ExprRef]] = None) -> tuple:
    """Rename all symbolic variables in *expr* to T1, T2, ... in order of
    first occurrence (left-to-right DFS).

    Paper §4.2.3: "symbolic values appearing in the constraints are renamed.
    SylQ-SV uses fresh symbolic values each time it runs, and variable
    renaming allows equivalent queries across runs."

    Returns (renamed_expr, rename_map_used).
    """
    if not Z3_AVAILABLE:
        return expr, {}

    var_names = _collect_vars_ordered(expr)
    if not var_names:
        return expr, rename_map or {}

    if rename_map is None:
        rename_map = {}

    # Build substitution list
    subs_from = []
    subs_to = []
    counter = len(rename_map)

    for vname in var_names:
        if vname in rename_map:
            continue
        counter += 1
        # We need the original Z3 variable to substitute
        try:
            orig_vars = z3util.get_vars(expr)
            for v in orig_vars:
                if str(v) == vname:
                    new_name = f"T{counter}"
                    if isinstance(v, BitVecRef):
                        new_var = BitVec(new_name, v.sort().size())
                    else:
                        # Boolean variable
                        from z3 import Bool
                        new_var = Bool(new_name)
                    rename_map[vname] = new_var
                    subs_from.append(v)
                    subs_to.append(new_var)
                    break
        except Exception:
            pass

    if subs_from:
        try:
            expr = substitute(expr, list(zip(subs_from, subs_to)))
        except Exception:
            pass

    return expr, rename_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_constraint(expr: ExprRef, rename_map: Optional[Dict] = None) -> tuple:
    """Apply all three normalization phases to a single Z3 constraint.

    Returns (normalized_expr, rename_map) where rename_map is updated
    with any new variable->T_i mappings.
    """
    if not Z3_AVAILABLE or expr is None:
        return expr, rename_map or {}

    # Phase 1: Simplify (concatenation NF + arithmetic NF)
    expr = _simplify_expr(expr)

    # Phase 2: Lexicographic ordering
    expr = _lexicographic_normalize(expr)

    # Phase 3: Variable renaming
    expr, rename_map = _rename_variables(expr, rename_map)

    return expr, rename_map


def normalize_query(expr: ExprRef) -> str:
    """Normalize a single Z3 constraint and return its string cache key."""
    if not Z3_AVAILABLE or expr is None:
        return str(expr)
    normalized, _ = normalize_constraint(expr)
    return str(normalized)


def normalize_query_list(constraints: list) -> str:
    """Normalize a list of Z3 constraints into a single canonical cache key.

    All constraints share the same rename map so that variable names are
    consistent across the conjunction.
    """
    if not Z3_AVAILABLE or not constraints:
        return str(constraints)

    rename_map: Dict = {}
    normalized_parts = []
    for c in constraints:
        nc, rename_map = normalize_constraint(c, rename_map)
        normalized_parts.append(str(nc))

    # Sort the normalized constraint strings for canonical ordering
    normalized_parts.sort()
    return " AND ".join(normalized_parts)
