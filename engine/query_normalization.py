"""Query normalization for SMT cache keys (Paper §4.2.3).

Three phases are applied to each Z3 constraint before cache lookup:

  1. Variable renaming:
     - Rename symbolic variables as T1, T2, ... in order of first occurrence
       (constraint list order, then left-to-right DFS within a constraint).
     - Allows cache hits across different runs with different variable names.
     - Runs first: symbol names are random per run, so nothing that orders terms
       may see them.

  2. Propositional term normalization:
     - Concatenation normal form  (standardize bitvector ops)
     - Arithmetic normal form     (simplify arithmetic)
     Both are polynomial-time transformations.

  3. Lexicographic ordering:
     - Sort terms in conjunctions/disjunctions by canonical string ordering.

Usage:
    key = normalize_query(z3_expr)      # single constraint
    key = normalize_query_list(z3_list) # list of constraints

    # Incremental, for DFS over growing path conditions:
    q = NormalizedQuery().extend(pc_a)
    q2 = q.extend(pc_b)                 # only pc_b is normalized
    key = q2.key()
"""

from __future__ import annotations

import hashlib

try:
    from z3 import (
        And,
        Const,
        ExprRef,
        Or,
        Z3_OP_UNINTERPRETED,
        is_and,
        is_app,
        is_bool,
        is_bv,
        is_or,
        simplify,
        substitute,
    )

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Phase 1: Variable renaming
# ---------------------------------------------------------------------------


def _collect_vars_ordered(expr: ExprRef) -> list[ExprRef]:
    """Collect symbolic variables from *expr* in left-to-right (DFS) order,
    preserving first-occurrence order. Shared sub-terms are visited once."""
    seen: set[int] = set()
    ordered: list[ExprRef] = []
    stack = [expr]
    while stack:
        e = stack.pop()
        eid = e.get_id()
        if eid in seen:
            continue
        seen.add(eid)
        if is_app(e) and e.num_args() == 0:
            # Numerals and True/False have their own decl kinds
            if e.decl().kind() == Z3_OP_UNINTERPRETED:
                ordered.append(e)
        else:
            stack.extend(reversed(e.children()))
    return ordered


def _canonical_var(index: int, var: ExprRef) -> tuple[str, ExprRef]:
    """Return (name, variable) for the *index*-th symbol, with the sort of *var*.

    The sort is part of the name: two queries that differ only in a variable's
    width are different queries and must not share a key.
    """
    if is_bool(var):
        tag = "b"
    elif is_bv(var):
        tag = f"bv{var.size()}"
    else:
        tag = "".join(ch if ch.isalnum() else "_" for ch in str(var.sort()))
    name = f"T{index}_{tag}"
    return name, Const(name, var.sort())


# ---------------------------------------------------------------------------
# Phase 2: Propositional term normalization (concatenation + arithmetic NF)
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
    try:
        # Z3's simplify with specific options for better normalization
        return simplify(
            expr,
            som=True,  # sum-of-monomials for arithmetic
            sort_sums=True,  # canonical ordering in sums
            pull_cheap_ite=True,
            flat=True,  # flatten nested And/Or
            elim_and=False,  # keep And nodes (not rewrite to Or+Not)
        )
    except Exception:
        return expr


# ---------------------------------------------------------------------------
# Phase 3: Lexicographic ordering
# ---------------------------------------------------------------------------


def _sort_key(expr: ExprRef) -> str:
    # sexpr() is native and complete; str() goes through the Python
    # pretty-printer, which is slow and elides large terms with "...".
    return expr.sexpr()


def _lexicographic_normalize(expr: ExprRef, memo: dict | None = None) -> ExprRef:
    """Recursively sort the children of all And/Or sub-expressions.

    Paper §4.2.3: "terms in a constraint are put in lexicographic order."
    """
    if memo is None:
        memo = {}
    eid = expr.get_id()
    done = memo.get(eid)
    if done is not None:
        return done
    out = expr
    if is_app(expr) and expr.num_args() > 0:
        # Process children first (bottom-up)
        children = [_lexicographic_normalize(c, memo) for c in expr.children()]
        if is_and(expr) or is_or(expr):
            children.sort(key=_sort_key)
            if len(children) == 1:
                out = children[0]
            else:
                out = And(*children) if is_and(expr) else Or(*children)
        else:
            try:
                out = expr.decl()(*children)
            except Exception:
                out = expr
    memo[eid] = out
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class QueryMemo:
    """Normalization results shared by every NormalizedQuery built from it.

    Holds references to the constraints it has seen (AST ids are only stable
    while the AST is alive), so its lifetime should match that of the search
    that produces those constraints.
    """

    __slots__ = ("norm", "vars")

    def __init__(self):
        # constraint AST id -> (constraint, its variables in first-occurrence order)
        self.vars: dict[int, tuple[ExprRef, list[ExprRef]]] = {}
        # (constraint AST id, canonical names of its variables) -> normalized text
        self.norm: dict[tuple[int, tuple[str, ...]], str] = {}


class NormalizedQuery:
    """An immutable, normalized conjunction that can be extended cheaply.

    ``extend`` normalizes only the constraints it is given: the canonical names
    of the existing constraints do not change when more are appended, so a DFS
    can carry one of these per stack frame instead of re-normalizing the whole
    path condition at every level.
    """

    __slots__ = ("_memo", "_parts", "_rename")

    def __init__(self, memo: QueryMemo | None = None):
        self._memo = memo if memo is not None else QueryMemo()
        # original variable AST id -> (canonical name, canonical variable)
        self._rename: dict[int, tuple[str, ExprRef]] = {}
        self._parts: tuple[str, ...] = ()

    def extend(self, constraints: list) -> NormalizedQuery:
        """Return a new query with *constraints* appended."""
        if not constraints:
            return self
        memo = self._memo
        rename = self._rename
        owned = False
        parts = list(self._parts)
        for c in constraints:
            cid = c.get_id()
            entry = memo.vars.get(cid)
            if entry is None:
                entry = (c, _collect_vars_ordered(c))
                memo.vars[cid] = entry
            variables = entry[1]
            for v in variables:
                vid = v.get_id()
                if vid not in rename:
                    if not owned:
                        rename = dict(rename)
                        owned = True
                    rename[vid] = _canonical_var(len(rename) + 1, v)
            targets = [rename[v.get_id()] for v in variables]
            norm_key = (cid, tuple(name for name, _ in targets))
            part = memo.norm.get(norm_key)
            if part is None:
                expr = c
                if variables:
                    expr = substitute(
                        expr, [(v, t) for v, (_, t) in zip(variables, targets)]
                    )
                expr = _lexicographic_normalize(_simplify_expr(expr))
                part = expr.sexpr()
                memo.norm[norm_key] = part
            parts.append(part)
        out = NormalizedQuery(memo)
        out._rename = rename
        out._parts = tuple(parts)
        return out

    def text(self) -> str:
        """Canonical text of the conjunction (order of constraints does not matter)."""
        return " AND ".join(sorted(self._parts))

    def key(self) -> str:
        """Fixed-size cache key for the conjunction."""
        return hashlib.sha256(self.text().encode()).hexdigest()


def normalize_query(expr: ExprRef) -> str:
    """Normalize a single Z3 constraint and return its string cache key."""
    if not Z3_AVAILABLE or expr is None:
        return str(expr)
    return NormalizedQuery().extend([expr]).text()


def normalize_query_list(constraints: list) -> str:
    """Normalize a list of Z3 constraints into a single canonical cache key.

    All constraints share the same rename map so that variable names are
    consistent across the conjunction.
    """
    if not Z3_AVAILABLE or not constraints:
        return str(constraints)
    return NormalizedQuery().extend(constraints).text()
