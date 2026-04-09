"""Stronger soundness conditions for skipping joint Z3 when combining PCs.

Disjoint-skip is valid only when path conditions are independent. Raw string
intersection of ``z3util.get_vars`` names can miss aliases (``|`` vs ``.``) or
claim independence when distinct RTL modules may still share hidden nets.

This module tightens the predicate:

- **Canonical names** — normalize separators / case so the same symbol is not
  treated as two different strings.
- **AST structure** — if two PCs share Select/Store roots or the same
  uninterpreted declaration, they are not independent (must run Z3).
- **Cross-module structure** — without an elaboration graph, assume any two
  distinct RTL modules may interact on nets not reflected in extracted names;
  same-module multi-cycle prefixes are never structurally independent.

Environment:

- ``SYLQ_FEASIBILITY_LEGACY_DISJOINT=1`` — restore legacy behavior (raw name
  intersection only).
- ``SYLQ_RELAX_CROSS_STRUCTURAL=1`` — do not apply conservative cross-module
  structural rule (still uses canonical + AST checks).
"""

from __future__ import annotations

import os
from typing import AbstractSet, Dict, List, Optional, Set

from z3 import ExprRef, Z3_OP_SELECT, Z3_OP_STORE, Z3_OP_UNINTERPRETED, is_app
from z3 import z3util

_LEGACY = os.environ.get("SYLQ_FEASIBILITY_LEGACY_DISJOINT", "").strip().lower() in (
    "1",
    "yes",
    "true",
)
_RELAX_CROSS_STRUCTURAL = os.environ.get(
    "SYLQ_RELAX_CROSS_STRUCTURAL", ""
).strip().lower() in ("1", "yes", "true")


def normalize_feasibility_var_name(raw: str) -> str:
    """Normalize a Z3 symbol string for overlap comparisons."""
    s = raw.strip().lower()
    for sep in ("|", "::"):
        s = s.replace(sep, ".")
    while ".." in s:
        s = s.replace("..", ".")
    parts = [p for p in s.split(".") if p]
    return ".".join(parts)


def canonical_var_set(names: AbstractSet[str]) -> Set[str]:
    return {normalize_feasibility_var_name(n) for n in names}


def canonical_name_sets_disjoint(v1: AbstractSet[str], v2: AbstractSet[str]) -> bool:
    return not (canonical_var_set(v1) & canonical_var_set(v2))


def _collect_ast_structure_keys(constraints: List[ExprRef]) -> Set[str]:
    """Keys for array/UF structure that can couple PCs without shared atom names."""
    out: Set[str] = set()
    seen: Set[int] = set()

    def visit(e: ExprRef) -> None:
        eid = id(e)
        if eid in seen:
            return
        seen.add(eid)
        if not is_app(e):
            return
        decl = e.decl()
        k = decl.kind()
        if k == Z3_OP_SELECT or k == Z3_OP_STORE:
            try:
                root = e.arg(0)
                out.add(f"arr:{str(root)}")
            except Exception:
                out.add(f"arr:{str(e)}")
        elif k == Z3_OP_UNINTERPRETED:
            out.add(f"uf:{str(decl)}")
        for i in range(e.num_args()):
            try:
                visit(e.arg(i))
            except Exception:
                pass

    for c in constraints:
        try:
            visit(c)
        except Exception:
            pass
    return out


def ast_structure_intersects(pc1: List[ExprRef], pc2: List[ExprRef]) -> bool:
    """True if PCs share array/UF structure that could create hidden coupling."""
    if not pc1 or not pc2:
        return False
    return bool(_collect_ast_structure_keys(pc1) & _collect_ast_structure_keys(pc2))


def vars_from_constraints(constraints: List[ExprRef]) -> Set[str]:
    """Variable names from Z3 constraints (same basis as dfs_iterator)."""
    out: Set[str] = set()
    for c in constraints:
        try:
            for v in z3util.get_vars(c):
                out.add(str(v))
        except Exception:
            pass
    return out


def cross_module_structurally_independent(
    partial_modules: AbstractSet[str],
    next_module: str,
    module_graph: Optional[Dict[str, Set[str]]],
) -> bool:
    """Whether cross-module combine may skip Z3 for *structural* RTL reasons alone.

    * ``module_graph``: optional undirected adjacency (each key maps to neighbors).
      If provided, we require SAT when ``next_module`` is adjacent to any module
      already in the partial prefix. Missing edges allow independence *only* for
      that pair (conservative: no multi-hop analysis).

    * If ``module_graph`` is ``None`` and structural relaxation is off, any merge
      that already includes at least one module and adds another *distinct*
      module is **not** structurally independent (hidden inter-module nets).

    * Same module appearing again (another cycle) is never structurally
      independent from its own prefix.
    """
    if _RELAX_CROSS_STRUCTURAL:
        return True
    if not partial_modules:
        return True
    if next_module in partial_modules:
        return False
    if module_graph is None:
        # No elaboration graph: distinct modules may share ports not visible as
        # identical Z3 names between partitions.
        return False
    for p in partial_modules:
        if p == next_module:
            continue
        nbr_p = module_graph.get(p, set())
        nbr_n = module_graph.get(next_module, set())
        if next_module in nbr_p or p in nbr_n:
            return False
    return True


def may_disjoint_skip_merge(
    partial_vars: AbstractSet[str],
    new_vars: AbstractSet[str],
    partial_pc: List[ExprRef],
    new_pc: List[ExprRef],
) -> bool:
    """Sound predicate for intra-module (block) merge: OK to skip joint Z3."""
    if _LEGACY:
        return not (set(partial_vars) & set(new_vars))
    if not canonical_name_sets_disjoint(partial_vars, new_vars):
        return False
    if ast_structure_intersects(partial_pc, new_pc):
        return False
    return True


def may_disjoint_skip_cross_module(
    partial_vars: AbstractSet[str],
    new_vars: AbstractSet[str],
    partial_pc: List[ExprRef],
    new_pc: List[ExprRef],
    partial_modules: AbstractSet[str],
    next_module: str,
    module_graph: Optional[Dict[str, Set[str]]],
) -> bool:
    """Sound predicate for cross-module partial combine: OK to skip joint Z3."""
    if _LEGACY:
        return not (set(partial_vars) & set(new_vars))
    if not cross_module_structurally_independent(
        partial_modules, next_module, module_graph
    ):
        return False
    if not canonical_name_sets_disjoint(partial_vars, new_vars):
        return False
    if ast_structure_intersects(partial_pc, new_pc):
        return False
    return True
