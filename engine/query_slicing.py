"""Query slicing using Quick-Union with weighted path compression (Paper §4.2.2).

At each branch point during symbolic execution, only the constraints whose
symbolic variables are in the same connected component as the branch condition
need to be considered.  This module provides:

  QuickUnion   – weighted quick-union with path compression
  slice_query  – given the full set of tracked constraints and the branch
                 condition, return the minimal subset of constraints needed.
"""

from __future__ import annotations
from typing import Dict, Set, List, Any, Optional

try:
    from z3 import ExprRef
    from z3 import z3util
except ImportError:
    ExprRef = None
    z3util = None


# ---------------------------------------------------------------------------
# Quick-Union with weighted path compression
# ---------------------------------------------------------------------------

class QuickUnion:
    """Quick-Union (weighted, with path compression) over arbitrary hashable keys.

    Paper §4.2.2: "We use the Quick-Union algorithm with weighted path
    compression (QU), which is a graph connectivity algorithm."
    """

    def __init__(self) -> None:
        self.parent: Dict[Any, Any] = {}
        self.size: Dict[Any, int] = {}

    # -- core operations ---------------------------------------------------

    def _ensure(self, x: Any) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: Any) -> Any:
        """Find root of *x* with path compression."""
        self._ensure(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: Any, y: Any) -> None:
        """Merge the sets containing *x* and *y* (weighted)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # attach smaller tree under root of larger tree
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]

    def connected(self, x: Any, y: Any) -> bool:
        return self.find(x) == self.find(y)

    def component(self, x: Any) -> Set[Any]:
        """Return all elements in the same component as *x*."""
        root = self.find(x)
        return {k for k in self.parent if self.find(k) == root}

    # -- helpers for symbolic variable tracking ----------------------------

    def union_vars(self, var_names: List[str]) -> None:
        """Union all variables in *var_names* together (they appear in the
        same constraint/branch condition)."""
        if len(var_names) < 2:
            for v in var_names:
                self._ensure(v)
            return
        first = var_names[0]
        for v in var_names[1:]:
            self.union(first, v)

    def register_constraint(self, constraint) -> None:
        """Extract symbolic variable names from a Z3 constraint and union
        them together."""
        if z3util is None:
            return
        try:
            var_names = [str(v) for v in z3util.get_vars(constraint)]
        except Exception:
            return
        self.union_vars(var_names)


# ---------------------------------------------------------------------------
# Query slicing
# ---------------------------------------------------------------------------

def get_vars_from_expr(expr) -> Set[str]:
    """Return the set of symbolic variable name strings in a Z3 expression."""
    if z3util is None:
        return set()
    try:
        return {str(v) for v in z3util.get_vars(expr)}
    except Exception:
        return set()


def slice_query(
    qu: QuickUnion,
    all_constraints: list,
    branch_vars: Set[str],
) -> list:
    """Return the subset of *all_constraints* whose variables overlap with the
    connected component(s) of *branch_vars* in the Quick-Union *qu*.

    Paper §4.2.2: "We use query slicing to find the parts of the query needed
    to evaluate the feasibility of the current branch condition."

    Parameters
    ----------
    qu : QuickUnion
        The union-find structure tracking symbolic-variable connectivity built
        up during execution so far.
    all_constraints : list[ExprRef]
        The full list of tracked path-condition constraints.
    branch_vars : set[str]
        Variable names appearing in the current branch condition.

    Returns
    -------
    list[ExprRef]
        The minimal subset of constraints needed for the SAT check.
    """
    if not branch_vars or not all_constraints:
        return list(all_constraints)

    # Determine the root(s) of the branch variables' component(s)
    branch_roots = set()
    for v in branch_vars:
        branch_roots.add(qu.find(v))

    sliced: list = []
    for c in all_constraints:
        c_vars = get_vars_from_expr(c)
        if not c_vars:
            # Constant constraint – always include
            sliced.append(c)
            continue
        # Include constraint if any of its variables shares a component root
        # with the branch variables
        for cv in c_vars:
            if qu.find(cv) in branch_roots:
                sliced.append(c)
                break

    return sliced
