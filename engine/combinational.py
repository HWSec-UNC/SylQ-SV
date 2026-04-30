"""Continuous-assign dependency tracking and re-evaluation (SylQ paper §4.4).

For each ContinuousAssign in a module we record the LHS signal and the set of
RHS signals it reads. A dirty bitset (one bit per assign) tells us which
continuous assigns must be re-evaluated: when any signal is written we OR in
the bits of the assigns that depend on it. At end of clock edge we iterate
the dirty set to a fixed point, writing each assign's LHS and dirtying any
further comb assigns that depend on it.
"""
import pyslang.ast as ps_ast
from helpers.rvalue_to_z3 import semantic_expr_to_z3
import z3
# TODO: Param look at this file

def _collect_rhs_signals(expr, out):
    if expr is None:
        return
    kind = getattr(expr, "kind", None)
    if kind == ps_ast.ExpressionKind.NamedValue:
        sym = getattr(expr, "symbol", None)
        if sym is not None and getattr(sym, "name", None):
            out.add(sym.name)
        return
    for attr in ("left", "right", "operand", "predicate", "value", "selector",
                 "min", "typ", "max", "expression", "operand1", "operand2"):
        sub = getattr(expr, attr, None)
        if sub is not None and sub is not expr:
            _collect_rhs_signals(sub, out)
    for attr in ("operands", "arguments", "elements"):
        seq = getattr(expr, attr, None)
        if seq is None:
            continue
        try:
            for item in seq:
                inner = getattr(item, "value", item)
                _collect_rhs_signals(inner, out)
        except TypeError:
            pass


def _lhs_name(expr):
    if expr is None:
        return None
    sym = getattr(expr, "symbol", None)
    if sym is not None and getattr(sym, "name", None):
        return sym.name
    inner = getattr(expr, "value", None)
    if inner is not None and inner is not expr:
        return _lhs_name(inner)
    return None


def build_comb_metadata(comb_assigns):
    """Return (assigns, lhs_by_idx, deps_by_signal).

    assigns         : list[ContinuousAssign]; list index == bit position in dirty bitset
    lhs_by_idx      : {assign_idx -> lhs signal name}
    deps_by_signal  : {rhs_signal_name -> list[assign_idx]} — reverse index used on every write
    """
    assigns = list(comb_assigns)
    lhs_by_idx = {}
    deps_by_signal = {}
    for idx, ca in enumerate(assigns):
        assignment = getattr(ca, "assignment", None)
        if assignment is None:
            continue
        lhs = _lhs_name(getattr(assignment, "left", None))
        if lhs is None:
            continue
        lhs_by_idx[idx] = lhs
        rhs_names = set()
        _collect_rhs_signals(getattr(assignment, "right", None), rhs_names)
        for name in rhs_names:
            deps_by_signal.setdefault(name, []).append(idx)
    return assigns, lhs_by_idx, deps_by_signal


def evaluate_dirty_comb(state, module_name, manager):
    """
    Fixed-point re-evaluation of dirty continuous assigns.

    Iteration bound protects against combinational feedback loops in the RTL.
    """
    assigns = getattr(manager, "comb_assigns", {}).get(module_name)
    if not assigns:
        return
    lhs_by_idx = manager.comb_lhs.get(module_name, {})
    deps_by_signal = manager.comb_deps.get(module_name, {})

    if not hasattr(state, "dirty") or state.dirty is None:
        return
    dirty = state.dirty.get(module_name, 0)
    if dirty == 0:
        return

    store = state.store.setdefault(module_name, {})
    while dirty:
        # Fun bitwise tricks.
        # First isolate the lowest set bit, then subtract one to get the 
        # desired index. Lastly, clear this bit.
        bit = dirty & -dirty
        idx = bit.bit_length() - 1
        dirty &= dirty - 1

        lhs = lhs_by_idx.get(idx)
        if lhs is None:
            continue
        ca = assigns[idx]
        assignment = getattr(ca, "assignment", None)
        if assignment is None:
            continue
        rhs = getattr(assignment, "right", None)
        if rhs is None:
            continue
        try:
            new_val = semantic_expr_to_z3(rhs, store, module_name)
        except Exception:
            new_val = None
        if new_val is None:
            continue

        old = store.get(lhs)
        if isinstance(old, z3.ExprRef) and isinstance(new_val, z3.ExprRef):
            if old.eq(new_val):
                continue 

        store[lhs] = new_val
        for dep_idx in deps_by_signal.get(lhs, ()):
            dirty |= 1 << dep_idx

    state.dirty[module_name] = dirty
