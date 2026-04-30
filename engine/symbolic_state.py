"""The Symbolic State is comprised of the path condition and the symbolic store. There 
are some other methods here that may be helpful, too."""

import z3
from z3 import Solver, Int, BitVec, BitVecSort
import random
import string

class SymbolicState:
    pc = Solver()
    assertion_counter = 0
    sort = BitVecSort(32)
    clock_cycle: int = 0
    #TODO need to change to be a nested mapping of module names to dictionaries
    # can be initalized at the beginning of the run 
    store = {}

    # set to true when evaluating a conditoin so that
    # evaluating the expression knows to add the expr to the
    # PC, set to false after
    cond: bool = False

    def get_symbolic_expr(self, module_name: str, var_name: str) -> str:
        """Just looks up a symbolic expression associated with a specific variable name
        in that particular module."""
        if '[' in var_name:
            name = var_name.split("[")[0]
            return self.store[module_name][name]
        elif '.' in var_name:
            real_module_name = var_name.split(".")[0]
            real_var_name = var_name.split(".")[1]
            return self.store[real_module_name][real_var_name]
        return self.store[module_name][var_name]

    def get_symbols(self):
        """Returns a list of all the symbols present in the symbolic state.
        This is useful in the parsing to z3 phase because we need to know what symbols to declare as constants."""
        symbols_list = []
        for module in self.store:
            for signal in self.store[module]:
                symbolic_expression = self.store[module][signal]
                symbols_list += symbolic_expression.split(" ")
        res = []
        for sym in symbols_list:
            if sym.isalnum():
                res.append(sym)
        return res

    def snapshot(self, module_name: str):
        """Return a lightweight copy of the store for one module."""
        return dict(self.store.get(module_name, {}))

    def fresh_for_block(self, module_name: str, base_store: dict):
        """Create a fresh SymbolicState pre-loaded with base_store for one module."""
        s = SymbolicState()
        s.store = {module_name: dict(base_store)}
        s.pc = Solver()

        s.pending_nba = {}
        # Dirty bitset per module: bit i set ⇒ continuous assign i must be re-evaluated.
        s.dirty = {}
        return s

    def mark_dirty(self, module_name: str, signal_name: str, manager) -> None:
        """OR the bits of every comb assign that reads *signal_name* into state.dirty."""
        if manager is None:
            return
        mod_deps = getattr(manager, "comb_deps", None)
        if not mod_deps:
            return
        deps_by_signal = mod_deps.get(module_name)
        if not deps_by_signal:
            return
        dep_idxs = deps_by_signal.get(signal_name)
        if not dep_idxs:
            return

        if not hasattr(self, "dirty") or self.dirty is None:
            self.dirty = {}
        bits = 0
        for idx in dep_idxs:
            bits |= 1 << idx
        self.dirty[module_name] = self.dirty.get(module_name, 0) | bits

    def flush_pending_nba(self, module_name: str, manager=None) -> None:
        """Apply deferred nonblocking assignments (`<=`) for one module to the store.

        Reads during the same simulated clock edge use ``store`` only; this merges
        pending RHS values after all statements for that edge have been executed.

        This is per CFG path (end of ``_execute_cfg_path``), not a design-wide
        timestep: other always blocks in the module are not executed in the same
        ``SymbolicState`` walk, so there is no single global NBA queue across blocks
        here (see ``execute_sv`` explore-phase comment in ``execution_engine.py``).

        If manager is provided, each flushed NBA target also dirties its comb
        dependents so the subsequent evaluate_dirty_comb pass re-runs them.
        """
        pend = getattr(self, "pending_nba", None)
        if not pend:
            return
        bucket = pend.get(module_name)
        if not bucket:
            return
        mod_store = self.store.setdefault(module_name, {})
        for k, v in bucket.items():
            mod_store[k] = v
            # TODO: Param Check 
            if manager is not None:
                self.mark_dirty(module_name, k, manager)
        bucket.clear()
    
    def advance_cycle(self, module_name: str, reg_decls: set) -> 'SymbolicState':
        """Create a new SymbolicState for the next clock cycle.
        Register values (signals in reg_decls) carry forward from this cycle's
        end store. Input signals get fresh symbolic variables.
        Args:
            module_name: The module whose store we are advancing.
            reg_decls: Set of signal names that are registers (hold state).

        Returns:
            A fresh SymbolicState with registers pre-loaded from this cycle's store.
        """
        next_state = SymbolicState()
        next_state.pending_nba = {}
        next_state.dirty = {}
        next_state.pc = Solver()
        next_state.store = {}

        current_store = self.store.get(module_name, {})
        next_store = {}

        for signal, val in current_store.items():
            if signal in reg_decls:
                # Register: carry forward the symbolic expression from end of this cycle
                next_store[signal] = val
            else:
                # Input wire: give a fresh symbolic variable for the new cycle
                fresh_name = ''.join(
                    random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                    for _ in range(16)
                )
                next_store[signal] = BitVec(fresh_name, 32)

        next_state.store[module_name] = next_store
        return next_state
