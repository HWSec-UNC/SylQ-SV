"""Stack-based DFS iterator for piecewise symbolic execution.

This module implements the professor's suggested depth-first traversal approach:
- Instead of materializing all path combinations, traverse the Cartesian product
  using a stack data structure
- Process one combo at a time, enabling early pruning when partial merges are UNSAT
- Cache leaf-level results that are reused across sibling iterations
- Bounded memory: O(depth) stack frames instead of O(product_size) materialized list

For blocks A, B, C (A is root, C is leaf), each with paths a1, a2, ..., b1, b2, ..., c1, c2, ...:
- Symbolically explore a1, then b1, then c1, then merge (a1,b1,c1)
- Then explore c2 and merge (a1,b1,c2), and so on until all of c is explored
- Then explore b2 and merge (a1,b2,c1), etc.
- Can discard b1 at this point since it won't be needed again for a while
- Keep all c1..cn cached since they're reused immediately with each new parent
"""

import os
import sys
import time
from typing import List, Dict, Iterator, Optional, Any, Callable, Tuple
from z3 import Solver, ExprRef
from z3 import z3util
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from functools import reduce
from operator import mul

# Disjoint = (varsA \ EXCLUDE) ∩ (varsB \ EXCLUDE) == {} (professor's definition)
# So blocks that only share rst/clk are still considered disjoint.
VARS_EXCLUDED_FROM_DISJOINT = frozenset(
    {"rst", "clk", "rst_n", "rst_ni", "clk_i", "rst_i", "reset", "clock"}
)

# Progress while merge DFS runs (set 0 to disable heartbeat)
_MERGE_HEARTBEAT_SEC = float(os.environ.get("SYLQ_MERGE_HEARTBEAT_SEC", "20"))
# Log individual SAT checks slower than this (seconds); 0 disables
_SLOW_SAT_WARN_SEC = float(os.environ.get("SYLQ_SLOW_SAT_WARN_SEC", "5"))


@dataclass
class DFSFrame:
    """A single frame in the DFS stack.
    
    Represents the state at one level of the Cartesian product traversal.
    """
    level: int                          # Which block/module level (0 = first block)
    iterator: Iterator                  # Iterator over results at this level
    current_result: Optional[dict]      # Current result being processed
    partial_pc: List[ExprRef]           # Accumulated path conditions up to this level
    partial_store: Dict[str, Any]       # Accumulated store up to this level
    partial_vars: set                   # Set of variable names in partial_pc
    is_feasible: bool = True            # Whether the partial merge is still SAT


class LRUCache:
    """Simple LRU cache for SAT results."""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: str) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
            self.cache[key] = value


def _vars_in_pcs_static(pc_list: list) -> set:
    """Extract variable names from a list of Z3 constraints (module-level helper)."""
    out = set()
    for c in pc_list:
        try:
            for v in z3util.get_vars(c):
                out.add(str(v))
        except Exception:
            pass
    return out


def partition_blocks(block_result_lists: List[List[dict]]) -> List[List[int]]:
    """Partition blocks into connected components based on shared symbolic variables.
    
    Two blocks are in the same component if any of their path results share
    a symbolic variable. Uses union-find for efficiency.
    
    Returns a list of groups, where each group is a list of block indices.
    """
    n = len(block_result_lists)
    if n == 0:
        return []
    
    # Compute the set of variables used by each block (across all its results)
    block_vars: List[set] = []
    for block_results in block_result_lists:
        vars_in_block: set = set()
        for r in block_results:
            vars_in_block |= _vars_in_pcs_static(r["pc"])
        block_vars.append(vars_in_block)
    
    # Union-find to group blocks that share variables
    parent = list(range(n))
    size = [1] * n
    
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
    
    for i in range(n):
        vi = block_vars[i] - VARS_EXCLUDED_FROM_DISJOINT
        for j in range(i + 1, n):
            vj = block_vars[j] - VARS_EXCLUDED_FROM_DISJOINT
            if vi & vj:
                union(i, j)
    
    # Group block indices by component
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    
    return list(groups.values())


class ReplayableMergeResults:
    """Lazily merges per-block outcome lists via DFSMergeIterator without materializing.

    Each call to ``iter(self)`` starts a fresh DFS merge over the same block lists.
    That is required when the same module's merged paths are iterated multiple times
    (e.g. DFSCrossModuleIterator with num_cycles > 1).
    """

    def __init__(
        self,
        block_result_lists: List[List[dict]],
        module_name: str = "",
        manager: Any = None,
    ):
        self._block_result_lists = block_result_lists
        self._module_name = module_name
        self._manager = manager

    def __iter__(self) -> Iterator[dict]:
        return iter(
            DFSMergeIterator(
                block_result_lists=self._block_result_lists,
                module_name=self._module_name,
                manager=self._manager,
                enable_early_pruning=True,
                enable_caching=True,
            )
        )


class LazyProduct:
    """Lazily produces the Cartesian product of disjoint component groups.
    
    When blocks within a module are partitioned into independent groups (no shared
    variables), we can avoid materializing the full product. Instead, we:
    1. Merge within each group (small, since group members share variables)
    2. Store per-group merged results (small lists) OR ReplayableMergeResults for lazy merge
    3. Lazily iterate over the Cartesian product of groups via nested DFS (not itertools.product),
       so replayable merge axes get a fresh iterator for each outer combination.
    
    For or1200_ctrl with 28 disjoint blocks: instead of materializing 109M+ combos,
    we store ~28 small lists and produce combos one at a time via __iter__.
    
    Iterable protocol: ``__iter__`` streams the product; ``__len__`` / ``logical_size`` are
    only precise when every component is a materialized list (no lazy merge axis).
    """
    
    def __init__(self, component_results: List):
        """
        Args:
            component_results: List of per-component sources: each is either a list of
                {pc, store} dicts or a ReplayableMergeResults.
        """
        self.component_results = component_results
        self._len: Optional[int] = None
        self._try_compute_len()

    def _try_compute_len(self) -> None:
        """Set _len to product of component lengths if every component has known len."""
        if not self.component_results:
            self._len = 1
            return
        sizes: List[int] = []
        for g in self.component_results:
            if isinstance(g, ReplayableMergeResults):
                self._len = None
                return
            try:
                sizes.append(len(g))
            except TypeError:
                self._len = None
                return
        self._len = reduce(mul, sizes, 1)
    
    @property
    def logical_size(self) -> Optional[int]:
        """Total Cartesian-product size if known; None if any axis is a lazy merge."""
        if self._len is None and self.component_results:
            self._try_compute_len()
        return self._len

    def __len__(self) -> int:
        n = self.logical_size
        if n is None:
            return sys.maxsize
        if n > sys.maxsize:
            return sys.maxsize
        return n
    
    def __iter__(self) -> Iterator[dict]:
        """Lazily produce merged results from Cartesian product of groups."""
        if not self.component_results:
            yield {"pc": [], "store": {}}
            return
        
        if len(self.component_results) == 1:
            yield from self.component_results[0]
            return

        def rec(i: int, acc_pc: list, acc_store: dict) -> Iterator[dict]:
            if i == len(self.component_results):
                yield {"pc": acc_pc, "store": dict(acc_store)}
                return
            for r in self.component_results[i]:
                yield from rec(i + 1, acc_pc + r["pc"], {**acc_store, **r["store"]})

        yield from rec(0, [], {})
    
    def __bool__(self) -> bool:
        ls = self.logical_size
        if ls is not None:
            return ls > 0
        return bool(self.component_results)
    
    @property
    def n_components(self) -> int:
        return len(self.component_results)
    
    @property
    def component_sizes(self) -> List[str]:
        """Human-readable size per component; lazy merge axes show as 'lazy'."""
        out: List[str] = []
        for g in self.component_results:
            if isinstance(g, ReplayableMergeResults):
                out.append("lazy")
            else:
                out.append(str(len(g)))
        return out
    
    def total_stored(self) -> int:
        """Sum of materialized list lengths; lazy merge groups contribute input block path counts."""
        total = 0
        for g in self.component_results:
            if isinstance(g, ReplayableMergeResults):
                for bl in g._block_result_lists:
                    total += len(bl)
            else:
                total += len(g)
        return total


class DFSMergeIterator:
    """Stack-based DFS iterator for merging block results.
    
    Instead of computing product(*block_results) and materializing all combinations,
    this iterator traverses the Cartesian product depth-first using an explicit stack.
    
    Key benefits:
    1. Memory: O(num_blocks) stack frames instead of O(product_size)
    2. Early pruning: If partial merge (a1, b1) is UNSAT, skip all c paths
    3. Caching: Leaf results are cached and reused across sibling iterations
    
    Usage:
        iterator = DFSMergeIterator(block_result_lists, sat_checker, cache)
        for merged_result in iterator:
            # merged_result is a feasible {pc: [...], store: {...}}
            process(merged_result)
    """
    
    def __init__(
        self,
        block_result_lists: List[List[dict]],
        module_name: str = "",
        manager: Any = None,
        check_sat_callback: Optional[Callable] = None,
        enable_early_pruning: bool = True,
        enable_caching: bool = True,
        solver_timeout: int = 10000,
        heartbeat_sec: Optional[float] = None,
    ):
        """Initialize the DFS merge iterator.
        
        Args:
            block_result_lists: List of per-block result lists. Each block's list
                contains dicts with 'pc' (path conditions) and 'store' (symbolic store).
            module_name: Name of the module being merged (for logging).
            manager: ExecutionManager for cache access.
            check_sat_callback: Optional callback for SAT checking. If None, uses default.
            enable_early_pruning: If True, prune when partial merge is UNSAT.
            enable_caching: If True, cache SAT results.
            solver_timeout: Timeout for Z3 solver in milliseconds.
            heartbeat_sec: Wall-clock interval for stall diagnostics (default: env
                SYLQ_MERGE_HEARTBEAT_SEC or 20s); 0 or None with env 0 disables.
        """
        self.block_result_lists = block_result_lists
        self.module_name = module_name
        self.manager = manager
        self.check_sat_callback = check_sat_callback or self._default_sat_check
        self.enable_early_pruning = enable_early_pruning
        self.enable_caching = enable_caching
        self.solver_timeout = solver_timeout
        
        self.num_levels = len(block_result_lists)
        self.stack: List[DFSFrame] = []
        
        # Local cache for partial merge results (supplements Redis cache)
        self._local_cache = LRUCache(maxsize=10000)
        
        # Statistics
        self.combos_checked = 0
        self.combos_pruned = 0
        self.cache_hits = 0
        if heartbeat_sec is None:
            self._heartbeat_sec = _MERGE_HEARTBEAT_SEC
        else:
            self._heartbeat_sec = float(heartbeat_sec)
        
    def _vars_in_pcs(self, pc_list: List[ExprRef]) -> set:
        """Extract variable names from a list of Z3 constraints."""
        out = set()
        for c in pc_list:
            try:
                for v in z3util.get_vars(c):
                    out.add(str(v))
            except Exception:
                pass
        return out
    
    def _default_sat_check(self, constraints: List[ExprRef]) -> bool:
        """Default SAT checking using Z3 solver."""
        if not constraints:
            return True
        t0 = time.monotonic()
        s = Solver()
        try:
            s.set("timeout", self.solver_timeout)
        except Exception:
            pass
        for c in constraints:
            s.add(c)
        out = str(s.check()) == "sat"
        dt = time.monotonic() - t0
        if _SLOW_SAT_WARN_SEC > 0 and dt >= _SLOW_SAT_WARN_SEC:
            print(
                f"    [merge-slow-sat] {self.module_name}: {dt:.1f}s for "
                f"{len(constraints)} constraint(s), result={'sat' if out else 'unsat'}",
                flush=True,
            )
        return out
    
    def _check_disjoint(self, vars1: set, vars2: set) -> bool:
        """Disjoint = (vars1 \\ {rst,clk}) ∩ (vars2 \\ {rst,clk}) == {}."""
        v1 = vars1 - VARS_EXCLUDED_FROM_DISJOINT
        v2 = vars2 - VARS_EXCLUDED_FROM_DISJOINT
        return not (v1 & v2)
    
    def _get_cache_key(self, constraints: List[ExprRef]) -> str:
        """Generate a cache key for a set of constraints."""
        try:
            from .query_normalization import normalize_query_list
            return "dfs_merge:" + normalize_query_list(constraints)
        except Exception:
            return "dfs_merge:" + str(sorted(str(c) for c in constraints))
    
    def _check_cached(self, cache_key: str) -> Optional[bool]:
        """Check if result is in cache. Returns True/False for SAT/UNSAT, None if not cached."""
        # Check local cache first
        local_result = self._local_cache.get(cache_key)
        if local_result is not None:
            self.cache_hits += 1
            return local_result == "sat"
        
        # Check Redis cache if available
        if self.manager and self.manager.cache:
            try:
                cached = self.manager.cache.get(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    result = cached.decode() == "sat"
                    self._local_cache.set(cache_key, cached.decode())
                    return result
            except Exception:
                pass
        return None
    
    def _store_cached(self, cache_key: str, is_sat: bool) -> None:
        """Store result in cache."""
        result_str = "sat" if is_sat else "unsat"
        self._local_cache.set(cache_key, result_str)
        
        if self.manager and self.manager.cache:
            try:
                self.manager.cache.set(cache_key, result_str)
            except Exception:
                pass
    
    def _check_partial_feasibility(
        self,
        partial_pc: List[ExprRef],
        partial_vars: set,
        new_pc: List[ExprRef],
    ) -> Tuple[bool, List[ExprRef], set]:
        """Check if merging new_pc with partial_pc is feasible.
        
        Implements Paper §4.3: slice → normalize → cache for merge queries.
        
        Returns:
            (is_feasible, combined_pc, combined_vars)
        """
        new_vars = self._vars_in_pcs(new_pc)
        combined_pc = partial_pc + new_pc
        combined_vars = partial_vars | new_vars
        
        # Disjoint skip: if no variable overlap, no need for SAT check
        if self._check_disjoint(partial_vars, new_vars):
            return (True, combined_pc, combined_vars)
        
        # Need SAT check
        if not combined_pc:
            return (True, combined_pc, combined_vars)
        
        # Paper §4.3: slice → normalize → cache
        # Overlapping vars (excluding rst/clk) are the "branch" for merge queries
        overlapping_vars = (partial_vars - VARS_EXCLUDED_FROM_DISJOINT) & (
            new_vars - VARS_EXCLUDED_FROM_DISJOINT
        )
        
        # Slice the query to only constraints relevant to overlapping variables
        sliced_pc = combined_pc
        if self.manager and hasattr(self.manager, 'qu_merge') and self.manager.qu_merge is not None:
            try:
                from .query_slicing import slice_query
                # Register constraints first
                for c in combined_pc:
                    self.manager.qu_merge.register_constraint(c)
                # Then slice
                if overlapping_vars:
                    sliced_pc = slice_query(self.manager.qu_merge, combined_pc, overlapping_vars)
            except Exception:
                sliced_pc = combined_pc
        
        # Check cache with sliced (and normalized) query
        if self.enable_caching:
            cache_key = self._get_cache_key(sliced_pc)
            cached_result = self._check_cached(cache_key)
            if cached_result is not None:
                return (cached_result, combined_pc, combined_vars)
        
        # Perform SAT check on sliced constraints (more efficient)
        is_sat = self.check_sat_callback(sliced_pc)
        
        # Store in cache
        if self.enable_caching:
            self._store_cached(cache_key, is_sat)
        
        return (is_sat, combined_pc, combined_vars)
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate over all feasible merged results using DFS."""
        if not self.block_result_lists or self.num_levels == 0:
            yield {"pc": [], "store": {}}
            return
        
        # Handle single-block case
        if self.num_levels == 1:
            for result in self.block_result_lists[0]:
                yield result
            return
        
        # Initialize stack with first level
        self._push_level(0, [], {}, set())
        
        yielded_count = 0
        merge_t0 = time.monotonic()
        last_hb = merge_t0
        
        while self.stack:
            frame = self.stack[-1]
            
            # Stall diagnostics: progress can be invisible if yields are rare but SAT is busy
            if self._heartbeat_sec > 0:
                now = time.monotonic()
                if now - last_hb >= self._heartbeat_sec:
                    depth = len(self.stack)
                    top_lv = frame.level
                    print(
                        f"    [merge-heartbeat] {self.module_name}: "
                        f"{now - merge_t0:.1f}s elapsed, stack_depth={depth}, "
                        f"top_level={top_lv}, yielded={yielded_count}, "
                        f"combos_checked={self.combos_checked}, pruned={self.combos_pruned}, "
                        f"cache_hits={self.cache_hits}",
                        flush=True,
                    )
                    last_hb = now
            
            # Try to get next result at current level
            try:
                result = next(frame.iterator)
                frame.current_result = result
            except StopIteration:
                # No more results at this level, backtrack
                self.stack.pop()
                continue
            
            self.combos_checked += 1
            
            # Check feasibility of partial merge
            if self.enable_early_pruning and frame.level > 0:
                is_feasible, new_pc, new_vars = self._check_partial_feasibility(
                    frame.partial_pc,
                    frame.partial_vars,
                    result["pc"],
                )
                if not is_feasible:
                    self.combos_pruned += 1
                    continue
            else:
                new_pc = frame.partial_pc + result["pc"]
                new_vars = frame.partial_vars | self._vars_in_pcs(result["pc"])
            
            new_store = {**frame.partial_store, **result["store"]}
            
            # If at last level, yield the complete merged result
            if frame.level == self.num_levels - 1:
                yielded_count += 1
                yield {"pc": new_pc, "store": new_store}
            else:
                # Push next level onto stack
                self._push_level(frame.level + 1, new_pc, new_store, new_vars)
    
    def _push_level(
        self,
        level: int,
        partial_pc: List[ExprRef],
        partial_store: Dict[str, Any],
        partial_vars: set,
    ) -> None:
        """Push a new level onto the DFS stack."""
        frame = DFSFrame(
            level=level,
            iterator=iter(self.block_result_lists[level]),
            current_result=None,
            partial_pc=partial_pc,
            partial_store=partial_store,
            partial_vars=partial_vars,
        )
        self.stack.append(frame)
    
    def get_stats(self) -> Dict[str, int]:
        """Return iteration statistics."""
        return {
            "combos_checked": self.combos_checked,
            "combos_pruned": self.combos_pruned,
            "cache_hits": self.cache_hits,
        }


class DFSCrossModuleIterator:
    """Stack-based DFS iterator for cross-module path combination.
    
    Similar to DFSMergeIterator but operates at the cross-module level,
    combining per-module merged results across multiple cycles.
    """
    
    def __init__(
        self,
        per_module_results: Dict[str, List[dict]],
        num_cycles: int,
        manager: Any = None,
        enable_early_pruning: bool = True,
        enable_caching: bool = True,
        solver_timeout: int = 10000,
    ):
        """Initialize the cross-module DFS iterator.
        
        Args:
            per_module_results: Dict mapping module_name -> list of merged results.
            num_cycles: Number of clock cycles to simulate.
            manager: ExecutionManager for cache access.
            enable_early_pruning: If True, prune when partial combination is UNSAT.
            enable_caching: If True, cache SAT results.
            solver_timeout: Timeout for Z3 solver in milliseconds.
        """
        self.module_names = list(per_module_results.keys())
        self.per_module_results = per_module_results
        self.num_cycles = num_cycles
        self.manager = manager
        self.enable_early_pruning = enable_early_pruning
        self.enable_caching = enable_caching
        self.solver_timeout = solver_timeout
        
        # Build the list of (module, cycle) combinations to traverse
        # Each level in the DFS is one (module, cycle) pair
        self.levels: List[Tuple[str, int]] = []
        for module_name in self.module_names:
            for cycle in range(num_cycles):
                self.levels.append((module_name, cycle))
        
        self.num_levels = len(self.levels)
        self.stack: List[DFSFrame] = []
        
        self._local_cache = LRUCache(maxsize=10000)
        
        # Statistics
        self.combos_checked = 0
        self.combos_pruned = 0
        self.cache_hits = 0
    
    def _vars_in_pcs(self, pc_list: List[ExprRef]) -> set:
        """Extract variable names from a list of Z3 constraints."""
        out = set()
        for c in pc_list:
            try:
                for v in z3util.get_vars(c):
                    out.add(str(v))
            except Exception:
                pass
        return out
    
    def _sat_check(self, constraints: List[ExprRef]) -> bool:
        """Check SAT using Z3 solver."""
        if not constraints:
            return True
        s = Solver()
        try:
            s.set("timeout", self.solver_timeout)
        except Exception:
            pass
        for c in constraints:
            s.add(c)
        return str(s.check()) == "sat"
    
    def _get_cache_key(self, constraints: List[ExprRef]) -> str:
        """Generate a cache key for a set of constraints."""
        try:
            from .query_normalization import normalize_query_list
            return "dfs_xmod:" + normalize_query_list(constraints)
        except Exception:
            return "dfs_xmod:" + str(sorted(str(c) for c in constraints))
    
    def _check_cached(self, cache_key: str) -> Optional[bool]:
        """Check if result is in cache."""
        local_result = self._local_cache.get(cache_key)
        if local_result is not None:
            self.cache_hits += 1
            return local_result == "sat"
        
        if self.manager and self.manager.cache:
            try:
                cached = self.manager.cache.get(cache_key)
                if cached is not None:
                    self.cache_hits += 1
                    result = cached.decode() == "sat"
                    self._local_cache.set(cache_key, cached.decode())
                    return result
            except Exception:
                pass
        return None
    
    def _store_cached(self, cache_key: str, is_sat: bool) -> None:
        """Store result in cache."""
        result_str = "sat" if is_sat else "unsat"
        self._local_cache.set(cache_key, result_str)
        
        if self.manager and self.manager.cache:
            try:
                self.manager.cache.set(cache_key, result_str)
            except Exception:
                pass
    
    def _check_partial_feasibility(
        self,
        partial_pc: List[ExprRef],
        partial_vars: set,
        new_pc: List[ExprRef],
    ) -> Tuple[bool, List[ExprRef], set]:
        """Check if merging is feasible.
        
        Implements slice → normalize → cache for cross-module queries.
        """
        new_vars = self._vars_in_pcs(new_pc)
        combined_pc = partial_pc + new_pc
        combined_vars = partial_vars | new_vars
        
        # Disjoint skip: (varsA \ {rst,clk}) ∩ (varsB \ {rst,clk}) == {}
        overlapping_vars = (partial_vars - VARS_EXCLUDED_FROM_DISJOINT) & (
            new_vars - VARS_EXCLUDED_FROM_DISJOINT
        )
        if not overlapping_vars:
            return (True, combined_pc, combined_vars)
        
        if not combined_pc:
            return (True, combined_pc, combined_vars)
        
        # Slice the query to constraints relevant to overlapping variables
        sliced_pc = combined_pc
        if self.manager and hasattr(self.manager, 'qu_merge') and self.manager.qu_merge is not None:
            try:
                from .query_slicing import slice_query
                # Register constraints
                for c in combined_pc:
                    self.manager.qu_merge.register_constraint(c)
                # Slice
                sliced_pc = slice_query(self.manager.qu_merge, combined_pc, overlapping_vars)
            except Exception:
                sliced_pc = combined_pc
        
        # Check cache with sliced query
        if self.enable_caching:
            cache_key = self._get_cache_key(sliced_pc)
            cached_result = self._check_cached(cache_key)
            if cached_result is not None:
                return (cached_result, combined_pc, combined_vars)
        
        is_sat = self._sat_check(sliced_pc)
        
        if self.enable_caching:
            self._store_cached(cache_key, is_sat)
        
        return (is_sat, combined_pc, combined_vars)
    
    def __iter__(self) -> Iterator[Tuple[Dict[str, List[dict]], List[ExprRef], Dict[str, Any]]]:
        """Iterate over all feasible cross-module combinations.
        
        Yields:
            (path_combo, all_pcs, all_stores) where:
            - path_combo: Dict mapping module_name -> list of cycle results
            - all_pcs: Combined path conditions
            - all_stores: Combined stores
        """
        if not self.levels:
            yield ({}, [], {})
            return
        
        # Track the current combo being built: module_name -> [cycle_results]
        current_combo: Dict[str, List[dict]] = {m: [] for m in self.module_names}
        
        # Initialize stack with first level
        self._push_level(0, [], {}, set(), current_combo)
        
        while self.stack:
            frame = self.stack[-1]
            module_name, cycle = self.levels[frame.level]
            
            try:
                result = next(frame.iterator)
                frame.current_result = result
            except StopIteration:
                self.stack.pop()
                # Backtrack: remove last added cycle result
                if frame.level > 0:
                    prev_module, prev_cycle = self.levels[frame.level - 1]
                    # Restore combo state (handled by stack frames)
                continue
            
            self.combos_checked += 1
            
            # Check partial feasibility
            if self.enable_early_pruning and frame.level > 0:
                is_feasible, new_pc, new_vars = self._check_partial_feasibility(
                    frame.partial_pc,
                    frame.partial_vars,
                    result["pc"],
                )
                if not is_feasible:
                    self.combos_pruned += 1
                    continue
            else:
                new_pc = frame.partial_pc + result["pc"]
                new_vars = frame.partial_vars | self._vars_in_pcs(result["pc"])
            
            # Update store with module-qualified names
            new_store = dict(frame.partial_store)
            for sig, expr in result["store"].items():
                new_store[f"{module_name}.{sig}"] = expr
            
            # Build current combo
            new_combo = {m: list(frame.partial_combo[m]) for m in self.module_names}
            new_combo[module_name].append(result)
            
            if frame.level == self.num_levels - 1:
                # Yield complete combination
                yield (new_combo, new_pc, new_store)
            else:
                # Push next level
                self._push_level(frame.level + 1, new_pc, new_store, new_vars, new_combo)
    
    def _push_level(
        self,
        level: int,
        partial_pc: List[ExprRef],
        partial_store: Dict[str, Any],
        partial_vars: set,
        partial_combo: Dict[str, List[dict]],
    ) -> None:
        """Push a new level onto the DFS stack."""
        module_name, cycle = self.levels[level]
        
        frame = DFSFrame(
            level=level,
            iterator=iter(self.per_module_results[module_name]),
            current_result=None,
            partial_pc=partial_pc,
            partial_store=partial_store,
            partial_vars=partial_vars,
        )
        # Store combo state in frame (extend DFSFrame for this)
        frame.partial_combo = partial_combo  # type: ignore
        self.stack.append(frame)
    
    def get_stats(self) -> Dict[str, int]:
        """Return iteration statistics."""
        return {
            "combos_checked": self.combos_checked,
            "combos_pruned": self.combos_pruned,
            "cache_hits": self.cache_hits,
        }


