"""This file is the entrypoint of the execution."""
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
from optparse import OptionParser
import time
import logging
import gc
from engine.execution_engine import ExecutionEngine
import pyslang.driver as ps_driver
from helpers.slang_helpers import SlangSymbolVisitor, SymbolicDFS
import redis
import threading
import time

gc.collect()

with open('errors.log', 'w'):
    pass
logging.basicConfig(filename='errors.log', level=logging.DEBUG)
logging.debug("Starting over")


INFO = "SylQ-SV Symbolic Execution Engine"
USAGE = "Usage: python3 -m main <num_cycles> <verilog_file>.v > out.txt"

# Global references so timeout_exit can print summary stats and request stop
_engine_ref = None
_start_time_ref = None

def timeout_exit():
    """This only happens when the timer runs out."""
    elapsed = time.process_time() - _start_time_ref if _start_time_ref else 0
    print("\n=== TIMEOUT: Execution time limit exceeded ===", flush=True)
    print(f"  Elapsed time: {elapsed:.4f}s", flush=True)
    if _engine_ref and hasattr(_engine_ref, '_last_manager'):
        # Mark the engine as timed out so long-running loops can exit cooperatively.
        try:
            setattr(_engine_ref, "timeout", True)
        except Exception:
            pass
        mgr = _engine_ref._last_manager
        if getattr(mgr, "estimated_global_combinations", None) is not None:
            print(
                f"  Paths explored (feasible): {mgr.estimated_global_combinations:,} "
                f"(estimated global product from per-module merge sizes)",
                flush=True,
            )
        elif getattr(mgr, "feasible_paths_unknown", False):
            print(
                "  Paths explored (feasible): not summarized as one number (lazy merge — "
                "global product not computed without iterating). See per-module "
                "'feasible merged paths' / merge lines above.",
                flush=True,
            )
        else:
            print(f"  Paths explored (feasible): {mgr.path_count}", flush=True)
        print(
            f"  AST branch visits (SymbolicDFS if/case/while): {mgr.branch_count} "
            f"(often 0 in piecewise mode: CFG uses edge constraints, not this counter)",
            flush=True,
        )
        n_assertions = len(mgr.assertions) if hasattr(mgr, 'assertions') else 0
        # Same count as final summary ("Assertions found"); not "checks per path".
        print(f"  Assertions found: {n_assertions}", flush=True)
        _feas = getattr(mgr, "solver_time", 0) or 0
        _asrt = getattr(mgr, "assertion_solver_time", 0) or 0
        print(
            f"  Solver time: feasibility={_feas:.4f}s, assertions={_asrt:.4f}s "
            f"(total {_feas + _asrt:.4f}s)",
            flush=True,
        )
        if hasattr(mgr, "feasibility_stats_line"):
            print(f"  {mgr.feasibility_stats_line()}", flush=True)
        if mgr.assertion_violation:
            print("  Result: VIOLATION FOUND (see above)", flush=True)
        else:
            print("  Result: No violation found within time limit", flush=True)
    print("=======================================", flush=True)

def showVersion():
    print(INFO)
    print(USAGE)
    sys.exit()
    
def main():
    """Entrypoint of the program."""
    # OR1200 and large designs recurse deeply; avoid RecursionError.
    sys.setrecursionlimit(50000)
    engine: ExecutionEngine = ExecutionEngine()
    optparser = OptionParser()
    optparser.add_option("-v", "--version", action="store_true", dest="showversion",
                         default=False, help="Show the version")
    optparser.add_option("-I", "--include", dest="include", action="append",
                         help="Include path")
    optparser.add_option("-D", dest="define", action="append",
                         default=[], help="Macro Definition")
    optparser.add_option("-B", "--debug", action="store_true", dest="showdebug", help="Debug Mode")
    optparser.add_option("-t", "--top", dest="topmodule",
                         default=None, help="Top module name (filters topInstances to this one)")
    optparser.add_option("--nobind", action="store_true", dest="nobind",
                         default=False, help="No binding traversal, Default=False")
    optparser.add_option("--noreorder", action="store_true", dest="noreorder",
                         default=False, help="No reordering of binding dataflow, Default=False")
    optparser.add_option("-o", "--output", dest="outputfile",
                         default="out.png", help="Graph file name, Default=out.png")
    optparser.add_option("-s", "--search", dest="searchtarget", action="append",
                         default=[], help="Search Target Signal")
    optparser.add_option("--sv", action="store_true", dest="sv",
                         default=False, help="enable SystemVerilog parser")
    optparser.add_option("--walk", action="store_true", dest="walk",
                         default=False, help="Walk contineous signals, Default=False")
    optparser.add_option("--identical", action="store_true", dest="identical",
                         default=False, help="# Identical Laef, Default=False")
    optparser.add_option("--step", dest="step", type='int',
                         default=1, help="# Search Steps, Default=1")
    optparser.add_option("--reorder", action="store_true", dest="reorder",
                         default=False, help="Reorder the contineous tree, Default=False")
    optparser.add_option("--delay", action="store_true", dest="delay",
                         default=False, help="Inset Delay Node to walk Regs, Default=False")
    optparser.add_option("--use_cache", action="store_true", dest="use_cache",
                         default=False, help="Use the query caching, Default=False")
    optparser.add_option("--explore_time", help="Time to explore in seconds", dest="explore_time")
    optparser.add_option(
        "--max-cross-module-paths",
        type="int",
        dest="max_cross_module_paths",
        default=None,
        help="Stop cross-module DFS after N global path combinations. "
             "Omit for no limit (can be very slow on large designs; use --explore_time to cap wall time)",
    )
    (options, args) = optparser.parse_args()


    num_cycles = args[0]
    filelist = args[1:]

    if options.showversion:
        showVersion()
    
    if options.use_cache:
        engine.cache = redis.Redis(host='localhost', port=6379, db=0)

    timer = None
    if options.explore_time:
        timer = threading.Timer(int(options.explore_time), timeout_exit)
        timer.start()

    if options.showdebug:
        engine.debug = True

    for f in filelist:
        if not os.path.exists(f):
            raise IOError("file not found: " + f)

    # Create a .F file listing all source files (pyslang expects command files, not source files directly).
    # If the single argument is already a .F/.f file, use it as the file list; otherwise write filelist.F.
    # Include paths (-I) are written as +incdir+<path> lines.
    if len(filelist) == 1 and (filelist[0].endswith('.F') or filelist[0].endswith('.f')):
        if not os.path.exists(filelist[0]):
            raise IOError("file list not found: " + filelist[0])
        # If user-provided .F file and we have includes, prepend them to a new file
        if options.include:
            with open(filelist[0], 'r') as orig:
                orig_content = orig.read()
            flist_path = "filelist.F"
            with open(flist_path, "w") as flist:
                for inc in options.include:
                    flist.write(f"+incdir+{inc}\n")
                flist.write(orig_content)
            filelist = [flist_path]
    elif len(filelist) >= 1:
        flist_path = "filelist.F"
        with open(flist_path, "w") as flist:
            if options.include:
                for inc in options.include:
                    flist.write(f"+incdir+{inc}\n")
            for f in filelist:
                flist.write(f + "\n")
        filelist = [flist_path]

    if len(filelist) == 0:
        showVersion()
    
    if options.sv:
        start = time.process_time()
        driver = ps_driver.Driver()
        driver.addStandardArgs()
        driver.processCommandFiles(filelist[0], True, True)
        driver.processOptions()
        driver.parseAllSources()
        
        compilation = driver.createCompilation()

        # pyslang 9.x has runFullCompilation; 7.x has reportCompilation
        if hasattr(driver, 'runFullCompilation'):
            successful_compilation = driver.runFullCompilation(False)
        else:
            successful_compilation = driver.reportCompilation(compilation, False)
        
        if successful_compilation:
            root = compilation.getRoot()
            top_instances = root.topInstances

            if options.topmodule:
                filtered = [t for t in top_instances if t.name == options.topmodule]
                if not filtered:
                    all_names = [t.name for t in top_instances]
                    print(f"ERROR: --top '{options.topmodule}' not found in topInstances: {all_names}",
                          file=sys.stderr, flush=True)
                    exit(1)
                top_instances = filtered
                print(f"--top: filtered to {options.topmodule} "
                      f"(skipping {len(root.topInstances) - 1} other top-level modules)",
                      flush=True)

            my_visitor_for_symbol = SymbolicDFS(num_cycles)

            symbol_visitor = SlangSymbolVisitor()

            # Store global refs for timeout reporting
            global _engine_ref, _start_time_ref
            _engine_ref = engine
            _start_time_ref = start

            engine.execute_sv(
                my_visitor_for_symbol,
                top_instances,
                None,
                num_cycles,
                max_cross_module_paths=options.max_cross_module_paths,
            )
            symbol_visitor.visit(top_instances)
            print(
                "  AST scan (SlangSymbolVisitor): branch_points={} paths={}".format(
                    symbol_visitor.branch_points, symbol_visitor.paths
                ),
                flush=True,
            )

        end = time.process_time()
        elapsed = end - start

        # --- Final summary report ---
        print("\n=== EXECUTION SUMMARY ===", flush=True)
        print(f"  Elapsed time: {elapsed:.4f}s", flush=True)
        if hasattr(engine, '_last_manager'):
            mgr = engine._last_manager
            if getattr(mgr, "estimated_global_combinations", None) is not None:
                print(
                    f"  Paths explored (feasible): {mgr.estimated_global_combinations:,} "
                    f"(estimated global product from per-module merge sizes)",
                    flush=True,
                )
            elif getattr(mgr, "feasible_paths_unknown", False):
                print(
                    "  Paths explored (feasible): not summarized as one number (lazy merge — "
                    "see log above for per-module feasible merged paths).",
                    flush=True,
                )
            else:
                print(f"  Paths explored (feasible): {mgr.path_count}", flush=True)
            print(
                f"  AST branch visits (SymbolicDFS if/case/while): {mgr.branch_count} "
                f"(often 0 in piecewise mode: CFG uses edge constraints, not this counter)",
                flush=True,
            )
            n_assertions = len(mgr.assertions) if hasattr(mgr, 'assertions') else 0
            print(f"  Assertions found: {n_assertions}", flush=True)
            print(
                f"  Solver time: feasibility={mgr.solver_time:.4f}s, "
                f"assertions={getattr(mgr, 'assertion_solver_time', 0) or 0:.4f}s "
                f"(total {mgr.solver_time + (getattr(mgr, 'assertion_solver_time', 0) or 0):.4f}s)",
                flush=True,
            )
            if hasattr(mgr, "feasibility_stats_line"):
                print(f"  {mgr.feasibility_stats_line()}", flush=True)
            if mgr.assertion_violation:
                print("  Result: ASSERTION VIOLATION FOUND", flush=True)
            elif n_assertions > 0:
                print("  Result: All paths explored, no violations", flush=True)
            elif getattr(mgr, "cross_module_stopped_reason", "") == "complete":
                print("  Result: Cross-module enumeration finished (exhausted feasible combinations)",
                      flush=True)
            elif getattr(mgr, "cross_module_stopped_reason", "") == "max_paths":
                print("  Result: Cross-module enumeration stopped at --max-cross-module-paths limit",
                      flush=True)
            elif getattr(mgr, "cross_module_stopped_reason", "") == "timeout":
                print("  Result: Cross-module enumeration interrupted (timeout)", flush=True)
            else:
                print("  Result: Exploratory run complete (no assertions in design)", flush=True)
        print("=========================", flush=True)

        if timer:
            timer.cancel()
        exit()

if __name__ == '__main__':
    main()