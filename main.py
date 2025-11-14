"""This file is the entrypoint of the execution."""
from __future__ import absolute_import
from __future__ import print_function
import z3
from z3 import Solver, Int, BitVec, Context, BitVecSort, ExprRef, BitVecRef, If, BitVecVal, And
import sys
import os
from optparse import OptionParser
from typing import Optional
import random, string
import time
from itertools import product
import logging
import gc
from engine.execution_manager import ExecutionManager
from engine.symbolic_state import SymbolicState
from helpers.rvalue_parser import tokenize, parse_tokens, evaluate
from engine.execution_engine import ExecutionEngine
import pyslang as ps
from helpers.slang_helpers import SlangSymbolVisitor, SymbolicDFS
# SlangNodeVisitor removed 
import redis
import threading
import time

from helpers.rvalue_to_z3 import parse_expr_to_Z3

gc.collect()

with open('errors.log', 'w'):
    pass
logging.basicConfig(filename='errors.log', level=logging.DEBUG)
logging.debug("Starting over")


INFO = "Verilog Symbolic Execution Engine"
USAGE = "Usage: python3 -m main <num_cycles> <verilog_file>.v > out.txt"
    
def timeout_exit():
    """This only happens when the timer runs out."""
    print("Execution time limit exceeded. Exiting.")
    sys.exit(1)

def showVersion():
    print(INFO)
    print(USAGE)
    sys.exit()
    
def main():
    """Entrypoint of the program."""
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
                         default="top", help="Top module, Default=top")
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

    # If more than one file, create a .F file listing all files
    if len(filelist) > 1:
        flist_path = "filelist.F"
        with open(flist_path, "w") as flist:
            for f in filelist:
                flist.write(f + "\n")
        filelist = [flist_path]

    if len(filelist) == 0:
        showVersion()
    
    if options.sv:
        start = time.process_time()
        driver = ps.Driver()
        driver.addStandardArgs()
        driver.processCommandFiles(filelist[0], True, True)
        driver.processOptions()
        driver.parseAllSources()
        
        compilation = driver.createCompilation()
        modules =  compilation.getDefinitions()

        # IF using pyslang 7.0.0, uncomment the following line and comment out the other successful_compilation
        # successful_compilation = driver.reportCompilation(compilation, False)

        #IF using pyslang 9.0.0/9.1.0, use this version of successful_compilation
        successful_compilation = driver.runFullCompilation(False)
        
        if successful_compilation:
            #print(driver.reportMacros())
            my_visitor_for_symbol = SymbolicDFS(num_cycles)
            #delegate method from z3Visitor
            my_visitor_for_symbol.expr_to_z3 = lambda m, s, e: parse_expr_to_Z3(e, s, m)

            symbol_visitor = SlangSymbolVisitor(num_cycles)
            engine.execute_sv(my_visitor_for_symbol, modules, None, num_cycles)
            symbol_visitor.visit(modules)
            print(symbol_visitor.branch_points)
            print(symbol_visitor.paths)
            
        end = time.process_time()
        print(f"Elapsed time {end - start}")
        if timer:
            timer.cancel()
        exit()

if __name__ == '__main__':
    main()
