# SylQ-SV - A Symbolic Execution Engine for SystemVerilog

## Requirements
* A Linux-based system (tested on Ubuntu 20.04)
* Recommended: 32+ GB RAM 
* Python3: 3.9 or later

## Getting Started
Install dependencies:
* pySlang 7.0: `python3 -m pip install pyslang`
* Redis 7.4: run `python3 -m pip install redis`
* z3: run `python3 -m pip install z3-solver`
* Icarus Verilog: 10.1 or later: run `sudo apt install iverilog`
* Jinja 2.10 or later: run `python3 -m pip install jinja2`
* PLY 3.4 or later: run `python3 -m pip install ply`
* networkx: run `python3 -m pip install networkx`
* matplotlib: run `python3 -m pip install matplotlib`

## Kick-the Tires
Goal: Run SylQ-SV on the OR1200 for 1 clock cycle for 5 minutes with query caching enabled, and inspect the number of symbolic paths and solver queries generated.

To run the experiment, execute the following command:

`python3 -m main 1 --sv designs/benchmarks/or1200/or1200.F --use_cache --explore_time=300> out.txt`

The expected output for the kick-the-tires test is a summary of the number
of paths and branch points visited. The number of paths and branch points explored will vary based on your environment, but we saw around 50,000 paths exploredand 8,100 branch points explored. 

## Notes
* For assertion checking, embed SVA assertions directly into the top-level module of your RTL design.
* The Redis cache file (`cache.rdb`) is persistent across runs and used to analyze query reuse.

## How To Cite
Please cite our ASPLOS paper when using SylQ-SV!

@inproceedings{Ryan2025Sylq, author = {Ryan, Kaki and Sturton, Cynthia}, title = {SylQ-SV: Scaling Symbolic Execution of Hardware Designs with Query Caching}, year = {2025}, publisher = {ACM}, booktitle = {Proceedings of the ACM International Conference on Architectural Support for Programming Languages and Operating Systems}, series = {ASPLOS} }
