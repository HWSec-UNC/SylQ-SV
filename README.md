# SylQ-SV - A Symbolic Execution Engine for SystemVerilog

## Requirements
* A Linux-based system (tested on Ubuntu 20.04)
* Recommended: 32+ GB RAM 
* Python3: 3.10 or later

## Getting Started
Clone the repo, including the submodules:

`git clone --recurse-submodules https://github.com/HWSec-UNC/SylQ-SV.git`

Install dependencies:
1. cd into SylQ-SV
2. Clone slang 10.x source code `git clone https://github.com/MikePopoloski/slang.git`
3. Create a Python venv `python3 -m venv venv` 
4. Activate the Python venv `source venv/bin/activate`
4. Install Python libraries `python3 -m pip install -r requirements.txt`

## Kick the Tires
Goal: Run SylQ-SV on the OR1200 for 1 clock cycle for 5 minutes with query caching enabled, and inspect the number of symbolic paths and solver queries generated.

To run the experiment, execute the following command:

`python3 -m main 1 --sv designs/benchmarks/or1200/or1200.F --use_cache --explore_time=300> out.txt`

The expected output for the kick-the-tires test is a summary of the number
of paths and branch points visited. The number of paths and branch points explored will vary based on your environment, but we saw around 50,000 paths exploredand 8,100 branch points explored. 

## Notes
* For assertion checking, embed SVA assertions directly into the top-level module of your RTL design.

## How To Cite
Please cite our ASPLOS paper when using SylQ-SV!

 @inproceedings{Ryan2025Sylq, 
 author = {Ryan, Kaki and Sturton, Cynthia}, 
 title = {{SylQ-SV}: Scaling Symbolic Execution of Hardware Designs with Query Caching}, 
 year = {2025}, 
 publisher = {ACM}, 
 booktitle = {Proceedings of the International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)}
 }
