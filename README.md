# SylQ-SV - A Symbolic Execution Engine for SystemVerilog

## Setup

Requirements
--------------------
* Python3: 3.9 or later
* pySlang 7.0: `python3 -m pip install pyslang`
* Redis 7.4: 

    `sudo apt-get install lsb-release curl gpg` 

    `curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg`

    `sudo chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg`

    `echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list`

    `sudo apt-get update`

    `sudo apt-get install redis`
- z3: run `python3 -m pip install z3-solver`
- Icarus Verilog: 10.1 or later: run `sudo apt install iverilog`
- Jinja 2.10 or later: run `python3 -m pip install jinja2`
- PLY 3.4 or later: run `python3 -m pip install ply`
- networkx: run `python3 -m pip install networkx`
- matplotlib: run `python3 -m pip install matplotlib`
