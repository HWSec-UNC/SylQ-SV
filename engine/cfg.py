"""Converts PySlang AST (representing SystemVerilog) into executable CFG structure that enables path exploration"""
from operator import indexOf
import os
import networkx as nx
import matplotlib.pyplot as plt
import pyslang.ast as ps_ast
from .basic_block_visitor import BasicBlockVisitor
from helpers.visitor_helpers import handles, build_lookup_table

class CFG:
    """Represents the control flow graph of a module/always block"""
    def __init__(self):
        # basic blocks. A list made up of slices of all_nodes determined by partition_points.
        self.basic_block_list = []

        # for partitioning
        self.curr_idx = 0

        # add all nodes in the always block
        self.all_nodes = []

        # partition indices
        self.partition_points = set()
        self.partition_points.add(0)

        # the edgelist will be a list of tuples of indices of the ast nodes blocks
        self.edgelist = []

        # edges between basic blocks, determined by the above edgelist
        self.cfg_edges = []

        # indices of basic blocks that need to connect to dummy exit node
        self.leaves = set()

        #paths... list of paths with start and end being the dummy nodes
        self.paths = []

        # name corresponding to the module. there could be multiple always blocks (or CFGS) per module
        self.module_name = ""

        # Whether this CFG represents combinational (always_comb) or sequential logic
        self.is_combinational = False

        # NetworkX DiGraph built by build_cfg; used by compute_direction
        self.graph = None

        # Decl nodes outside the always block to be executed once up front for all paths
        self.decls = []

        # Combinational logic nodes outside the always block to be twice for all paths
        self.comb = []

        # the nodes in the AST that correspond to always blocks
        self.always_blocks = []
        # always_comb / always_latch blocks (Paper §4.4 piecewise optimisation)
        self.always_comb_blocks = []

        # branch-point set
        # for each basic statement, there may be some indpendent branching points
        self.ind_branch_points = {1: set()}

        # stack of flags for if we are looking at a block statement
        self.block_smt = [False]

        # how many nested block statements we've seen so far
        self.block_stmt_depth = 0

        #submodules defined
        self.submodules = []

        # InstanceBodySymbol for the module being analyzed; set by get_always_sv
        self._instance_body = None

        # Visitor for partitioning and building basic blocks
        self.basic_block_visitor = BasicBlockVisitor(self)

    def reset(self):
        """Return to defaults."""
        self.__init__()

    def get_always_sv(self, ast):
        """
        Extracts always blocks from PySlang AST using the always block visitor. The visitor
        will populate our lists of always blocks, declarations, and
        combinational logic statements automatically.
        """
        self._instance_body = ast
        visitor = AlwaysBlockVisitor(
            self.always_blocks, self.always_comb_blocks, self.decls, self.comb,
        )
        ast.visit(lookup_table=visitor.lookup_table)

    def build_cfg(self, always_block):
        """Build networkx digraph / CFG for the Statement body of a ProceduralBlock symbol."""

        # Give the visitor the module scope so it can evaluate loop bounds
        self.basic_block_visitor._scope = self._instance_body

        # finds all edges and partition points in the always block for the CFG
        always_block.visit(lookup_table=self.basic_block_visitor.lookup_table)

        # Partitions the nodes into basic blocks
        # The partition points are the first node in each basic block
        self._partition()

        # Finds the paths between basic blocks
        self._make_paths()

        G = nx.DiGraph()
        for block in self.basic_block_list:
            G.add_node(indexOf(self.basic_block_list, block), data=tuple(block))

        G.add_node(-1, data="Dummy Start")
        G.add_node(-2, data="Dummy End")

        for block1, block2, condition in self.cfg_edges:
            G.add_edge(block1, block2, condition=condition)

        G.add_edge(-1, 0)

        # Connect dangling control-flow edges (e.g. if-without-else false
        # branches) to the exit node BEFORE computing leaves so _find_leaves
        # won't duplicate them.
        exit_connected = set()
        for (node_idx, condition) in self.basic_block_visitor.edge_stack:
            bb = self._find_basic_block(node_idx)
            G.add_edge(bb, -2, condition=condition)
            exit_connected.add(bb)

        self._find_leaves()
        for leaf in self.leaves:
            if leaf not in exit_connected:
                G.add_edge(leaf, -2)

        self.graph = G

        # TODO: Will this materialization blow up memory?
        # Should restore lazy generator-based path enumeration for large designs.
        self.paths = list(nx.all_simple_paths(G, source=-1, target=-2))

    def _partition(self):
        """Partitions all_nodes into basic blocks based on partition_points"""
        if not self.all_nodes:
            return

        sorted_points = sorted(list(self.partition_points))
        sorted_points.append(len(self.all_nodes))

        self.basic_block_list = []

        for i in range(len(sorted_points) - 1):
            start_idx = sorted_points[i]
            end_idx = sorted_points[i+1]

            # Slice from this leader to the next leader
            basic_block = self.all_nodes[start_idx:end_idx]

            if basic_block:
                self.basic_block_list.append(basic_block)

    def _make_paths(self):
        """Map the edge between AST nodes to a path between basic blocks."""
        for (node1, node2, condition) in self.edgelist:
            block1 = self._find_basic_block(node1)
            block2 = self._find_basic_block(node2)

            if block1 != block2:
                self.cfg_edges.append((block1, block2, condition))

    def _find_basic_block(self, node_idx):
        """Given a node index, find the index of the basic block that we're in."""
        if node_idx < len(self.all_nodes):
            node = self.all_nodes[node_idx]
        else:
            node = self.all_nodes[len(self.all_nodes)-1]

        for block in self.basic_block_list:
            if node in block:
                return indexOf(self.basic_block_list, block)

        raise ValueError(
            f"Node index {node_idx} not found in any basic block "
            f"(all_nodes length={len(self.all_nodes)}, "
            f"basic_block_list length={len(self.basic_block_list)})"
        )

    def _find_leaves(self):
        """Find leaves in cfg, to know which nodes need to connect to dummy exit."""
        if self.cfg_edges:
            starts = set(edge[0] for edge in self.cfg_edges)
            ends = set(edge[1] for edge in self.cfg_edges)
            self.leaves = ends - starts
        elif self.basic_block_list:
            self.leaves = {len(self.basic_block_list) - 1}
        else:
            self.leaves = set()

    def get_paths(self):
        """Return an iterator over all CFG paths (backward-compat wrapper)."""
        return iter(self.paths)

    def get_path_count(self):
        """Return the number of paths in the CFG (backward-compat wrapper)."""
        return len(self.paths)

    def compute_direction(self, path):
        """Derive a direction value for each real basic block from edge conditions.

        Returns a list aligned with the non-negative nodes in *path*.
        Mapping: 'true' → 1, 'false' → 0, None (sequential) → 1, other → label.
        """
        if self.graph is None:
            return []
        directions = []
        for i in range(len(path)):
            if path[i] < 0:
                continue
            if i > 0:
                edge_data = self.graph.get_edge_data(path[i - 1], path[i])
                if edge_data:
                    cond = edge_data.get('condition')
                    if cond == 'true':
                        directions.append(1)
                    elif cond == 'false':
                        directions.append(0)
                    elif cond is None:
                        directions.append(1)
                    else:
                        directions.append(cond)
                else:
                    directions.append(1)
            else:
                directions.append(1)
        return directions

    def _print_simple_paths_with_conditions(self, G, paths):
        for i, path in enumerate(paths):
            print(f"\n--- Path {i} ---")
            path_str = ""

            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]

                edge_data = G.get_edge_data(u, v)
                condition = edge_data.get("condition", "sequential")
                if condition is None:
                    condition = "sequential"

                if condition == "true":
                    label = "[TRUE] -> "
                elif condition == "false":
                    label = "[FALSE] -> "
                elif condition == 'sequential':
                    label = "-> "
                else:
                    label = f"[CASE: {condition}] -> "

                path_str += f"Block({u}) {label}"

            path_str += f"Block({path[-1]})"
            print(path_str)

    def save_cfg_visualization(self, G, filename="cfg_output.png"):
        """
        Renders the CFG using a multipartite (layered) layout and saves it.
        """
        plt.figure(figsize=(14, 12))

        try:
            levels = nx.shortest_path_length(G, source=-1)
            for node, dist in levels.items():
                G.nodes[node]['layer'] = dist

            max_level = max(levels.values()) if levels else 0
            G.nodes[-2]['layer'] = max_level + 1

            pos = nx.multipartite_layout(G, subset_key="layer", align='vertical')

            pos = {node: (coords[1], -coords[0]) for node, coords in pos.items()}

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pos = nx.spring_layout(G, seed=42)

        color_map = []
        for node in G.nodes():
            if node == -1: color_map.append('limegreen')
            elif node == -2: color_map.append('tomato')
            else: color_map.append('skyblue')

        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=color_map, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        nx.draw_networkx_edges(
            G, pos,
            arrowstyle='->',
            arrowsize=20,
            edge_color='gray',
            width=1.5,
            connectionstyle="arc3,rad=0.1"
        )

        edge_labels = {}
        for u, v, data in G.edges(data=True):
            cond = data.get('condition')
            if cond is not None:
                edge_labels[(u, v)] = str(cond)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=9)

        plt.title(f"Multipartite CFG: {filename}", pad=20)
        plt.axis('off')

        file_ext = os.path.splitext(filename)[1][1:]
        plt.savefig(filename, format=file_ext, bbox_inches='tight', dpi=300)
        print(f"CFG saved successfully to {os.path.abspath(filename)}")
        plt.close()

class AlwaysBlockVisitor:
    """
    Visitor for a pyslang InstanceBody symbol that extracts ProceduralBlock,
    Variable/Net, and ContinuousAssign symbols from the symbol tree.
    """

    _COMB_KINDS = frozenset({
        ps_ast.ProceduralBlockKind.AlwaysComb,
        ps_ast.ProceduralBlockKind.AlwaysLatch,
    })
    _SKIP_KINDS = frozenset({
        ps_ast.ProceduralBlockKind.Initial,
        ps_ast.ProceduralBlockKind.Final,
    })

    def __init__(self, always_blocks, always_comb_blocks, decls, comb):
        self.always_blocks = always_blocks
        self.always_comb_blocks = always_comb_blocks
        self.decls = decls
        self.comb = comb
        self.lookup_table = build_lookup_table(self)

    def __call__(self, node):
        """Visitor dispatcher called by pyslang. Only handles Symbol nodes."""
        if not isinstance(node, ps_ast.Symbol):
            return
        handler = self.lookup_table.get(node.kind)
        if handler:
            return handler(node)

    ### SYMBOL HANDLERS ###

    @handles(ps_ast.SymbolKind.ProceduralBlock)
    def handle_procedural_block(self, node: ps_ast.Symbol):
        """Sort procedural blocks by kind: comb vs ff/always, skip initial/final."""
        kind = node.procedureKind
        if kind in self._SKIP_KINDS:
            return ps_ast.VisitAction.Skip
        if kind in self._COMB_KINDS:
            self.always_comb_blocks.append(node)
        else:
            self.always_blocks.append(node)
        return ps_ast.VisitAction.Skip

    @handles(ps_ast.SymbolKind.Variable, ps_ast.SymbolKind.Net)
    def handle_variable(self, node: ps_ast.Symbol):
        """Captures data declarations."""
        self.decls.append(node)

    @handles(ps_ast.SymbolKind.ContinuousAssign)
    def handle_continuous_assign(self, node: ps_ast.Symbol):
        """Captures continuous assignments (combinational logic)."""
        self.comb.append(node)
