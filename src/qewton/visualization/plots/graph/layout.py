"""Renderer-agnostic layered layout of a computation graph for
:class:`~qewton.visualization.plots.graph.GraphPlot`."""

from qewton.constraints.base import Constraint
from qewton.data.dataloaders.base import DataNode
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.visualization.plots.result import ClusterBox, EdgeLayout, GraphLayoutResult, NodeLayout, PortLayout

# Layout constants, in arbitrary but self-consistent units (not pixels).
# Node width is fixed; the node title is drawn rotated by 90 degrees, so a
# node's height grows with its label length instead (see _node_size).
NODE_WIDTH = 1.0
COLUMN_GAP = 0.9
ROW_GAP = 0.6
PORT_ROW_HEIGHT = 0.35
BASE_HEIGHT = 0.55
# Per-character height contribution of the vertical node title.
CHAR_HEIGHT = 0.1
LABEL_PADDING = 0.4
CLUSTER_MARGIN = 0.5
CLUSTER_GAP = 0.8
BARYCENTER_PASSES = 4
# Distance in rows within which a long edge (one skipping at least one
# rank) counts as cutting through a node it passes; the penalty tapers
# linearly to 0 at this distance. See _reduce_node_crossings.
NODE_CROSSING_RADIUS = 1.0


class GraphLayout:
    """Computes renderer-agnostic node, port, edge and cluster positions for
    a graph.

    A layered DAG layout drawn left to right: a node's rank (column) is the
    longest path from a source, then a few barycenter passes reorder each
    rank to reduce edge crossings, followed by a pass that moves unrelated
    nodes out of the way of edges skipping ranks. Expanded composites are
    laid out recursively inside a cluster.

    Notes:
        Layout runs in two passes: first every node, port and cluster is placed
        (recursing into expanded composites), then every edge is built from the
        collected port positions, since a composite's boundary ports are only
        positioned once its content has been measured.
    """

    @staticmethod
    def compute(graph, depth: int = 0) -> GraphLayoutResult:
        """Compute the layout of `graph`.

        Args:
            graph (Graph): The sorted :class:`~qewton.graphs.graphs.Graph` to
                lay out.
            depth: How many levels of composite nodes are expanded into
                clusters.

        Returns:
            GraphLayoutResult: All nodes, ports, edges and clusters in one
                global frame.

        Raises:
            AssertionError: If the graph has not been sorted.
        """
        assert graph.sorted_nodes, (
            "Graph must be sorted before plotting - call graph.sort() (or "
            "graph.setup()) first."
        )
        nodes, ports, clusters, port_anchor, levels = [], [], [], {}, []
        _place_level(graph, depth, 0.0, 0.0, nodes, ports, clusters, port_anchor, levels)

        edges = []
        for graph_like in levels:
            _build_edges(graph_like, port_anchor, edges)
        return GraphLayoutResult(nodes=nodes, ports=ports, edges=edges, clusters=clusters)


def _node_label(node) -> str:
    return node.name


def _node_category(node) -> str:
    """Coloring bucket of a node. Constraint is checked before GraphNode,
    since some constraints (e.g. MSEConstraint) are also GraphNodes."""
    if isinstance(node, Constraint):
        return "constraint"
    if isinstance(node, DataNode):
        return "datanode"
    if isinstance(node, GraphNode):
        return "graphnode"
    return "default"


def _node_size(node) -> tuple[float, float]:
    """(width, height) of a leaf node's box. Width is NODE_WIDTH; height is
    the max of the base minimum, enough rows for the node's ports, and
    enough length for its vertical title."""
    width = NODE_WIDTH
    n_ports = max(1, len(node.input_ports), len(node.output_ports))
    port_height = PORT_ROW_HEIGHT * n_ports + 0.2
    title_height = CHAR_HEIGHT * len(_node_label(node)) + LABEL_PADDING
    height = max(BASE_HEIGHT, port_height, title_height)
    return width, height


def _register_ports(
    ports, x, center_y, height, port_anchor, ports_out, show_labels: bool, is_input: bool
) -> list[PortLayout]:
    """Anchors `ports` stacked vertically at a fixed x (left edge for inputs,
    right edge for outputs), records each anchor in the shared port_anchor
    map used by edge building, and appends each PortLayout to ports_out."""
    n = len(ports)
    layouts = []
    for i, port in enumerate(ports):
        y = center_y - height / 2 + (i + 0.5) * height / n if n else center_y
        layout = PortLayout(name=port.name, x=x, y=y, is_input=is_input, show_label=show_labels)
        layouts.append(layout)
        ports_out.append(layout)
        port_anchor[port] = (x, y)
    return layouts


def _rank_nodes(graph_like, level_nodes) -> dict:
    """Rank of each node: longest path from a source, in one pass over the
    topologically sorted level_nodes. Edges from outside this level are
    ignored, so a boundary-fed node is a source within its level."""
    node_set = set(level_nodes)
    rank = {}
    for node in level_nodes:
        preds = [
            edge.from_port.node
            for edge in graph_like.incoming_edges.get(node, [])
            if edge.from_port.node in node_set
        ]
        rank[node] = max((rank[p] for p in preds), default=-1) + 1
    return rank


def _reduce_crossings(graph_like, level_nodes, by_rank: dict[int, list], rank: dict) -> None:
    """Barycenter passes (Sugiyama heuristic): reorder each rank by the mean
    position of its neighbors in the adjacent rank, alternating the sweep
    direction; then _reduce_node_crossings for long edges. Mutates
    by_rank's lists in place."""
    node_set = set(level_nodes)
    position = {node: float(i) for nodes in by_rank.values() for i, node in enumerate(nodes)}

    def neighbors(node, incoming: bool):
        edges = (
            graph_like.incoming_edges.get(node, [])
            if incoming
            else graph_like.outgoing_edges.get(node, [])
        )
        attr = "from_port" if incoming else "to_port"
        return [p for e in edges if (p := getattr(e, attr).node) in node_set]

    def reorder(rank_nodes, incoming: bool):
        if not rank_nodes:
            return
        def barycenter(node):
            ns = neighbors(node, incoming)
            return sum(position[n] for n in ns) / len(ns) if ns else position[node]
        rank_nodes.sort(key=barycenter)
        for i, node in enumerate(rank_nodes):
            position[node] = float(i)

    max_rank = max(by_rank) if by_rank else 0
    for _ in range(BARYCENTER_PASSES):
        for r in range(1, max_rank + 1):
            reorder(by_rank.get(r, []), incoming=True)
        for r in range(max_rank - 1, -1, -1):
            reorder(by_rank.get(r, []), incoming=False)

    # A long edge (rank gap > 1) has no endpoint in the ranks it passes
    # through, so barycenter ordering cannot keep nodes there out of its
    # way; _reduce_node_crossings handles that case.
    def _related(a, b) -> frozenset:
        """`a`, `b` and every node directly adjacent to either. These are
        exempt from the node-crossing penalty of the edge (a, b)."""
        related = {a, b}
        for n in (a, b):
            related.update(p for e in graph_like.outgoing_edges.get(n, []) if (p := e.to_port.node) in node_set)
            related.update(p for e in graph_like.incoming_edges.get(n, []) if (p := e.from_port.node) in node_set)
        return frozenset(related)

    def _candidate_edges():
        # Regular edges. outgoing_edges can point at a node outside this
        # level (a boundary edge to the composite itself), which has no
        # rank, so filter by node_set.
        for node in level_nodes:
            for edge in graph_like.outgoing_edges.get(node, []):
                if edge.to_port.node in node_set:
                    yield node, edge.to_port.node
        # Skip connections do not affect ranks but usually span several,
        # so they are candidates too.
        for edge in getattr(graph_like, "skip_connections", []):
            if edge.from_port.node in node_set and edge.to_port.node in node_set:
                yield edge.from_port.node, edge.to_port.node

    long_edges = [
        (src, dst, _related(src, dst))
        for src, dst in _candidate_edges()
        if rank[dst] - rank[src] > 1
    ]
    _reduce_node_crossings(by_rank, rank, position, long_edges)


def _reduce_node_crossings(
    by_rank: dict[int, list], rank: dict, position: dict, long_edges: list[tuple]
) -> None:
    """Reorders each rank so that unrelated nodes avoid the long edges
    passing through it.

    For each intermediate rank `r` of a long edge, the edge's straight line
    gives a virtual row at `r` (linear interpolation of its endpoint rows;
    row indices are comparable across ranks because all ranks of a level
    share one row height). A node within NODE_CROSSING_RADIUS rows of it is
    penalized, unless it is one of the edge's related nodes. Greedy
    adjacent swaps are applied until no swap lowers the penalty. Mutates
    by_rank and position in place."""
    if not long_edges:
        return

    def penalty_at(node, r) -> float:
        row = position[node]
        total = 0.0
        for src, dst, related in long_edges:
            if node in related:
                continue
            sr, dr = rank[src], rank[dst]
            if not (sr < r < dr):
                continue
            t = (r - sr) / (dr - sr)
            virtual_row = position[src] + (position[dst] - position[src]) * t
            gap = abs(row - virtual_row)
            if gap < NODE_CROSSING_RADIUS:
                total += 1.0 - gap / NODE_CROSSING_RADIUS
        return total

    for r, rank_nodes in by_rank.items():
        if len(rank_nodes) < 2:
            continue
        if not any(rank[src] < r < rank[dst] for src, dst, _related in long_edges):
            continue  # no long edge even spans this rank - nothing to dodge
        changed = True
        while changed:
            changed = False
            for i in range(len(rank_nodes) - 1):
                a, b = rank_nodes[i], rank_nodes[i + 1]
                before = penalty_at(a, r) + penalty_at(b, r)
                position[a], position[b] = position[b], position[a]
                after = penalty_at(a, r) + penalty_at(b, r)
                if after < before - 1e-9:
                    rank_nodes[i], rank_nodes[i + 1] = b, a
                    changed = True
                else:
                    position[a], position[b] = position[b], position[a]  # revert


def _place_level(
    graph_like, remaining_depth, offset_x, offset_y, nodes_out, ports_out, clusters_out, port_anchor, levels_out
) -> tuple[float, float]:
    """Pass 1 of GraphLayout.compute(): positions every node/port of this
    level, recursing into expanded composites, and appends the globally
    positioned NodeLayout/PortLayout/ClusterBox entries to the *_out
    accumulators. Records every visited level in levels_out for pass 2.
    Returns this level's (width, height)."""
    levels_out.append(graph_like)
    level_nodes = graph_like.sorted_nodes
    rank = _rank_nodes(graph_like, level_nodes)
    by_rank: dict[int, list] = {}
    for node in level_nodes:
        by_rank.setdefault(rank[node], []).append(node)
    _reduce_crossings(graph_like, level_nodes, by_rank, rank)

    if not level_nodes:
        return 0.0, 0.0

    # Column x-offsets are cumulative: a rank containing an expanded
    # composite gets the composite's full measured width.
    rank_widths = {
        r: max(_measure_node_width(n, remaining_depth) for n in nodes)
        for r, nodes in by_rank.items()
    }
    rank_x, cursor = {}, offset_x
    for r in sorted(by_rank):
        rank_x[r] = cursor + rank_widths[r] / 2
        cursor += rank_widths[r] + COLUMN_GAP
    total_width = cursor - COLUMN_GAP - offset_x

    # Row height also accounts for the measured height of expanded
    # composites, so nested clusters fit inside their parent.
    row_height = max((_measure_node_height(n, remaining_depth) for n in level_nodes), default=BASE_HEIGHT) + ROW_GAP

    for r, rank_nodes in by_rank.items():
        for i, node in enumerate(rank_nodes):
            x = rank_x[r]
            y = offset_y + i * row_height

            if isinstance(node, GraphNode) and remaining_depth > 0 and node._graph.sorted_nodes:
                _place_cluster(
                    node, x, y, remaining_depth, nodes_out, ports_out, clusters_out, port_anchor, levels_out
                )
                continue

            width, height = _node_size(node)
            show_labels = len(node.input_ports) + len(node.output_ports) > 1
            nodes_out.append(
                NodeLayout(
                    label=_node_label(node),
                    kind=type(node).__name__,
                    category=_node_category(node),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    input_ports=_register_ports(
                        node.input_ports, x - width / 2, y, height, port_anchor, ports_out, show_labels, True
                    ),
                    output_ports=_register_ports(
                        node.output_ports, x + width / 2, y, height, port_anchor, ports_out, show_labels, False
                    ),
                )
            )

    height = max(len(nodes) for nodes in by_rank.values()) * row_height
    return total_width, height


def _measure_node_width(node, remaining_depth) -> float:
    """Width a node occupies when placed, without placing anything. Must
    stay in sync with _place_cluster's width computation."""
    if isinstance(node, GraphNode) and remaining_depth > 0 and node._graph.sorted_nodes:
        gap_in = CLUSTER_GAP if node.input_ports else 0.0
        gap_out = CLUSTER_GAP if node.output_ports else 0.0
        inner_width = _measure_level_width(node._graph, remaining_depth - 1)
        return gap_in + inner_width + gap_out + 2 * CLUSTER_MARGIN
    return _node_size(node)[0]


def _measure_level_width(graph_like, remaining_depth) -> float:
    level_nodes = graph_like.sorted_nodes
    if not level_nodes:
        return 0.0
    rank = _rank_nodes(graph_like, level_nodes)
    by_rank: dict[int, list] = {}
    for node in level_nodes:
        by_rank.setdefault(rank[node], []).append(node)
    total = sum(
        max(_measure_node_width(n, remaining_depth) for n in nodes) + COLUMN_GAP
        for nodes in by_rank.values()
    )
    return total - COLUMN_GAP


def _measure_node_height(node, remaining_depth) -> float:
    """Height a node occupies when placed, without placing anything. Must
    stay in sync with _place_cluster's half_height computation."""
    if isinstance(node, GraphNode) and remaining_depth > 0 and node._graph.sorted_nodes:
        in_height = _port_stack_height(node.input_ports)
        out_height = _port_stack_height(node.output_ports)
        inner_height = _measure_level_height(node._graph, remaining_depth - 1)
        return max(inner_height, in_height, out_height) + 2 * CLUSTER_MARGIN
    return _node_size(node)[1]


def _measure_level_height(graph_like, remaining_depth) -> float:
    level_nodes = graph_like.sorted_nodes
    if not level_nodes:
        return 0.0
    rank = _rank_nodes(graph_like, level_nodes)
    by_rank: dict[int, list] = {}
    for node in level_nodes:
        by_rank.setdefault(rank[node], []).append(node)
    row_height = max(_measure_node_height(n, remaining_depth) for n in level_nodes) + ROW_GAP
    return max(len(nodes) for nodes in by_rank.values()) * row_height


def _build_edges(graph_like, port_anchor, edges_out) -> None:
    """Pass 2 of GraphLayout.compute(): builds every edge of this level
    whose two ports have an anchor. Covers outgoing_edges, the
    edges_from_outside (whose source is the outer composite, so they are
    not in outgoing_edges) and skip_connections."""
    for node in graph_like.sorted_nodes:
        for edge in graph_like.outgoing_edges.get(node, []):
            if edge.from_port not in port_anchor or edge.to_port not in port_anchor:
                continue  # endpoint outside the drawn depth (a collapsed composite's inside)
            edges_out.append(
                EdgeLayout(
                    points=[port_anchor[edge.from_port], port_anchor[edge.to_port]],
                    label=_edge_label(graph_like, node, edge),
                )
            )
    for edge in getattr(graph_like, "edges_from_outside", []):
        if edge.from_port in port_anchor and edge.to_port in port_anchor:
            edges_out.append(
                EdgeLayout(
                    points=[port_anchor[edge.from_port], port_anchor[edge.to_port]],
                    label=_edge_label(graph_like, edge.from_port.node, edge),
                )
            )
    for edge in getattr(graph_like, "skip_connections", []):
        if edge.from_port in port_anchor and edge.to_port in port_anchor:
            edges_out.append(EdgeLayout(points=[port_anchor[edge.from_port], port_anchor[edge.to_port]]))


def _edge_label(graph_like, source_node, edge) -> str:
    config = graph_like.dynamic_data_configs.get(source_node, {}).get(edge.from_port)
    if config is None:
        return ""
    return f"[{', '.join(str(axis) for axis in config.axes)}]"


def _place_cluster(
    node, x, y, remaining_depth, nodes_out, ports_out, clusters_out, port_anchor, levels_out
) -> None:
    """Places a composite node's inner graph inline inside a labeled
    cluster centered at (x, y), with the composite's own ports on the
    cluster boundary (inputs left, outputs right). Edges are built later,
    in pass 2."""
    inner_graph = node._graph
    in_height = _port_stack_height(node.input_ports)
    out_height = _port_stack_height(node.output_ports)

    # The column reserved by _place_level has this composite's measured
    # width centered at x; measure the content first so the cluster is
    # centered at x as well.
    gap_in = CLUSTER_GAP if node.input_ports else 0.0
    gap_out = CLUSTER_GAP if node.output_ports else 0.0
    predicted_inner_width = _measure_level_width(inner_graph, remaining_depth - 1)
    total_width = gap_in + predicted_inner_width + gap_out + 2 * CLUSTER_MARGIN

    cluster_x0 = x - total_width / 2
    cluster_x1 = x + total_width / 2
    inner_offset_x = cluster_x0 + CLUSTER_MARGIN + gap_in

    inner_width, inner_height = _place_level(
        inner_graph, remaining_depth - 1, inner_offset_x, y, nodes_out, ports_out, clusters_out, port_anchor,
        levels_out,
    )

    in_ports = _register_ports(
        node.input_ports, cluster_x0, y, in_height, port_anchor, ports_out, show_labels=True, is_input=True
    )
    out_ports = _register_ports(
        node.output_ports, cluster_x1, y, out_height, port_anchor, ports_out, show_labels=True, is_input=False
    )

    half_height = max(inner_height, in_height, out_height) / 2 + CLUSTER_MARGIN
    clusters_out.append(
        ClusterBox(
            label=f"{node.name} ({type(node).__name__}, {len(inner_graph.sorted_nodes)} nodes)",
            category=_node_category(node),
            x0=cluster_x0,
            y0=y - half_height,
            x1=cluster_x1,
            y1=y + half_height,
            input_ports=in_ports,
            output_ports=out_ports,
        )
    )


def _port_stack_height(ports) -> float:
    if not ports:
        return 0.0
    return max(BASE_HEIGHT, PORT_ROW_HEIGHT * len(ports) + 0.2)
