from qewton.constraints.base import Constraint
from qewton.data.dataloaders.base import DataNode
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.visualization.plots.result import ClusterBox, EdgeLayout, GraphLayoutResult, NodeLayout, PortLayout

#: Layout constants - "units" are arbitrary but self-consistent (node height
#: is derived from label length in the same units used for spacing), not
#: literal pixels. Tunable if real graphs render too cramped/sparse.
#: Node width is now fixed, not label-driven (see _node_size) - the card's
#: title renders rotated 90 degrees (see NodeLinkArtist._card_annotations),
#: so it's the node's HEIGHT that needs to grow with label length instead,
#: keeping nodes/columns narrow even for a long chain of nodes with varied
#: name lengths - exactly what point 7 of Nick's correction list asked for.
NODE_WIDTH = 1.0
#: Scaled down from the old 1.4 to keep the same ratio to node width as
#: before (COLUMN_GAP was ~0.93x MIN_WIDTH=1.5) - at the old absolute value
#: a column gap would now be wider than a node itself, exactly the "too
#: wide" look point 7 asked to avoid.
COLUMN_GAP = 0.9
ROW_GAP = 0.6
PORT_ROW_HEIGHT = 0.35
BASE_HEIGHT = 0.55
#: Per-character height contribution for the now-vertical title, the
#: rotated-90-degrees counterpart of the old (horizontal-title) CHAR_WIDTH.
CHAR_HEIGHT = 0.1
LABEL_PADDING = 0.4
CLUSTER_MARGIN = 0.5
CLUSTER_GAP = 0.8
BARYCENTER_PASSES = 4
#: Row-distance (in the same "how many rows apart" units as a rank's own
#: node ordering) within which a "long" edge (one skipping at least one
#: intermediate rank entirely) is considered to visually cut through a row
#: it passes near - see _reduce_node_crossings. 1.0 means "anything closer
#: than one full row spacing away" gets penalized, tapering to 0 at exactly
#: that distance.
NODE_CROSSING_RADIUS = 1.0


class GraphLayout:
    """Computes node/edge positions for a Graph - the layout counterpart to
    `evaluate()` for other plot families: renderer-agnostic, and reusable
    unchanged by a future non-Plotly renderer since it makes no drawing
    decisions at all.

    A pure layered-DAG layout: rank = longest path from a source (computed
    in one pass over Graph.sorted_nodes's topological order), then a few
    barycenter passes reorder each rank to reduce edge crossings - no
    external layout engine, so no new dependency. Considered and dropped:
    using graphviz's `dot` purely for layout (not rendering) - `dot` is a
    system binary, not `pip install`-able alone, a meaningfully bigger ask
    than every other optional dependency in this codebase (kaleido, h5py,
    optuna), for graphs that are typically fairly linear chains with
    occasional composite clusters, not the dense hairballs that would
    actually need dot's more sophisticated crossing minimization.

    Two passes, not one: (1) place every node/port/cluster, recursing into
    expanded composites, accumulating every level's ports into one shared
    port_anchor map; (2) only then build every edge, at every level
    (including a composite's own edges_from_outside/edges_to_outside -
    exactly the edges GraphNode.__init__ creates to map a composite's ports
    onto its inner graph's boundary nodes). Positions have to exist on both
    sides of an edge before it can be drawn, and a composite's own ports are
    positioned as part of *its parent's* level, so building edges level by
    level during placement would need each level's boundary ports placed
    before its own content - exactly backwards from how placement naturally
    recurses (content's width determines where its parent places the
    interface boxes around it). Splitting into two passes avoids that
    ordering fight entirely.
    """

    @staticmethod
    def compute(graph, depth: int = 0) -> GraphLayoutResult:
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
    """Coloring bucket - broader than `kind` (the exact class name, kept
    for hover text), so unrelated node types sharing a role read as the
    same color instead of each needing its own hardcoded entry. Order
    matters: MSEConstraint/PINNConstraint are simultaneously Constraint
    *and* GraphNode (they wrap an inner graph to compute their residual),
    so Constraint must be checked first or they'd read as plain graphnodes."""
    if isinstance(node, Constraint):
        return "constraint"
    if isinstance(node, DataNode):
        return "datanode"
    if isinstance(node, GraphNode):
        return "graphnode"
    return "default"


def _node_size(node) -> tuple[float, float]:
    """Width is a fixed constant (NODE_WIDTH) - the title no longer grows it,
    since it's drawn rotated 90 degrees (see NodeLinkArtist._card_annotations)
    and so needs HEIGHT, not width, to fit a long name. Height is the max of
    three independent floors: the base minimum, enough rows for the node's
    own ports, and enough length for its (now vertical) title."""
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
    right edge for outputs), records each port's anchor in the shared
    port_anchor map (the only thing edge-building needs, at any level), and
    appends each one to the flat ports_out accumulator (what NodeLinkArtist
    actually draws circles/labels from - see GraphLayoutResult.ports)."""
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
    """Longest path from a source, one pass over the already-topologically-
    sorted level_nodes - predecessors are guaranteed to have a final rank
    already when a node is processed. Edges crossing this level's boundary
    (from Graph.connect_from_outside_of_graph) are naturally excluded, since
    their source isn't in this level's node set - exactly right, a
    boundary-fed node is a source within its own level."""
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
    """A handful of barycenter passes (Sugiyama's classic heuristic):
    reorder each rank by the mean position of its neighbors in the adjacent
    rank, alternating which neighbor direction drives the sweep. Then one
    more pass specifically targets a different problem barycenter alone
    never touches - see _reduce_node_crossings. Mutates by_rank's lists in
    place."""
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

    # A "long" edge - one whose rank gap is more than 1, i.e. it skips at
    # least one intermediate rank entirely rather than connecting adjacent
    # columns - never shares an endpoint with any node in a rank it merely
    # passes through, so plain barycenter ordering (which only ever pulls a
    # node toward ITS OWN neighbors) has no way to even notice that edge,
    # let alone route around whatever ends up sitting in that column at the
    # row its straight line happens to cross. _reduce_node_crossings is a
    # separate, deliberately small refinement pass purely for that case.
    def _related(a, b) -> frozenset:
        """Every node directly adjacent (either direction) to `a` or `b`,
        `a`/`b` themselves included - the set this long edge's presence
        near a row is entirely unsurprising for, since they're already
        visually tied to one of its own endpoints by a real edge of their
        own. Excluded from the node-crossing penalty below for exactly that
        reason: a bystander with NO connection to either endpoint ending up
        in this edge's way is the actual problem worth nudging away from,
        not a node that was always going to cluster near here anyway."""
        related = {a, b}
        for n in (a, b):
            related.update(p for e in graph_like.outgoing_edges.get(n, []) if (p := e.to_port.node) in node_set)
            related.update(p for e in graph_like.incoming_edges.get(n, []) if (p := e.from_port.node) in node_set)
        return frozenset(related)

    def _candidate_edges():
        # Regular edges - `node_set`-filtered here too (not just in the
        # skip_connections loop below): outgoing_edges can point at a node
        # OUTSIDE this level entirely (a boundary edge leaving to the
        # composite wrapper itself, rather than one of its own already-
        # placed inner nodes - see _build_edges's own comment on exactly
        # this), which `rank` has no entry for at all. The original,
        # pre-refactor one-line version of this filtered via `in node_set
        # and rank[...] - ...` in the same `if`, short-circuiting before
        # ever indexing `rank` - splitting the edge collection out into
        # this generator dropped that guard by accident, which is exactly
        # what turned into a bare KeyError on a real (depth=1) graph.
        for node in level_nodes:
            for edge in graph_like.outgoing_edges.get(node, []):
                if edge.to_port.node in node_set:
                    yield node, edge.to_port.node
        # ...and skip_connections (see _build_edges) - a ResNet-style skip
        # is, structurally, exactly the case this whole pass exists for: it
        # explicitly jumps past one or more ranks by design, yet _rank_nodes
        # never counts it (a node's rank only ever follows its "regular"
        # dependency edges), so without this it wouldn't show up as a long
        # edge at all despite being the single most likely one to visually
        # cut across an unrelated node sitting in between.
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
    """Nudges a rank's own node ordering to also dodge any "long" edge that
    passes straight through that rank on its way from an earlier one to a
    later one (see _reduce_crossings) - a cost term entirely separate from
    (and additive to, not a replacement for) ordinary edge-crossing
    minimization, since the two can conflict: the best ordering for
    minimizing crossings between ADJACENT ranks is not necessarily the one
    that keeps a long edge, several ranks away from either of its own
    endpoints, clear of an unrelated node.

    For each intermediate rank `r` a long edge spans, that edge's own
    straight-line path implies a "virtual row" at `r` - the linear
    interpolation, between the edge's two (rank, row) endpoints, of its row
    at `r` - using the same row-index units `_reduce_crossings`'s own
    `position` dict already tracks (valid because every rank within one
    level shares the same row_height/offset_y - see _place_level - so row
    index alone, with no y-coordinate math, is already directly comparable
    across ranks). A rank's own node sitting within NODE_CROSSING_RADIUS
    rows of that virtual row reads, once actually drawn, as the edge
    visually cutting across that node's card - UNLESS it's one of that
    edge's own `related` nodes (see _related above), which pay no penalty
    at all: without that exemption, a node that's only "in the way" because
    it's genuinely wired to one of the edge's own endpoints could get
    shuffled around for no visual benefit, and - worse - in a rank where
    every node happens to be one of a handful of mutually-interchangeable
    bystanders, the total penalty is identical under ANY permutation (one
    of them always ends up in the one bad slot), so a swap search blind to
    "related" never finds a reason to prefer putting the genuinely-expected
    node there instead of an unconnected one.

    Only adjacent-swap exchanges are tried (a simple greedy local search,
    repeated to a fixed point) - cheap, and sufficient: a long edge is the
    exception in a typical qewton graph (mostly linear chains plus
    occasional composites), not the common case, so this only ever has a
    handful of candidate ranks/swaps to actually consider."""
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
    """Pass 1 (see GraphLayout.compute()): positions every node/port at this
    level - recursing into expanded composites - and appends already-
    globally-positioned NodeLayout/PortLayout/ClusterBox entries to the
    *_out accumulators. Records `graph_like` (and every inner graph it
    expands into) in levels_out, so pass 2 knows every level whose edges
    need building. Returns this level's own (width, height) bounding box, so
    a caller expanding a composite can size the cluster outline placed
    around it."""
    levels_out.append(graph_like)
    level_nodes = graph_like.sorted_nodes
    rank = _rank_nodes(graph_like, level_nodes)
    by_rank: dict[int, list] = {}
    for node in level_nodes:
        by_rank.setdefault(rank[node], []).append(node)
    _reduce_crossings(graph_like, level_nodes, by_rank, rank)

    if not level_nodes:
        return 0.0, 0.0

    # Column x-offsets are cumulative, not evenly spaced: a rank containing
    # an expanded composite needs its true (possibly much larger) expanded
    # width counted here, via _measure_node_width's dry-run of the same
    # recursive sizing _place_cluster performs for real below - otherwise a
    # wide cluster would overlap whatever's in the neighboring column.
    rank_widths = {
        r: max(_measure_node_width(n, remaining_depth) for n in nodes)
        for r, nodes in by_rank.items()
    }
    rank_x, cursor = {}, offset_x
    for r in sorted(by_rank):
        rank_x[r] = cursor + rank_widths[r] / 2
        cursor += rank_widths[r] + COLUMN_GAP
    total_width = cursor - COLUMN_GAP - offset_x

    # Same reasoning as rank_widths above, for the vertical direction: a
    # node about to expand into a cluster needs its true expanded height
    # counted here too, or the outer level's own returned height (used by
    # _place_cluster's half_height for *its* parent, if nested) understates
    # how tall this level actually is - an outer cluster would then draw
    # too short to contain a taller nested one.
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
    """Dry-run companion to _place_level/_place_cluster: computes the same
    width without placing anything, so a rank's column can be sized to fit
    an expanded composite *before* that composite is actually placed. Must
    stay structurally in sync with _place_cluster's real math below. Ports
    themselves add no width - they sit directly on the cluster boundary,
    not in their own box - only the gap to the first/last inner column
    does."""
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
    """Dry-run companion to _place_level/_place_cluster, mirroring
    _measure_node_width but for the vertical direction: computes the same
    height an expanded composite would actually need, without placing
    anything, so row spacing can be reserved correctly *before* a node in
    that row is placed. Must stay structurally in sync with _place_cluster's
    real half_height math."""
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
    """Pass 2 (see GraphLayout.compute()): every port_anchor entry needed by
    any edge at this level - including boundary edges crossing into/out of
    it - already exists by the time this runs for any level, since pass 1
    completes fully (parent and every descendant) before pass 2 starts.

    Walking outgoing_edges alone misses half the boundary edges a composite
    creates: connect_to_outside_of_graph's edge has from_port on an *inner*
    node, so it lands in outgoing_edges[inner_node] and gets found the
    normal way - but connect_from_outside_of_graph's edge has from_port on
    the *outer* composite itself, which is never a member of this graph's
    own node set at all, so it never appears in outgoing_edges here
    (Graph.connect_from_outside_of_graph only appends it to
    edges_from_outside and incoming_edges[to_node]). Iterating
    edges_from_outside explicitly catches exactly those - and only those:
    everything else reachable from edges_from_outside is already covered by
    the outgoing_edges loop, so this can't double-draw an edge.
    """
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
    """Replaces a composite node's single box with its inner graph, drawn
    inline inside a labeled cluster outline. The composite's own ports have
    no box of their own - they sit directly on the cluster boundary (left
    edge for inputs, right edge for outputs), the same as any other port
    sits on its owning box's edge, just with the "box" being the whole
    cluster rectangle. Only positions things - edges (including the ones
    crossing this cluster's boundary) are all built later, in pass 2.
    """
    inner_graph = node._graph
    in_height = _port_stack_height(node.input_ports)
    out_height = _port_stack_height(node.output_ports)

    # The top-level column allocator (_place_level, via _measure_node_width)
    # already reserved a column of exactly this composite's *measured*
    # total width, centered at x - so the cluster must actually be drawn
    # centered at x too, or it drifts into whatever's in the next column.
    # Pre-measuring the inner content's width lets the cluster boundary be
    # placed symmetrically around x up front, in one pass - no separate
    # reconciliation needed after the real (possibly nested) recursive
    # placement runs.
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
