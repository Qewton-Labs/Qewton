import pytest

from qewton.graphs.graphs import Graph
from qewton.visualization.plots.graph.layout import GraphLayout


def _box_overlaps(a, b) -> bool:
    al, ar, ab, at = a.x - a.width / 2, a.x + a.width / 2, a.y - a.height / 2, a.y + a.height / 2
    bl, br, bb, bt = b.x - b.width / 2, b.x + b.width / 2, b.y - b.height / 2, b.y + b.height / 2
    return al < br - 1e-9 and bl < ar - 1e-9 and ab < bt - 1e-9 and bb < at - 1e-9


class TestGraphLayoutCompute:
    def test_requires_a_sorted_graph(self):
        with pytest.raises(AssertionError, match="sorted"):
            GraphLayout.compute(Graph())

    def test_collapsed_composite_is_a_single_box_per_node(self, simple_graph):
        result = GraphLayout.compute(simple_graph, depth=0)
        assert len(result.nodes) == 3  # Source, fcn, Loss
        assert len(result.clusters) == 0
        labels = {n.label for n in result.nodes}
        assert labels == {"Source", "fcn", "Loss"}

    def test_expanded_composite_produces_a_cluster_and_no_interface_boxes(self, simple_graph):
        """Regression: composite ports used to render as separate small
        boxes labeled 'fcn (in)'/'fcn (out)' - now they're just circles on
        the cluster's own boundary, no extra NodeLayout at all."""
        result = GraphLayout.compute(simple_graph, depth=1)
        assert len(result.clusters) == 1
        assert not any("(in)" in n.label or "(out)" in n.label for n in result.nodes)
        # inner FCN chain: linear, ReLU, linear, ReLU, linear
        inner_labels = [n.label for n in result.nodes if n.label != "Source" and n.label != "Loss"]
        assert inner_labels.count("linear") == 3
        assert inner_labels.count("ReLU") == 2

    @pytest.mark.parametrize("depth", [0, 1, 2, 3])
    def test_no_node_boxes_overlap_at_any_depth(self, simple_graph, depth):
        result = GraphLayout.compute(simple_graph, depth=depth)
        nodes = result.nodes
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                assert not _box_overlaps(a, b), f"{a.label!r} overlaps {b.label!r} at depth={depth}"

    def test_expanded_cluster_fully_contains_its_nested_clusters(self, simple_graph):
        """Regression: a rank's height used to come from leaf-only sizing
        even when that rank's node was about to expand into a much taller
        nested cluster - the outer cluster then didn't vertically contain
        what was nested inside it."""
        result = GraphLayout.compute(simple_graph, depth=2)
        clusters = sorted(result.clusters, key=lambda c: (c.x1 - c.x0) * (c.y1 - c.y0), reverse=True)
        outer = clusters[0]
        for inner in clusters[1:]:
            if outer.x0 <= inner.x0 and inner.x1 <= outer.x1:  # inner belongs to outer's column
                assert outer.y0 <= inner.y0
                assert inner.y1 <= outer.y1

    def test_boundary_edges_into_and_out_of_an_expanded_composite_are_not_dropped(self, simple_graph):
        """Regression: Graph.connect_from_outside_of_graph's edge has
        from_port on the *outer* composite, which is never a member of the
        inner graph's own outgoing_edges - a naive outgoing_edges-only walk
        silently drops it. Only connect_to_outside_of_graph's edge (from_port
        on an *inner* node) was found the naive way."""
        collapsed = GraphLayout.compute(simple_graph, depth=0)
        expanded = GraphLayout.compute(simple_graph, depth=1)
        # collapsed: Source->fcn, fcn->Loss (2 edges)
        assert len(collapsed.edges) == 2
        # expanded: Source->fcn(in), fcn(in)->linear0, 4 internal chain edges,
        # linear4->fcn(out), fcn(out)->Loss = 8
        assert len(expanded.edges) == 8

    def test_edges_carry_the_dataconfiguration_axes_label(self, simple_graph):
        result = GraphLayout.compute(simple_graph, depth=0)
        assert any(edge.label for edge in result.edges)

    def test_ports_are_flat_across_leaf_and_cluster_boundary_alike(self, simple_graph):
        collapsed = GraphLayout.compute(simple_graph, depth=0)
        expanded = GraphLayout.compute(simple_graph, depth=1)
        # every leaf node's ports show up in the flat list too
        assert len(collapsed.ports) >= len(collapsed.nodes)
        # the composite's own 2 ports (in fcn) still appear once expanded,
        # now anchored on the cluster boundary rather than an owning node
        assert len(expanded.ports) > len(collapsed.ports)


class TestNodeCategory:
    def test_generic_leaf_nodes_get_the_default_category(self, simple_graph):
        result = GraphLayout.compute(simple_graph, depth=0)
        source = next(n for n in result.nodes if n.label == "Source")
        assert source.category == "default"

    def test_graphnode_composites_get_the_graphnode_category(self, simple_graph):
        result = GraphLayout.compute(simple_graph, depth=0)
        fcn = next(n for n in result.nodes if n.label == "fcn")
        assert fcn.category == "graphnode"

    def test_nested_graphnode_leaves_are_still_categorized_correctly(self, simple_graph):
        """FunctionalLinear (inside an expanded Linear, inside FCN) must be
        recognized via isinstance(GraphNode), not by name matching."""
        result = GraphLayout.compute(simple_graph, depth=2)
        functional_linear = next(n for n in result.nodes if n.label == "functional_linear")
        assert functional_linear.category == "graphnode"
