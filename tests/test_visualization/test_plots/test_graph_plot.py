import pytest

from qewton.visualization.figure import Figure
from qewton.visualization.layout import Overlay
from qewton.visualization.plots.graph import GraphPlot
from qewton.visualization.plots.graph.layout import GraphLayout
from qewton.visualization.renderers.plotly.graph import NodeLinkArtist
from qewton.visualization.themes import DARK_THEME, LIGHT_THEME


class TestGraphPlot:
    def test_no_specs_and_no_controls(self, simple_graph):
        """A computation graph isn't a value source mapped to a role - this
        family deliberately has neither, unlike every DataPlot/TablePlot."""
        plot = GraphPlot(simple_graph)
        assert plot.controls == []
        assert not hasattr(plot, "color")

    def test_embedding_dim_is_none(self, simple_graph):
        assert GraphPlot(simple_graph).embedding_dim is None

    def test_draws_exactly_three_traces_regardless_of_depth(self, simple_graph):
        for depth in (0, 1, 2):
            backend_figure = Figure(GraphPlot(simple_graph, depth=depth)).draw()
            assert len(backend_figure.data) == 3  # node-hover, port, edge

    def test_redraw_does_not_duplicate_traces_shapes_or_annotations(self, simple_graph):
        fig = Figure(GraphPlot(simple_graph, depth=1))
        backend_figure = fig.draw()
        n_traces = len(backend_figure.data)
        n_shapes = len(backend_figure.layout.shapes)
        n_annotations = len(backend_figure.layout.annotations)
        fig.draw()
        assert len(backend_figure.data) == n_traces
        assert len(backend_figure.layout.shapes) == n_shapes
        assert len(backend_figure.layout.annotations) == n_annotations

    def test_two_graphplots_in_one_figure_do_not_clobber_each_others_shapes(self, simple_graph):
        fig = Figure(Overlay(GraphPlot(simple_graph, depth=0), GraphPlot(simple_graph, depth=1)))
        backend_figure = fig.draw()
        assert len(backend_figure.data) == 6  # 3 traces each
        fig.draw()
        assert len(backend_figure.data) == 6

    @pytest.mark.parametrize("theme", [LIGHT_THEME, DARK_THEME])
    def test_node_categories_are_visually_distinct(self, theme):
        """Constraint, data and model cards must be told apart at a glance -
        whatever the theme's actual colors are."""
        fills = {
            category: NodeLinkArtist._node_colors(category, theme)[0]
            for category in ("constraint", "datanode", "graphnode")
        }
        assert len(set(fills.values())) == 3

    @pytest.mark.parametrize("depth", [0, 1])
    def test_every_port_is_hoverable_with_its_name_and_direction(self, simple_graph, depth):
        layout = GraphLayout.compute(simple_graph, depth=depth)
        backend_figure = Figure(GraphPlot(simple_graph, depth=depth)).draw()
        port_trace = backend_figure.data[1]
        assert len(port_trace.x) == len(layout.ports) > 0
        assert all(
            text.endswith("(in)") or text.endswith("(out)") for text in port_trace.hovertext
        )

    def test_hover_text_includes_kind(self, simple_graph):
        backend_figure = Figure(GraphPlot(simple_graph, depth=0)).draw()
        node_trace = backend_figure.data[0]
        assert any("FCN" in text for text in node_trace.hovertext)
