import importlib.util

import pytest

from qewton.visualization.figure import Figure
from qewton.visualization.layout import Overlay, Row
from qewton.visualization.plots.graph import GraphPlot
from qewton.visualization.plots.graph.layout import GraphLayout
from qewton.visualization.plots.result import NodeCategory
from qewton.visualization.renderers.plotly.graph import NodeLinkArtist
from qewton.visualization.themes import DARK_THEME, LIGHT_THEME


#: Drawing the cards of a graph needs the optional shapely package.
needs_shapely = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="needs shapely"
)


class TestGraphPlot:
    def test_a_graph_has_no_controls_and_one_result(self, simple_graph):
        plot = GraphPlot(simple_graph, show_port_names=True)
        assert plot.controls == []
        (result,) = plot.evaluate()
        assert result.show_port_names and result.show_edge_labels

    @needs_shapely
    def test_a_graph_is_drawn_in_a_2d_cell_with_hidden_axes(self, simple_graph):
        backend_figure = Figure(GraphPlot(simple_graph)).draw()
        assert backend_figure.layout.xaxis.visible is False
        assert backend_figure.layout.yaxis.visible is False

    @needs_shapely
    def test_draws_exactly_three_traces_regardless_of_depth(self, simple_graph):
        for depth in (0, 1, 2):
            backend_figure = Figure(GraphPlot(simple_graph, depth=depth)).draw()
            assert len(backend_figure.data) == 3  # node-hover, port, edge

    @needs_shapely
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

    @needs_shapely
    def test_two_graphplots_in_one_figure_do_not_clobber_each_others_shapes(self, simple_graph):
        fig = Figure(Row(GraphPlot(simple_graph, depth=0), GraphPlot(simple_graph, depth=1)))
        backend_figure = fig.draw()
        assert len(backend_figure.data) == 6  # 3 traces each
        fig.draw()
        assert len(backend_figure.data) == 6

    @needs_shapely
    def test_shapes_and_texts_lie_in_the_axes_of_their_cell(self, simple_graph):
        fig = Figure(Row(GraphPlot(simple_graph), GraphPlot(simple_graph)))
        backend_figure = fig.draw()
        refs = {shape.xref for shape in backend_figure.layout.shapes}
        assert refs == {"x", "x2"}

    @pytest.mark.parametrize("theme", [LIGHT_THEME, DARK_THEME])
    def test_node_categories_are_visually_distinct(self, theme):
        """Constraint, data and model cards must be told apart at a glance -
        whatever the theme's actual colors are."""
        fills = {
            category: NodeLinkArtist._node_colors(category, theme)[0]
            for category in NodeCategory
        }
        assert len(set(fills.values())) == 3

    @needs_shapely
    @pytest.mark.parametrize("depth", [0, 1])
    def test_every_port_is_hoverable_with_its_name_and_direction(self, simple_graph, depth):
        layout = GraphLayout.compute(simple_graph, depth=depth)
        backend_figure = Figure(GraphPlot(simple_graph, depth=depth)).draw()
        port_trace = backend_figure.data[1]
        assert len(port_trace.x) == len(layout.ports) > 0
        assert all(
            text.endswith("(in)") or text.endswith("(out)") for text in port_trace.hovertext
        )

    @needs_shapely
    def test_hover_text_includes_kind(self, simple_graph):
        backend_figure = Figure(GraphPlot(simple_graph, depth=0)).draw()
        node_trace = backend_figure.data[0]
        assert any("FCN" in text for text in node_trace.hovertext)


def test_a_graph_is_one_artist_in_a_figure(simple_graph):
    from recording_renderer import RecordingRenderer

    backend = Figure(GraphPlot(simple_graph), renderer=RecordingRenderer()).draw()
    (artist,) = backend["artists"]
    assert artist.result.nodes
