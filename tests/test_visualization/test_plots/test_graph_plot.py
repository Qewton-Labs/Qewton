from qewton.visualization.figure import Figure
from qewton.visualization.layout import Overlay
from qewton.visualization.plots.graph import GraphPlot
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

    def test_node_fill_color_reflects_category(self, simple_graph):
        backend_figure = Figure(GraphPlot(simple_graph, depth=0), theme=LIGHT_THEME).draw()
        node_shapes = [s for s in backend_figure.layout.shapes if s.fillcolor != "rgba(0,0,0,0)"]
        colors = {s.fillcolor for s in node_shapes}
        # fcn (graphnode) and Source/Loss (default) must differ
        assert LIGHT_THEME.node_color_by_type["graphnode"] in colors
        assert LIGHT_THEME.node_color_default in colors

    def test_ports_fill_with_the_theme_background_not_a_fixed_white(self, simple_graph):
        light = Figure(GraphPlot(simple_graph), theme=LIGHT_THEME).draw()
        dark = Figure(GraphPlot(simple_graph), theme=DARK_THEME).draw()
        port_trace_light = light.data[1]
        port_trace_dark = dark.data[1]
        assert port_trace_light.marker.color == LIGHT_THEME.background_color
        assert port_trace_dark.marker.color == DARK_THEME.background_color

    def test_hover_text_includes_kind(self, simple_graph):
        backend_figure = Figure(GraphPlot(simple_graph, depth=0)).draw()
        node_trace = backend_figure.data[0]
        assert any("FCN" in text for text in node_trace.hovertext)
