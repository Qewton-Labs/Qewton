from typing import Annotated

import numpy as np
import pytest

from qewton.backends import TensorType
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import GraphAwareNode, Node
from qewton.visualization.nodes import PlotNode
from qewton.visualization.plots.data.curve import LinePlot
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.spec import ColorSpec


def _constant_source(value, config: DataConfiguration, **kwargs):
    """A minimal real Node whose single output declares `config` - used to
    feed a real, unified DataConfiguration to a PlotNode through normal
    graph connection/unification, not by hand-injecting one."""

    class ConstantSource(Node[TensorType]):
        def forward(self) -> Annotated[TensorType, config]:
            return value

    return ConstantSource(**kwargs)


class TestPlotNodeShape:
    def test_is_graph_aware(self):
        assert isinstance(PlotNode(), GraphAwareNode)

    def test_has_exactly_one_input_port_and_no_output_ports(self):
        """The no-return-annotation subtlety this depends on: `-> None`
        would build one real (always-unused) output port; omitting the
        annotation entirely is what actually yields zero."""
        node = PlotNode()
        assert [p.name for p in node.input_ports] == ["x"]
        assert node.output_ports == []

    def test_setup_stores_the_graph_for_forward_to_use_later(self):
        node = PlotNode()
        assert node._graph is None
        graph = Graph()
        node.setup(graph)
        assert node._graph is graph


class TestPlotNodeInGraph:
    def test_auto_selects_and_draws_via_real_graph_unification(self, small_mesh_geometry, tmp_path):
        """No manual DataConfiguration override - the mesh field's real
        shape reaches PlotNode purely through graph.connect()/setup()."""
        U = Variable("u", 1)
        n = small_mesh_geometry.mesh.vertices.shape[0]
        field = np.random.rand(n, 1)
        config = DataConfiguration(GeometryAxes(small_mesh_geometry), FeatureAxes(U))

        source = _constant_source(field, config, name="Source")
        path = tmp_path / "out.html"
        plot_node = PlotNode(show=False, save_path=str(path), name="Plot")

        graph = Graph()
        graph.add_node(source)
        graph.add_node(plot_node)
        graph.connect(source.output_ports[0], plot_node.input_ports[0])
        graph.setup()

        resolved = plot_node.input_ports[0].get_data_configuration(graph)
        assert resolved.feature_axes is not None  # unification propagated the real shape

        graph.run()
        assert path.exists() and path.stat().st_size > 0

    def test_explicit_plot_type_is_a_pass_through_with_kwargs(self, monkeypatch):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        data = np.random.rand(10, 1)

        source = _constant_source(data, config, name="Source")
        plot_node = PlotNode(plot_type=LinePlot, show=False, x=sample_axis, y=Y, name="Plot")

        captured = {}

        def fake_show(self):
            captured["plot"] = self.plots[0]

        from qewton.visualization.figure import Figure
        monkeypatch.setattr(Figure, "show", fake_show)

        graph = Graph()
        graph.add_node(source)
        graph.add_node(plot_node)
        graph.connect(source.output_ports[0], plot_node.input_ports[0])
        graph.setup()
        plot_node.show = True  # exercise the show() path with the monkeypatched Figure
        graph.run()

        assert isinstance(captured["plot"], LinePlot)
        assert captured["plot"].x.variable_or_axes is sample_axis
        assert captured["plot"].y.variable_or_axes is Y

    def test_incompatible_connection_fails_at_run_time_with_auto_plots_own_error(self):
        """No graph-build-time shape check exists (the port is a wildcard,
        see PlotNode's docstring) - the failure surfaces the first time the
        graph actually runs, via auto_plot's own validation."""
        config = DataConfiguration(BatchAxes(5), BatchAxes(3))  # no FeatureAxes at all
        source = _constant_source(np.zeros((5, 3)), config, name="Source")
        plot_node = PlotNode(show=False, name="Plot")

        graph = Graph()
        graph.add_node(source)
        graph.add_node(plot_node)
        graph.connect(source.output_ports[0], plot_node.input_ports[0])
        graph.setup()

        with pytest.raises(ValueError, match="no FeatureAxes"):
            graph.run()
