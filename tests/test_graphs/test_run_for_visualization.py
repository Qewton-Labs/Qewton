"""Running a graph once for visualization: in mesh mode, reporting progress,
and visualizing its stored values without running again."""

import numpy as np
import pytest

from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.graphs.graphs import Graph
from qewton.visualization import HeatmapPlot
from qewton.visualization.layout import Overlay


@pytest.fixture
def graph_sampler_model():
    X = Variable("x", 2)
    U = Variable("u", 1)
    sampler = GridSampler(Rectangle(X, [0.0, 0.0], 1.0, 1.0), 20)
    model = FCN(in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1)
    graph = Graph()
    graph.connect(sampler, model)
    graph.setup()
    return graph, sampler, model


class TestRunWithDiscretization:
    def test_sends_the_mesh_through_the_graph_and_restores_the_sampler(
        self, graph_sampler_model
    ):
        graph, sampler, model = graph_sampler_model
        graph.run(discretization=0.1)
        meshed = np.asarray(
            sampler.backend.to_numpy(graph.value_of(sampler.output_ports[0]))
        )
        assert sampler.mesh_mode is False
        graph.run()
        sampled = np.asarray(
            sampler.backend.to_numpy(graph.value_of(sampler.output_ports[0]))
        )
        # The mesh of the unit square at 0.1 has more points than the 20 sampled.
        assert meshed.shape[0] > sampled.shape[0] == 20
        assert graph.value_of(model.output_ports[0]) is not None

    def test_the_same_discretization_gives_the_same_points(self, graph_sampler_model):
        graph, sampler, _ = graph_sampler_model
        graph.run(discretization=0.1)
        first = np.asarray(
            sampler.backend.to_numpy(graph.value_of(sampler.output_ports[0]))
        )
        graph.run(discretization=0.1)
        second = np.asarray(
            sampler.backend.to_numpy(graph.value_of(sampler.output_ports[0]))
        )
        np.testing.assert_array_equal(first, second)


class TestRunProgress:
    def test_reports_each_node_once_when_it_is_done_in_execution_order(
        self, graph_sampler_model
    ):
        graph, sampler, model = graph_sampler_model
        done = []
        graph.run(on_node_done=done.append)
        assert done == [sampler, model]

    def test_reports_a_node_after_its_value_is_there(self, graph_sampler_model):
        graph, sampler, model = graph_sampler_model
        seen = []
        graph.run(
            on_node_done=lambda node: seen.append(
                graph.value_of(node.output_ports[0]) is not None
            )
        )
        assert seen == [True, True]


class TestVisualizeWithoutRunning:
    def test_draws_the_values_of_the_last_run(self, graph_sampler_model, monkeypatch):
        graph, _, model = graph_sampler_model
        graph.run(discretization=0.05)
        calls = []
        original = FCN.__call__
        monkeypatch.setattr(
            FCN,
            "__call__",
            lambda self, *a, **k: calls.append(1) or original(self, *a, **k),
        )
        layout = graph.visualize(model.output_ports[0], run=False)
        assert calls == []
        assert isinstance(layout, Overlay)
        assert isinstance(layout.plots[0], HeatmapPlot)

    def test_matches_visualizing_with_a_run_at_the_same_discretization(
        self, graph_sampler_model
    ):
        graph, _, model = graph_sampler_model
        graph.run(discretization=0.05)
        stored = graph.visualize(model.output_ports[0], run=False)
        fresh = graph.visualize(model.output_ports[0], max_vertex_distance=0.05)
        assert type(stored.plots[0]) is type(fresh.plots[0])

    def test_refuses_a_reference_without_running(self, graph_sampler_model):
        graph, _, model = graph_sampler_model
        with pytest.raises(ValueError, match="run"):
            graph.visualize(model.output_ports[0], reference=lambda x: x, run=False)


class TestMeshModeSizes:
    def test_switching_to_mesh_mode_sizes_the_configurations_before_running(
        self, graph_sampler_model
    ):
        graph, sampler, model = graph_sampler_model

        def points():
            # The configuration object is updated in place: read its size now.
            return model.output_ports[0].get_data_configuration(graph).shape[0]

        before = points()
        sampler.set_mesh_mode(0.1)
        meshed = points()
        sampler.unset_mesh_mode()
        assert before == points() == 20
        assert meshed > 20
