import numpy as np
import pytest

from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.graphs.graphs import Graph
from qewton.visualization.auto import auto_plot
from qewton.visualization.figure import Figure
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.data.mesh import MeshSurfacePlot
from qewton.visualization.plots.data.points import PointCloudPlot


@pytest.fixture
def sampler_and_model():
    X = Variable("x", 2)
    U = Variable("u", 1)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    sampler = GridSampler(square, 20)
    model = FCN(
        in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1,
    )
    return sampler, model


@pytest.fixture
def connected_graph(sampler_and_model):
    sampler, model = sampler_and_model
    graph = Graph()
    graph.connect(sampler, model)
    graph.setup()
    return graph, sampler, model


class TestVisualize:
    def test_switches_the_sampler_into_mesh_mode_for_the_run_and_back(self, connected_graph):
        graph, sampler, model = connected_graph
        assert sampler.mesh_mode is False

        graph.visualize(model.output_ports[0])

        assert sampler.mesh_mode is False  # restored after the run

    def test_produces_a_mesh_surface_plot_from_mesh_mode_output(self, connected_graph):
        """visualize(single_port) returns a bare Plot, not a 1-element
        list - only multiple requested ports come back as a list."""
        graph, sampler, model = connected_graph
        plot = graph.visualize(model.output_ports[0])
        assert isinstance(plot, MeshSurfacePlot)

    def test_the_resulting_plot_draws_without_error(self, connected_graph):
        graph, sampler, model = connected_graph
        plots = graph.visualize(model.output_ports[0])
        backend_figure = Figure(plots).draw()
        assert len(backend_figure.data) > 0

    def test_output_is_detached_even_though_the_model_requires_grad(self, connected_graph):
        """The model's parameters require grad, so its output does too -
        visualize() must hand auto_plot()/Plot.evaluate() already-detached
        data, the same as every other caller does by convention."""
        graph, sampler, model = connected_graph
        plot = graph.visualize(model.output_ports[0])
        result = plot.evaluate()
        assert isinstance(np.asarray(result.color), np.ndarray)

    def test_raw_sampling_without_mesh_mode_yields_a_point_cloud_plot(self, connected_graph):
        """Outside of visualize()'s own mesh-mode run, the same graph's
        normal run() produces a plain, unstructured batch - auto_plot then
        falls back to PointCloudPlot instead of MeshFieldPlot."""
        graph, sampler, model = connected_graph
        graph.run()
        config = model.output_ports[0].get_data_configuration(graph)
        value = model.backend.to_numpy(model.output_ports[0].value)
        plot = auto_plot(value, config)
        assert isinstance(plot, PointCloudPlot)

    def test_samplers_on_path_to_finds_the_grid_sampler(self, connected_graph):
        graph, sampler, model = connected_graph
        found = graph._samplers_on_path_to((model.output_ports[0],))
        assert found == [sampler]

    def test_visualizing_the_samplers_own_output_port_works_too(self, connected_graph):
        """Not just a downstream node's port - the sampler's own output
        port is itself a valid target."""
        graph, sampler, model = connected_graph
        plot = graph.visualize(sampler.output_ports[0])
        assert isinstance(plot, Plot)

    def test_multiple_ports_still_return_a_list(self, connected_graph):
        graph, sampler, model = connected_graph
        plots = graph.visualize(model.output_ports[0], sampler.output_ports[0])
        assert isinstance(plots, list)
        assert len(plots) == 2

    def test_sampled_geometry_is_plain_numpy_after_visualize(self, connected_graph):
        """The model/sampler can stay on whatever device they were trained
        on the whole time - visualize() moves only the data a Plot actually
        reads, right before handing it over, not the graph itself."""
        graph, sampler, model = connected_graph
        graph.visualize(model.output_ports[0])
        assert isinstance(sampler.sampled_geometry.discretization_points, np.ndarray)


class TestVisualizeDevice:
    def test_an_explicit_device_moves_every_node_needed_for_the_run(self, connected_graph):
        graph, sampler, model = connected_graph
        graph.visualize(model.output_ports[0], device="cpu")
        assert sampler._device == "cpu"

    def test_device_is_forwarded_into_mesh_mode_sampling(self, connected_graph):
        """Geometry.create_mesh()'s own device otherwise defaults to cpu
        regardless of the sampler's actual device - an explicit device=
        must reach mesh-mode's point generation too, not just move nodes."""
        graph, sampler, model = connected_graph
        original_visualization_mesh = sampler.sampled_geometry.visualization_mesh
        captured = {}

        def spy(max_vertex_distance, device=None):
            captured["device"] = device
            return original_visualization_mesh(max_vertex_distance, device)

        sampler.sampled_geometry.visualization_mesh = spy
        graph.visualize(model.output_ports[0], device="cpu")
        assert captured["device"] == "cpu"

    def test_omitting_device_leaves_nodes_untouched(self, connected_graph):
        graph, sampler, model = connected_graph
        original_device = sampler._device
        graph.visualize(model.output_ports[0])
        assert sampler._device == original_device
