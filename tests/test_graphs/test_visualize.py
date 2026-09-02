import numpy as np
import pytest

from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.graphs.graphs import Graph
from qewton.optim.base import EvaluationPhase
from qewton.visualization.auto import auto_plot
from qewton.visualization.figure import Figure
from qewton.visualization.plots.base import Plot
from qewton.visualization.layout import Overlay, Row
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.data.points import PointCloudPlot


@pytest.fixture
def sampler_and_model():
    X = Variable("x", 2)
    U = Variable("u", 1)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    sampler = GridSampler(square, 20)
    model = FCN(
        in_neurons=X,
        hidden_neurons=4,
        out_neurons=U,
        n_hidden_layers=1,
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
    def test_switches_the_sampler_into_mesh_mode_for_the_run_and_back(
        self, connected_graph
    ):
        graph, sampler, model = connected_graph
        assert sampler.mesh_mode is False

        graph.visualize(model.output_ports[0])

        assert sampler.mesh_mode is False  # restored after the run

    def test_produces_a_mesh_field_plot_from_mesh_mode_output(self, connected_graph):
        """visualize(single_port) returns Overlay(plot), never a bare Plot
        or a list."""
        graph, sampler, model = connected_graph
        layout = graph.visualize(model.output_ports[0])
        assert isinstance(layout, Overlay)
        assert isinstance(layout.plots[0], MeshFieldPlot)

    def test_the_resulting_plot_draws_without_error(self, connected_graph):
        graph, sampler, model = connected_graph
        plots = graph.visualize(model.output_ports[0])
        backend_figure = Figure(plots).draw()
        assert len(backend_figure.data) > 0

    def test_output_is_detached_even_though_the_model_requires_grad(
        self, connected_graph
    ):
        """The model's parameters require grad, so its output does too -
        visualize() must hand auto_plot()/Plot.evaluate() already-detached
        data, the same as every other caller does by convention."""
        graph, sampler, model = connected_graph
        layout = graph.visualize(model.output_ports[0])
        result = layout.plots[0].evaluate()
        assert isinstance(np.asarray(result.color), np.ndarray)

    def test_raw_sampling_without_mesh_mode_yields_a_point_cloud_plot(
        self, connected_graph
    ):
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
        layout = graph.visualize(sampler.output_ports[0])
        assert isinstance(layout.plots[0], Plot)

    def test_multiple_ports_return_a_row(self, connected_graph):
        graph, sampler, model = connected_graph
        layout = graph.visualize([model.output_ports[0], sampler.output_ports[0]])
        assert isinstance(layout, Row)
        assert len(layout.plots) == 2

    def test_reference_with_more_than_one_port_raises(self, connected_graph):
        graph, sampler, model = connected_graph
        config = model.output_ports[0].get_data_configuration(graph)
        with pytest.raises(ValueError, match="single port"):
            graph.visualize(
                [model.output_ports[0], sampler.output_ports[0]],
                reference=np.zeros((1, 1)),
                reference_config=config,
            )

    def test_reference_as_an_unrelated_port_raises_on_variable_mismatch(
        self, connected_graph
    ):
        graph, sampler, model = connected_graph
        with pytest.raises(ValueError, match="same Variable instance"):
            graph.visualize(model.output_ports[0], reference=sampler.output_ports[0])

    def test_sampled_geometry_is_plain_numpy_after_visualize(self, connected_graph):
        """The model/sampler can stay on whatever device they were trained
        on the whole time - visualize() moves only the data a Plot actually
        reads, right before handing it over, not the graph itself."""
        graph, sampler, model = connected_graph
        graph.visualize(model.output_ports[0])
        assert isinstance(sampler.sampled_geometry.discretization_points, np.ndarray)


class TestVisualizeDevice:
    def test_an_explicit_device_moves_every_node_needed_for_the_run(
        self, connected_graph
    ):
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


@pytest.fixture
def reference_setup(connected_graph):
    """A PointCloud reference geometry (inside the sampler's Rectangle
    domain) with known values, using the exact same Variable as the
    model's own output."""
    graph, sampler, model = connected_graph
    U = model.output_ports[0].get_data_configuration(graph).feature_axes.variables
    points = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]], dtype=np.float32)
    ref_values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    ref_geometry = PointCloud(Variable("y", 2), points)
    ref_config = DataConfiguration(GeometryAxes(ref_geometry), FeatureAxes(U))
    return ref_values, ref_config


class TestVisualizeWithReference:
    def test_produces_a_row_of_reference_prediction_and_error(
        self, connected_graph, reference_setup
    ):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0], reference=ref_values, reference_config=ref_config
        )
        assert isinstance(layout, Row)
        assert len(layout.plots) == 3
        assert [p.title for p in layout.plots] == ["Reference", "Prediction", "Error"]

    def test_reference_and_prediction_share_one_scale_error_gets_its_own(
        self, connected_graph, reference_setup
    ):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0], reference=ref_values, reference_config=ref_config
        )
        reference_plot, prediction_plot, error_plot = layout.plots
        assert reference_plot.color.scale is prediction_plot.color.scale
        assert error_plot.color.scale is not reference_plot.color.scale
        assert error_plot.color.scale.symmetric is True

    def test_share_scale_false_gives_independent_scales(
        self, connected_graph, reference_setup
    ):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            share_scale=False,
        )
        reference_plot, prediction_plot, _ = layout.plots
        assert reference_plot.color.scale is None
        assert prediction_plot.color.scale is None

    def test_error_none_omits_the_error_panel(self, connected_graph, reference_setup):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            error=None,
        )
        assert len(layout.plots) == 2
        assert [p.title for p in layout.plots] == ["Reference", "Prediction"]

    def test_signed_error_is_prediction_minus_reference(
        self, connected_graph, reference_setup
    ):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            error="signed",
        )
        reference_plot, prediction_plot, error_plot = layout.plots
        ref_values = np.asarray(reference_plot.evaluate().color)
        pred_values = np.asarray(prediction_plot.evaluate().color)
        error_values = np.asarray(error_plot.evaluate().color)
        assert np.allclose(error_values, pred_values - ref_values)

    def test_absolute_error_is_non_negative(self, connected_graph, reference_setup):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            error="absolute",
        )
        _, _, error_plot = layout.plots
        error_values = np.asarray(error_plot.evaluate().color)
        assert np.all(error_values >= 0)
        assert error_plot.color.scale.vmin == 0.0

    def test_relative_error_masks_near_zero_reference_with_nan(self, connected_graph):
        graph, sampler, model = connected_graph
        U = model.output_ports[0].get_data_configuration(graph).feature_axes.variables
        points = np.array([[0.2, 0.2], [0.5, 0.5]], dtype=np.float32)
        ref_values = np.array([[0.0], [2.0]], dtype=np.float32)  # first ~0 -> NaN
        ref_geometry = PointCloud(Variable("y", 2), points)
        ref_config = DataConfiguration(GeometryAxes(ref_geometry), FeatureAxes(U))

        layout = graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            error="relative",
        )
        _, _, error_plot = layout.plots
        error_values = np.asarray(error_plot.evaluate().color)
        assert np.isnan(error_values[0])
        assert not np.isnan(error_values[1])

    def test_invalid_error_mode_raises(self, connected_graph, reference_setup):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        with pytest.raises(ValueError, match="signed"):
            graph.visualize(
                model.output_ports[0],
                reference=ref_values,
                reference_config=ref_config,
                error="bogus",
            )

    def test_variable_identity_mismatch_raises(self, connected_graph):
        graph, sampler, model = connected_graph
        other_variable = Variable("not_u", 1)  # not the model's own output Variable
        points = np.array([[0.2, 0.2]], dtype=np.float32)
        ref_geometry = PointCloud(Variable("y", 2), points)
        ref_config = DataConfiguration(
            GeometryAxes(ref_geometry), FeatureAxes(other_variable)
        )
        with pytest.raises(ValueError, match="same Variable instance"):
            graph.visualize(
                model.output_ports[0],
                reference=np.zeros((1, 1)),
                reference_config=ref_config,
            )

    def test_missing_geometry_axes_on_reference_raises(self, connected_graph):
        graph, sampler, model = connected_graph
        U = model.output_ports[0].get_data_configuration(graph).feature_axes.variables
        ref_config = DataConfiguration(FeatureAxes(U))
        with pytest.raises(ValueError, match="GeometryAxes"):
            graph.visualize(
                model.output_ports[0],
                reference=np.zeros((1, 1)),
                reference_config=ref_config,
            )

    def test_missing_reference_config_for_plain_data_raises(self, connected_graph):
        """reference_config is required when reference isn't a Port or a
        callable - there's no geometry/config of its own to derive one
        from otherwise."""
        graph, sampler, model = connected_graph
        with pytest.raises(ValueError, match="reference_config"):
            graph.visualize(model.output_ports[0], reference=np.zeros((1, 1)))

    def test_active_discretization_is_cleared_after_the_call(
        self, connected_graph, reference_setup
    ):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        graph.visualize(
            model.output_ports[0], reference=ref_values, reference_config=ref_config
        )
        assert sampler.active_discretization is None

    def test_draws_without_error(self, connected_graph, reference_setup):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        layout = graph.visualize(
            model.output_ports[0], reference=ref_values, reference_config=ref_config
        )
        backend_figure = Figure(layout).draw()
        assert len(backend_figure.data) > 0

    def test_prediction_and_error_reuse_the_references_resolved_controls(
        self, connected_graph, reference_setup, monkeypatch
    ):
        """one ControlSpec instance shared
        across every panel - prediction/error must be resolved with
        whatever reference_plot's own auto_plot() call already resolved,
        not independently re-resolve their own."""
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        from qewton.visualization import auto as auto_module

        real_auto_plot = auto_module.auto_plot
        seen_controls = []

        def spy(*args, **kwargs):
            seen_controls.append(kwargs.get("controls"))
            return real_auto_plot(*args, **kwargs)

        monkeypatch.setattr(auto_module, "auto_plot", spy)
        graph.visualize(
            model.output_ports[0], reference=ref_values, reference_config=ref_config
        )
        # probe, reference (both resolve their own - unconstrained), then
        # prediction and error - the latter two must carry the exact same,
        # already-resolved controls list reference_plot ended up with.
        assert len(seen_controls) == 4
        assert seen_controls[2] is not None
        assert seen_controls[2] is seen_controls[3]

    def test_variables_redirects_matching_specs_to_one_shared_variablespec(self):
        """Unit-level check of _redirect_to_shared_variable (used by
        variables= ): every PlotSpec
        attribute naming one of `variables` is redirected to the shared
        instance; attributes naming something else are left alone."""
        from qewton.config.axes import BatchAxes
        from qewton.visualization.plots.data.curve import LinePlot
        from qewton.visualization.plots.spec import VariableSpec

        temperature = Variable("temperature", 1)
        pressure = Variable("pressure", 1)
        sample_axis = BatchAxes(5)
        data = np.random.rand(5, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(temperature))
        plot = LinePlot(data, config, x=sample_axis, y=temperature)

        shared_spec = VariableSpec([temperature, pressure])
        Graph._redirect_to_shared_variable(plot, [temperature, pressure], shared_spec)
        assert plot.y.embedded_variable_spec is shared_spec

        unrelated = Variable("other", 1)
        plot2 = LinePlot(data, config, x=sample_axis, y=temperature)
        Graph._redirect_to_shared_variable(plot2, [unrelated], shared_spec)
        assert plot2.y.embedded_variable_spec is None


@pytest.fixture
def operator_learning_graph():
    """Two independent models fed by the same sampler, both declared with
    the exact same output Variable U - stands in for a DataLoader's "true"
    output port vs. a model's own prediction, without needing GridDataSet
    machinery (see examples/OperatorLearning/DeepONets/integrator.ipynb for
    the real dataloader-driven version)."""
    X = Variable("x", 2)
    U = Variable("u", 1)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    sampler = GridSampler(square, 20)
    model = FCN(in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1)
    reference_model = FCN(
        in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1
    )
    graph = Graph()
    graph.connect(sampler, model)
    graph.connect(sampler, reference_model)
    graph.setup()
    return graph, sampler, model, reference_model


class TestVisualizeWithPortReference:
    def test_produces_a_row_of_reference_prediction_and_error(
        self, operator_learning_graph
    ):
        graph, sampler, model, reference_model = operator_learning_graph
        layout = graph.visualize(
            model.output_ports[0], reference=reference_model.output_ports[0]
        )
        assert isinstance(layout, Row)
        assert len(layout.plots) == 3
        assert [p.title for p in layout.plots] == ["Reference", "Prediction", "Error"]

    def test_both_sides_come_from_one_run(self, operator_learning_graph):
        """No sampler geometry substitution for a Port reference - both
        sides are already graph data evaluated together."""
        graph, sampler, model, reference_model = operator_learning_graph
        graph.visualize(model.output_ports[0], reference=reference_model.output_ports[0])
        assert sampler.active_discretization is None
        assert sampler.mesh_mode is False

    def test_signed_error_matches_the_two_predictions(self, operator_learning_graph):
        graph, sampler, model, reference_model = operator_learning_graph
        layout = graph.visualize(
            model.output_ports[0], reference=reference_model.output_ports[0]
        )
        reference_plot, prediction_plot, error_plot = layout.plots
        ref_values = np.asarray(reference_plot.evaluate().color)
        pred_values = np.asarray(prediction_plot.evaluate().color)
        error_values = np.asarray(error_plot.evaluate().color)
        assert np.allclose(error_values, pred_values - ref_values)

    def test_shares_scale_between_reference_and_prediction(self, operator_learning_graph):
        graph, sampler, model, reference_model = operator_learning_graph
        layout = graph.visualize(
            model.output_ports[0], reference=reference_model.output_ports[0]
        )
        reference_plot, prediction_plot, error_plot = layout.plots
        assert reference_plot.color.scale is prediction_plot.color.scale
        assert error_plot.color.scale is not reference_plot.color.scale

    def test_draws_without_error(self, operator_learning_graph):
        graph, sampler, model, reference_model = operator_learning_graph
        layout = graph.visualize(
            model.output_ports[0], reference=reference_model.output_ports[0]
        )
        backend_figure = Figure(layout).draw()
        assert len(backend_figure.data) > 0


class TestVisualizeWithConfigOverrides:
    """prediction_config/reference_config let the caller supply the
    correct DataConfiguration in place of whatever
    port.get_data_configuration(self) would otherwise return - needed when
    a node's own dynamic config doesn't carry the Variable it was actually
    declared with (e.g. a GraphNode-wrapped model resolving its config from
    an inner graph, rather than the outer port the caller connected)."""

    def test_prediction_config_overrides_the_variable_used_for_the_check(
        self, connected_graph
    ):
        graph, sampler, model = connected_graph
        real_pred_config = model.output_ports[0].get_data_configuration(graph)
        V = Variable("v", 1)  # not model's real output Variable (U)
        override_config = DataConfiguration(
            real_pred_config.geometry_axes, FeatureAxes(V)
        )

        points = np.array([[0.2, 0.2]], dtype=np.float32)
        ref_geometry = PointCloud(Variable("y", 2), points)
        ref_config = DataConfiguration(GeometryAxes(ref_geometry), FeatureAxes(V))

        # Without the override, model's real output Variable (U) doesn't
        # match ref's V.
        with pytest.raises(ValueError, match="same Variable instance"):
            graph.visualize(
                model.output_ports[0],
                reference=np.zeros((1, 1)),
                reference_config=ref_config,
            )

        # prediction_config supplies V too - the check now passes without
        # touching port.get_data_configuration() at all.
        layout = graph.visualize(
            model.output_ports[0],
            reference=np.zeros((1, 1)),
            reference_config=ref_config,
            prediction_config=override_config,
        )
        assert isinstance(layout, Row)

    def test_reference_config_overrides_the_variable_used_for_the_check(self):
        X = Variable("x", 2)
        U = Variable("u", 1)
        W = Variable("w", 1)  # reference_model's real output Variable - not U
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
        sampler = GridSampler(square, 20)
        model = FCN(in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1)
        reference_model = FCN(
            in_neurons=X, hidden_neurons=4, out_neurons=W, n_hidden_layers=1
        )
        graph = Graph()
        graph.connect(sampler, model)
        graph.connect(sampler, reference_model)
        graph.setup()

        # Without the override, reference_model's real W doesn't match
        # model's real U.
        with pytest.raises(ValueError, match="same Variable"):
            graph.visualize(
                model.output_ports[0], reference=reference_model.output_ports[0]
            )

        real_ref_config = reference_model.output_ports[0].get_data_configuration(graph)
        override_config = DataConfiguration(real_ref_config.geometry_axes, FeatureAxes(U))
        layout = graph.visualize(
            model.output_ports[0],
            reference=reference_model.output_ports[0],
            reference_config=override_config,
        )
        assert isinstance(layout, Row)


def _zero_reference(points):
    return np.zeros(points.shape[:-1] + (1,))


class TestVisualizeWithCallableReference:
    """reference= as a plain callable - evaluated at the model's own
    natural mesh-mode points, no separate geometry to build by hand."""

    def test_produces_a_row_of_reference_prediction_and_error(self, connected_graph):
        graph, sampler, model = connected_graph
        layout = graph.visualize(model.output_ports[0], reference=_zero_reference)
        assert isinstance(layout, Row)
        assert len(layout.plots) == 3
        assert [p.title for p in layout.plots] == ["Reference", "Prediction", "Error"]

    def test_evaluates_the_reference_at_the_models_own_points(self, connected_graph):
        graph, sampler, model = connected_graph
        captured = {}

        def ref_fn(points):
            captured["points"] = points
            return _zero_reference(points)

        layout = graph.visualize(model.output_ports[0], reference=ref_fn)
        reference_plot, prediction_plot, _ = layout.plots
        assert "points" in captured
        assert isinstance(captured["points"], np.ndarray)
        # ref/pred share the same geometry - the reference is evaluated at
        # exactly the points the model itself used, not a separate one.
        assert (
            reference_plot.data_config.geometry_axes.geometry
            is prediction_plot.data_config.geometry_axes.geometry
        )

    def test_signed_error_matches_prediction_minus_reference(self, connected_graph):
        graph, sampler, model = connected_graph
        layout = graph.visualize(model.output_ports[0], reference=_zero_reference)
        reference_plot, prediction_plot, error_plot = layout.plots
        ref_values = np.asarray(reference_plot.evaluate().color)
        pred_values = np.asarray(prediction_plot.evaluate().color)
        error_values = np.asarray(error_plot.evaluate().color)
        assert np.allclose(error_values, pred_values - ref_values)

    def test_shares_scale_between_reference_and_prediction(self, connected_graph):
        graph, sampler, model = connected_graph
        layout = graph.visualize(model.output_ports[0], reference=_zero_reference)
        reference_plot, prediction_plot, error_plot = layout.plots
        assert reference_plot.color.scale is prediction_plot.color.scale
        assert error_plot.color.scale is not reference_plot.color.scale

    def test_draws_without_error(self, connected_graph):
        graph, sampler, model = connected_graph
        layout = graph.visualize(model.output_ports[0], reference=_zero_reference)
        backend_figure = Figure(layout).draw()
        assert len(backend_figure.data) > 0

    def test_a_lambda_works_too(self, connected_graph):
        graph, sampler, model = connected_graph
        layout = graph.visualize(
            model.output_ports[0], reference=lambda points: _zero_reference(points)
        )
        assert isinstance(layout, Row)
        assert len(layout.plots) == 3


class TestVisualizeEvaluationPhase:
    """mode= controls which EvaluationPhase every DataLoader-like node on
    the path is run in - defaults to VALIDATION, both with and without a
    reference."""

    def _spy_on_run_mode(self, monkeypatch):
        seen_modes = []
        original = Graph._run_nodes_needed_for

        def spy(self, ports, mode):
            seen_modes.append(mode)
            return original(self, ports, mode)

        monkeypatch.setattr(Graph, "_run_nodes_needed_for", spy)
        return seen_modes

    def test_defaults_to_validation(self, connected_graph, monkeypatch):
        graph, sampler, model = connected_graph
        seen_modes = self._spy_on_run_mode(monkeypatch)
        graph.visualize(model.output_ports[0])
        assert seen_modes == [EvaluationPhase.VALIDATION]

    def test_mode_is_overridable(self, connected_graph, monkeypatch):
        graph, sampler, model = connected_graph
        seen_modes = self._spy_on_run_mode(monkeypatch)
        graph.visualize(model.output_ports[0], mode=EvaluationPhase.TEST)
        assert seen_modes == [EvaluationPhase.TEST]

    def test_reference_path_also_respects_mode(
        self, connected_graph, reference_setup, monkeypatch
    ):
        graph, sampler, model = connected_graph
        ref_values, ref_config = reference_setup
        seen_modes = self._spy_on_run_mode(monkeypatch)
        graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            mode=EvaluationPhase.TEST,
        )
        assert seen_modes == [EvaluationPhase.TEST]

    def test_port_reference_path_also_respects_mode(
        self, operator_learning_graph, monkeypatch
    ):
        graph, sampler, model, reference_model = operator_learning_graph
        seen_modes = self._spy_on_run_mode(monkeypatch)
        graph.visualize(
            model.output_ports[0],
            reference=reference_model.output_ports[0],
            mode=EvaluationPhase.TEST,
        )
        assert seen_modes == [EvaluationPhase.TEST]


@pytest.fixture
def mixed_variable_graph():
    """A model whose single output port is a MIXED-dim composed Variable
    (a 2D vector V and a scalar S) - auto_plot can't dispatch this as one
    quantity at all (no Plot family takes a dim=3 field on a 2D domain),
    the scenario variables= narrowing exists for."""
    X = Variable("x", 2)
    V = Variable("v", 2)
    S = Variable("s", 1)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    sampler = GridSampler(square, 20)
    model = FCN(in_neurons=X, hidden_neurons=4, out_neurons=V + S, n_hidden_layers=1)
    graph = Graph()
    graph.connect(sampler, model)
    graph.setup()
    return graph, sampler, model, V, S


class TestVisualizeWithVariablesNarrowing:
    """variables= narrows port's (and reference's) composed Variable down
    to just the listed ones before auto_plot ever runs - a single element
    selects it outright; several also get a shared dropdown, same as the
    (pre-existing) same-dim-bundle case, now reachable even when the full
    port Variable itself isn't jointly dispatchable."""

    def test_no_reference_single_variable_selects_the_vector_component(
        self, mixed_variable_graph
    ):
        from qewton.visualization.plots.data.mesh import MeshVectorPlot

        graph, sampler, model, V, S = mixed_variable_graph
        layout = graph.visualize(model.output_ports[0], variables=[V])
        assert isinstance(layout, Overlay)
        assert isinstance(layout.plots[0], MeshVectorPlot)

    def test_no_reference_single_variable_selects_the_scalar_component(
        self, mixed_variable_graph
    ):
        graph, sampler, model, V, S = mixed_variable_graph
        layout = graph.visualize(model.output_ports[0], variables=[S])
        assert isinstance(layout.plots[0], MeshFieldPlot)

    def test_no_reference_without_variables_raises_on_the_full_mixed_output(
        self, mixed_variable_graph
    ):
        graph, sampler, model, V, S = mixed_variable_graph
        with pytest.raises(ValueError):
            graph.visualize(model.output_ports[0])

    def test_reference_single_variable_lets_an_otherwise_mismatched_reference_pass(
        self, mixed_variable_graph
    ):
        graph, sampler, model, V, S = mixed_variable_graph
        points = np.array([[0.2, 0.2], [0.5, 0.5]], dtype=np.float32)
        ref_values = np.array([[1.0], [2.0]], dtype=np.float32)
        ref_geometry = PointCloud(Variable("y", 2), points)
        ref_config = DataConfiguration(GeometryAxes(ref_geometry), FeatureAxes(S))

        with pytest.raises(ValueError, match="same Variable"):
            graph.visualize(
                model.output_ports[0], reference=ref_values, reference_config=ref_config
            )

        layout = graph.visualize(
            model.output_ports[0],
            reference=ref_values,
            reference_config=ref_config,
            variables=[S],
            error="absolute",
        )
        assert isinstance(layout, Row)
        assert len(layout.plots) == 3

    def test_multiple_variables_drop_one_and_switch_between_the_rest(self):
        """Three same-dim scalars bundled together - variables=[T, P] must
        drop Q entirely (not just hide it) while still offering a live
        dropdown between T and P, exactly like the pre-existing same-dim
        dropdown case."""
        X = Variable("x", 2)
        T = Variable("temperature", 1)
        P = Variable("pressure", 1)
        Q = Variable("humidity", 1)
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
        sampler = GridSampler(square, 20)
        model = FCN(in_neurons=X, hidden_neurons=4, out_neurons=T + P + Q, n_hidden_layers=1)
        graph = Graph()
        graph.connect(sampler, model)
        graph.setup()

        layout = graph.visualize(model.output_ports[0], variables=[T, P])
        plot = layout.plots[0]
        assert isinstance(plot, MeshFieldPlot)
        assert plot.color.embedded_variable_spec is not None
        candidate_names = {v.name for v in plot.color.embedded_variable_spec.candidates}
        assert candidate_names == {"temperature", "pressure"}
