import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization.plots.base import Plot, axis_names_from_variable
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.plots.spec import (
    AxisSpec,
    ColorSpec,
    FixedSpec,
    Scale,
    SliderSpec,
)


class TestAxisNamesFromVariable:
    """Every returned label is wrapped for TeX math-mode rendering (see
    PlotSpec.math_name) - these are always axis titles."""

    def test_decomposes_a_composed_variable_by_leaf_name(self):
        x, y, z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        composed = x * y * z
        assert axis_names_from_variable(composed, 3) == ["$x$", "$y$", "$z$"]

    def test_auto_named_children_of_a_plain_multi_dim_variable(self):
        var = Variable("x", dim=3)
        assert axis_names_from_variable(var, 3) == ["$x_1$", "$x_2$", "$x_3$"]

    def test_falls_back_to_generic_names_on_leaf_count_mismatch(self):
        var = Variable("x", dim=2)
        assert axis_names_from_variable(var, 3) == ["$x$", "$y$", "$z$"]

    def test_falls_back_to_generic_names_when_variable_is_none(self):
        assert axis_names_from_variable(None, 2) == ["$x$", "$y$"]

    def test_falls_back_beyond_three_axes(self):
        assert axis_names_from_variable(None, 5) == [
            "$x$",
            "$y$",
            "$z$",
            "$axis_3$",
            "$axis_4$",
        ]


class TestPlotTheme:
    def test_theme_is_set_once_and_not_overwritten(self):
        plot = Plot()
        assert plot.theme is None
        plot.theme = "first"
        plot.theme = "second"
        assert plot.theme == "first"

    def test_embedding_dim_defaults_to_2(self):
        assert Plot().embedding_dim == 2


class TestPlotLabel:
    """label is the legend-entry text, distinct from title (a panel/
    subplot heading)."""

    def test_defaults_to_none(self):
        assert Plot().label is None

    def test_returns_the_given_label(self):
        assert Plot(label="Predicted").label == "Predicted"

    def test_is_independent_of_title(self):
        plot = Plot(title="Panel Title", label="Legend Entry")
        assert plot.title == "Panel Title"
        assert plot.label == "Legend Entry"


class TestColorValues:
    @staticmethod
    def _scatter_plot(n_batches, n_samples, control_factory=None, scale=None):
        X, Y = Variable("x", 1), Variable("y", 1)
        data = np.random.randn(n_batches, n_samples, 2)
        batch_axis = BatchAxes(n_batches)
        config = DataConfiguration(batch_axis, BatchAxes(n_samples), FeatureAxes(X * Y))
        color = ColorSpec(X, scale=scale) if scale is not None else None
        controls = [control_factory(batch_axis)] if control_factory is not None else []
        plot = ScatterPlot(data, config, x=X, y=Y, color=color, controls=controls)
        return plot, (controls[0] if controls else None)

    def test_none_when_no_color_spec(self):
        plot, _ = self._scatter_plot(1, 10)
        assert plot.color_values() is None

    def test_none_when_color_spec_has_no_scale(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        data = np.random.randn(10, 2)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y, color=ColorSpec(X))
        assert plot.color_values() is None

    def test_trains_on_every_slider_state_not_just_current(self):
        """Regression: color_values() must iterate every SliderSpec state
        (restoring afterward), or a shared colorbar silently resets to
        whatever's on screen at redraw time instead of covering everything
        scrubbing could show."""
        scale = Scale()
        plot, slider = self._scatter_plot(
            3,
            10,
            control_factory=lambda axis: SliderSpec(
                axis, init_state=None, minimum=None, maximum=None
            ),
            scale=scale,
        )
        original_state = slider.state
        values = plot.color_values()
        assert values is not None
        # one batch's worth of x per state, all 3 states -> 3x the samples
        assert values.shape[0] == 3 * 10
        assert slider.state == original_state  # restored

    def test_fixed_spec_is_left_at_its_one_state(self):
        scale = Scale()
        plot, fixed = self._scatter_plot(
            3,
            10,
            control_factory=lambda axis: FixedSpec(
                init_state=1, n_dimensions=1, variable_or_axes=axis
            ),
            scale=scale,
        )
        values = plot.color_values()
        assert values.shape[0] == 10  # only the one fixed state, not all 3


class TestDataPlotControlResolution:
    """Regression coverage for the length-1-slice normalization bug found
    this session: a control naming one component of a multi-dim GeometryAxes
    resolves to a length-1 slice, which must be treated as a single
    dimension index, not rejected."""

    def test_slider_on_one_component_of_a_composite_geometry_variable(self):
        k = Variable("k", 1)
        u = Variable("u", 1)
        v = Variable("v", 1)
        grid_variable = k * u * v
        point_grid = np.random.rand(3, 4, 4, 3)
        geometry = GridGeometry(variable=grid_variable, point_grid=point_grid)
        field = Variable("f", 1)
        data = np.random.rand(3, 4, 4, 1)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(field))

        slider = SliderSpec(k, init_state=None, minimum=None, maximum=None)
        # DataPlot itself is abstract (no evaluate()/create_artist()) - just
        # constructing it is enough to exercise the resolve loop.
        plot = DataPlot(data, config, controls=[slider])
        assert slider.minimum == 0
        assert slider.maximum == 2  # k has size 3

        sliced, index_map, slice_map = plot.apply_controls()
        assert sliced.shape == (4, 4, 1)  # k-dimension consumed

    def test_apply_controls_with_no_controls_is_a_no_op(self):
        data = np.arange(24).reshape(2, 3, 4)
        config = DataConfiguration(
            BatchAxes(2), BatchAxes(3), FeatureAxes(Variable("f", 4))
        )
        plot = DataPlot(data, config)
        sliced, index_map, slice_map = plot.apply_controls()
        assert sliced.shape == (2, 3, 4)
        assert index_map(2) == 2
        assert slice_map((slice(None), slice(None), slice(0, 4))) == (
            slice(None),
            slice(None),
            slice(0, 4),
        )
