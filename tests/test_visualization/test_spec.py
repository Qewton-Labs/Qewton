import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.spec import (
    AxisSpec,
    ColorSpec,
    FacetSpec,
    FixedSpec,
    PlotSpec,
    Scale,
    SliderSpec,
    TimeSpec,
    VariableSpec,
)


def test_as_single_dim_passes_through_int():
    assert PlotSpec.as_single_dim(3) == 3


def test_as_single_dim_normalizes_length_one_slice():
    assert PlotSpec.as_single_dim(slice(2, 3)) == 2


def test_as_single_dim_rejects_multi_length_slice():
    with pytest.raises(AssertionError):
        PlotSpec.as_single_dim(slice(2, 5))


def test_get_slice_resolves_batch_axis_by_identity():
    batch = BatchAxes(10)
    other = BatchAxes(10)  # same shape, different identity - must NOT match
    config = DataConfiguration(batch, FeatureAxes(Variable("u", 1)))
    axis_slc, entry_slc = PlotSpec.get_slice(batch, config)
    assert axis_slc == 0 and entry_slc is None
    with pytest.raises(ValueError):
        PlotSpec.get_slice(other, config)


def test_get_slice_resolves_feature_channel():
    X = Variable("x", 1)
    Y = Variable("y", 1)
    config = DataConfiguration(BatchAxes(5), FeatureAxes(X * Y))
    axis_slc, entry_slc = PlotSpec.get_slice(Y, config)
    # Y is the feature axis (index 1); entry_slc selects Y's channel within it
    assert axis_slc == 1
    assert entry_slc == slice(1, 2)


class TestPlotSpecMathName:
    def test_variable_backed_spec_is_wrapped_for_math_mode(self):
        spec = AxisSpec(Variable("u", 1))
        assert spec.name == "u"
        assert spec.math_name == "$u$"

    def test_axes_backed_spec_is_not_wrapped(self):
        axis = BatchAxes(5)
        spec = AxisSpec(axis)
        assert spec.math_name == spec.name == str(axis)

    def test_table_column_key_is_not_wrapped(self):
        spec = AxisSpec("some_column")
        assert spec.math_name == spec.name == "some_column"

    def test_1d_geometry_axes_backed_spec_uses_the_geometrys_own_variable(self):
        """A GeometryAxes itself, not a Variable, is what auto_plot()'s
        1D-mesh/1D-point-cloud LinePlot dispatch passes as x= (resolving
        the axis needs the whole GeometryAxes) - the geometry's own single
        coordinate Variable is still the name a reader recognizes, not
        str(geometry_axes)."""
        from qewton.geometries.discrete.point_cloud import PointCloud

        T = Variable("t", 1)
        points = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        geometry = PointCloud(T, points)
        spec = AxisSpec(GeometryAxes(geometry))
        assert spec.name == "t"
        assert spec.math_name == "$t$"

    def test_multi_component_geometry_axes_falls_back_to_str(self):
        from qewton.geometries.discrete.point_cloud import PointCloud

        X = Variable("x", 2)
        points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        geometry = PointCloud(X, points)
        spec = AxisSpec(GeometryAxes(geometry))
        assert spec.math_name == spec.name == str(spec.variable_or_axes)


class TestSliderSpecResolve:
    def test_explicit_bounds_are_never_overwritten(self):
        spec = SliderSpec(BatchAxes(5), init_state=2, minimum=0, maximum=1)
        spec.resolve(range(100))
        assert spec.minimum == 0
        assert spec.maximum == 1
        assert spec.state == 2

    def test_auto_resolves_bounds_from_values(self):
        spec = SliderSpec(BatchAxes(5), init_state=None, minimum=None, maximum=None)
        spec.resolve(range(5))  # matches DataPlot's range(data.shape[dim]) convention
        assert spec.minimum == 0
        assert spec.maximum == 4
        assert spec.state == 0  # defaults to minimum

    def test_auto_resolves_bounds_from_arbitrary_values(self):
        """TablePlot passes actual distinct column values, not a 0-based
        index range - SliderSpec must not assume the latter."""
        spec = SliderSpec(BatchAxes(5), init_state=None, minimum=None, maximum=None)
        spec.resolve([10.5, 20.5, 30.5])
        assert spec.minimum == 10.5
        assert spec.maximum == 30.5
        assert spec.state == 10.5


class TestFacetSpecResolve:
    def test_explicit_values_are_never_overwritten(self):
        spec = FacetSpec(BatchAxes(5), values=[9, 8, 7])
        spec.resolve(range(100))
        assert spec.values == [9, 8, 7]

    def test_auto_resolves_values(self):
        spec = FacetSpec(BatchAxes(3))
        spec.resolve(range(3))
        assert spec.values == [0, 1, 2]

    def test_state_defaults_to_zero_regardless_of_values(self):
        """FacetSpec.__init__ hardcodes init_state=0 (unlike SliderSpec/
        TimeSpec, which start at None and get filled from resolve()) -
        harmless, since Figure._draw_plot() drives faceting off
        spec.values directly, never off spec.state."""
        spec = FacetSpec(BatchAxes(3), values=[5, 6, 7])
        assert spec.state == 0

    def test_orientation_must_be_row_or_col(self):
        with pytest.raises(AssertionError):
            FacetSpec(BatchAxes(3), orientation="diagonal")


class TestTimeSpecResolve:
    def test_auto_resolves_values_and_default_duration(self):
        spec = TimeSpec(BatchAxes(4))
        assert spec.duration == 500
        spec.resolve(range(4))
        assert spec.values == [0, 1, 2, 3]
        assert spec.state == 0

    def test_explicit_values_are_never_overwritten(self):
        spec = TimeSpec(BatchAxes(4), values=[3, 2, 1, 0], duration=100)
        spec.resolve(range(100))
        assert spec.values == [3, 2, 1, 0]
        assert spec.duration == 100


def test_fixed_spec_state_is_immutable():
    spec = FixedSpec(init_state=2, n_dimensions=1, variable_or_axes=BatchAxes(5))
    assert spec.state == 2
    with pytest.raises(ValueError):
        spec.state = 3


class TestColorSpec:
    def test_accepts_a_variable(self):
        spec = ColorSpec(Variable("u", 1))
        assert spec.name == "u"

    def test_accepts_a_plain_string_column_key(self):
        """TablePlot's own family: no Variable/Axes at all, just a column
        name - ColorSpec must not require a Variable."""
        spec = ColorSpec("loss")
        assert spec.name == "loss"


class TestVariableSpec:
    def test_requires_at_least_two_candidates(self):
        with pytest.raises(AssertionError):
            VariableSpec([Variable("u", 1)])

    def test_requires_matching_dims(self):
        with pytest.raises(AssertionError):
            VariableSpec([Variable("u", 1), Variable("v", 2)])

    def test_state_defaults_to_the_first_candidate(self):
        u, v = Variable("u", 1), Variable("v", 1)
        spec = VariableSpec([u, v])
        assert spec.state is u

    def test_state_can_be_set_by_index_or_by_variable(self):
        u, v = Variable("u", 1), Variable("v", 1)
        spec = VariableSpec([u, v])
        spec.state = 1
        assert spec.state is v
        spec.state = u
        assert spec.state is u

    def test_state_rejects_a_variable_outside_the_candidates(self):
        u, v, w = Variable("u", 1), Variable("v", 1), Variable("w", 1)
        spec = VariableSpec([u, v])
        with pytest.raises(AssertionError):
            spec.state = w

    def test_color_spec_transparently_unwraps_the_current_selection(self):
        """This is the whole point: ColorSpec never needs to know
        VariableSpec exists - reading .variable_or_axes/.name just reflects
        whichever candidate is currently selected."""
        u, v = Variable("u", 1), Variable("v", 1)
        selector = VariableSpec([u, v])
        color = ColorSpec(selector)
        assert color.variable_or_axes is u
        assert color.name == "u"

        selector.state = v
        assert color.variable_or_axes is v
        assert color.name == "v"

    def test_get_variable_slice_follows_the_current_selection(self):
        """Confirms selecting a variable really is just selecting indices -
        no changes anywhere else in the DataConfiguration/PlotSpec pipeline
        are needed for this to resolve to the right slice."""
        u, v = Variable("u", 1), Variable("v", 1)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(u * v))
        selector = VariableSpec([u, v])
        color = ColorSpec(selector)

        assert config.get_variable_slice(color.variable_or_axes) == (slice(None), slice(0, 1))
        selector.state = v
        assert config.get_variable_slice(color.variable_or_axes) == (slice(None), slice(1, 2))

    def test_mesh_field_plot_draws_whichever_variable_is_currently_selected(self, small_mesh_geometry):
        """End-to-end through a real Plot.evaluate(), not just get_slice()
        math - confirms MeshFieldPlot needed zero changes to support this."""
        from qewton.visualization.plots.data.mesh import MeshFieldPlot

        temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
        n = small_mesh_geometry.mesh.vertices.shape[0]
        data = np.stack([np.full(n, 10.0), np.full(n, 20.0)], axis=-1)
        config = DataConfiguration(
            GeometryAxes(small_mesh_geometry), FeatureAxes(temperature * pressure)
        )
        selector = VariableSpec([temperature, pressure])
        plot = MeshFieldPlot(data, config, color=ColorSpec(selector))

        assert np.all(plot.evaluate().color == 10.0)
        selector.state = pressure
        assert np.all(plot.evaluate().color == 20.0)


class TestScale:
    def test_observed_range_is_none_before_any_observation(self):
        scale = Scale()
        assert scale.range is None

    def test_observe_widens_the_range_across_calls(self):
        scale = Scale()
        scale.observe([1.0, 2.0, 3.0])
        scale.observe([-5.0, 0.0])
        assert scale.range == (-5.0, 3.0)

    def test_observe_ignores_nan(self):
        scale = Scale()
        scale.observe(np.array([1.0, np.nan, 5.0]))
        assert scale.range == (1.0, 5.0)

    def test_explicit_bounds_win_over_observed(self):
        scale = Scale(vmin=-1.0, vmax=1.0)
        scale.observe([100.0, -100.0])
        assert scale.range == (-1.0, 1.0)

    def test_symmetric_centers_on_zero(self):
        scale = Scale(symmetric=True)
        scale.observe([2.0, -5.0])
        assert scale.range == (-5.0, 5.0)

    def test_claim_colorbar_is_first_come_first_served(self):
        scale = Scale()
        assert scale.claim_colorbar() is True
        assert scale.claim_colorbar() is False
        assert scale.claim_colorbar() is False

    def test_reset_clears_observations_and_colorbar_claim(self):
        scale = Scale()
        scale.observe([1.0, 2.0])
        scale.claim_colorbar()
        scale.reset()
        assert scale.range is None
        assert scale.claim_colorbar() is True
