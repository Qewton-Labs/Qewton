import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.spec import (
    ColorSpec,
    FacetSpec,
    FixedSpec,
    PlotSpec,
    Scale,
    SliderSpec,
    TimeSpec,
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
        name - ColorSpec must not require a Variable (see the input-family
        boundary, implementation plan section 4)."""
        spec = ColorSpec("loss")
        assert spec.name == "loss"


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
