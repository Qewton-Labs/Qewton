import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.plots.spec import ColorSpec, FixedSpec, Scale, SliderSpec


class TestScatterPlot:
    def test_evaluate_flattens_over_the_implicit_samples_axis(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(25)
        data = np.random.randn(25, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y)
        result = plot.evaluate()
        assert result.x.shape == (25,)
        assert result.y.shape == (25,)
        assert result.color is None

    def test_no_geometry_axes_required(self):
        """A third family alongside grids and meshes: works with only
        BatchAxes + FeatureAxes, no GeometryAxes anywhere - confirms Plot
        really doesn't need a geometry."""
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y)
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "scatter"
        assert backend_figure.data[0].mode == "markers"

    def test_an_extra_untouched_dimension_just_flattens_into_the_sample_count(self):
        """x and y are value roles over an *implicit* samples axis, not one
        named structural domain (unlike LinePlot.x) - any dimension neither
        control nor color/x/y claims just becomes more samples once
        flattened, no error."""
        X, Y = Variable("x", 1), Variable("y", 1)
        extra_axis = BatchAxes(3)
        sample_axis = BatchAxes(10)
        data = np.random.randn(3, 10, 2)
        config = DataConfiguration(extra_axis, sample_axis, FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y)
        result = plot.evaluate()
        assert result.x.shape == (30,)

    def test_resolving_the_extra_dimension_with_a_slider_selects_one_batch(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        extra_axis = BatchAxes(3)
        sample_axis = BatchAxes(10)
        data = np.random.randn(3, 10, 2)
        config = DataConfiguration(extra_axis, sample_axis, FeatureAxes(X * Y))
        slider = SliderSpec(extra_axis, init_state=None, minimum=None, maximum=None)
        plot = ScatterPlot(data, config, x=X, y=Y, controls=[slider])
        result = plot.evaluate()
        assert result.x.shape == (10,)
        assert np.allclose(result.x, data[0, :, 0])

    def test_x_and_y_must_be_scalar(self):
        XY = Variable("xy", 2)
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 3)
        config = DataConfiguration(sample_axis, FeatureAxes(XY * Y))
        with pytest.raises(ValueError, match="must be scalar"):
            ScatterPlot(data, config, x=XY, y=Y)

    def test_two_scatter_plots_share_a_scale(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(10)
        data_a = np.random.randn(10, 2)
        data_b = np.random.randn(10, 2) * 10  # a wider range
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        shared = Scale()
        plot_a = ScatterPlot(data_a, config, x=X, y=Y, color=ColorSpec(X, scale=shared))
        plot_b = ScatterPlot(data_b, config, x=X, y=Y, color=ColorSpec(X, scale=shared))
        fig = Figure([plot_a, plot_b])
        fig.draw()
        assert shared.range is not None
        # both traces are pinned to the same trained range, not their own local one
        assert fig.backend_figure.data[0].marker.cmin == fig.backend_figure.data[1].marker.cmin
        assert fig.backend_figure.data[0].marker.cmax == fig.backend_figure.data[1].marker.cmax
