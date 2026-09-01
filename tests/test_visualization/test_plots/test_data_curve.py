import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.curve import LinePlot, PathPlot
from qewton.visualization.plots.data.samples import BarPlot
from qewton.visualization.plots.spec import VectorSpec


class TestLinePlot:
    def test_evaluate_flattens_y_over_x(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(20)
        y_values = np.sin(np.linspace(0, 6, 20))[:, None]
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(y_values, config, x=sample_axis, y=Y)
        result = plot.evaluate()
        assert result.x.shape == (20,)
        assert np.allclose(result.y, y_values[:, 0])

    def test_unresolved_batch_dimension_raises(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(5)
        extra_axis = BatchAxes(3)
        data = np.random.rand(3, 5, 1)
        config = DataConfiguration(extra_axis, sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y)
        with pytest.raises(ValueError, match="unresolved"):
            plot.evaluate()

    def test_draws_and_shares_y_axis_range_across_plots(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y)
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "scatter"
        assert backend_figure.data[0].mode == "lines"

    def test_legend_name_prefers_label_over_the_y_variable(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y, label="Predicted")
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].name == "Predicted"

    def test_legend_name_falls_back_to_the_y_variable_not_the_title(self):
        """title is a panel/subplot heading, not a legend label - setting
        one must not leak into the trace's legend entry."""
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y, title="Panel Heading")
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].name == "y"


class TestBarPlot:
    def test_shares_lineplot_evaluate_and_only_overrides_the_artist(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(6)
        counts = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])[:, None]
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = BarPlot(counts, config, x=sample_axis, y=Y)
        assert isinstance(plot, LinePlot)
        result = plot.evaluate()
        assert np.allclose(result.y, counts[:, 0])
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "bar"

    def test_redraw_updates_values_without_duplicating_trace(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(4)
        data = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        fig = Figure(BarPlot(data, config, x=sample_axis, y=Y))
        backend_figure = fig.draw()
        fig.draw()
        assert len(backend_figure.data) == 1


class TestPathPlot:
    def test_evaluate_returns_ordered_positions(self):
        X = Variable("x", 1)
        Y = Variable("y", 1)
        sample_axis = BatchAxes(15)
        t = np.linspace(0, 6, 15)
        positions = np.stack([np.cos(t), np.sin(t)], axis=1)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = PathPlot(positions, config, position=VectorSpec(X * Y))
        result = plot.evaluate()
        assert result.positions.shape == (15, 2)
        assert plot.embedding_dim == 2

    def test_3d_path_draws_as_scatter3d(self):
        X, Y, Z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        sample_axis = BatchAxes(10)
        positions = np.random.rand(10, 3)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y * Z))
        plot = PathPlot(positions, config, position=VectorSpec(X * Y * Z))
        assert plot.embedding_dim == 3
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "scatter3d"

    def test_axis_titles_come_from_the_vectorspec_variable(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(5)
        positions = np.random.rand(5, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = PathPlot(positions, config, position=VectorSpec(X * Y))
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.xaxis.title.text == "$x$"
        assert backend_figure.layout.yaxis.title.text == "$y$"
