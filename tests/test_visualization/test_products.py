"""Data plots on product geometries: a factor named by a control gets
tensor axes of its own."""

import numpy as np
import pytest

from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    Figure,
    HeatmapPlot,
    LinePlot,
    SliderAxis,
    SurfacePlot,
    TimeAxis,
    VectorPlot,
    auto_plot,
)

from recording_renderer import RecordingRenderer

T, X = Variable("t", 1), Variable("x", 2)
U, V = Variable("u", 1), Variable("v", 2)


def _product(time_first=True):
    time, square = Interval(T, 0.0, 2.0), Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    return (time * square if time_first else square * time).create_mesh(0.5)


def _field(points):
    # A value per point that tells its time and place apart.
    return points @ np.array([100.0, 10.0, 1.0])[: points.shape[1]]


def _on(geometry, time_column=0):
    points = np.asarray(geometry.discretization_points)
    t = points[:, time_column]
    xy = np.delete(points, time_column, axis=1)
    data = np.stack([_field(np.column_stack([t, xy])), *xy.T], axis=-1)
    config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U * V))
    return data, config


class TestSplitInPlots:
    @pytest.mark.parametrize("time_first", [True, False])
    def test_a_slider_on_time_steps_through_the_time_factor(self, time_first):
        geometry = _product(time_first)
        data, config = _on(geometry, 0 if time_first else 2)
        time = SliderAxis(T)
        plot = HeatmapPlot(data, config, position=X, color=U, controls=[time])
        assert plot.domain.variable == X
        times = geometry.factors[0 if time_first else 1].coordinates()[:, 0]
        for state, t in enumerate(times):
            time.state = state
            (result,) = plot.evaluate()
            expected = _field(np.column_stack([np.full(9, t), result.positions]))
            np.testing.assert_allclose(result.values, expected)

    def test_a_binding_names_a_factor_too(self):
        data, config = _on(_product())
        plot = LinePlot(data, config, x=T, y=U, controls=[SliderAxis(X)])
        (result,) = plot.evaluate()
        assert result.positions.shape == (5, 2)

    def test_other_axes_keep_their_place(self):
        data, config = _on(_product())
        runs = BatchAxes(2)
        config = DataConfiguration(runs, *config.axes)
        data = np.stack([data, 2 * data])
        plot = HeatmapPlot(
            data, config, position=X, color=U, controls=[SliderAxis(runs), TimeAxis(T)]
        )
        assert plot.data.shape == (2, 5, 9, 3)
        np.testing.assert_allclose(plot.data[1], 2 * plot.data[0])

    def test_a_vector_with_fewer_components_than_the_product(self):
        data, config = _on(_product())
        plot = VectorPlot(data, config, position=X, vector=V, controls=[SliderAxis(T)])
        (result,) = plot.evaluate()
        assert result.vectors.shape == (9, 2)

    def test_without_a_named_factor_the_data_is_unchanged(self):
        geometry = Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        data = np.zeros(config.shape)
        plot = HeatmapPlot(data, config, position=X, color=U)
        assert plot.data is data and plot.data_config is config

    def test_an_unsplit_product_is_not_drawn_as_a_heatmap(self):
        data, config = _on(_product())
        with pytest.raises(ValueError):
            HeatmapPlot(data, config, position=X, color=U)


def test_data_of_another_shape_than_its_configuration_raises():
    data, config = _on(_product())
    with pytest.raises(ValueError, match=r"shape \(44, 3\)"):
        HeatmapPlot(data[:44], config, position=X, color=U, controls=[SliderAxis(T)])


def test_a_sampler_on_a_product_in_mesh_mode():
    sampler = GridSampler(Interval(T, 0.0, 1.0) * Rectangle(X, [0.0, 0.0], 1.0, 1.0), 50)
    with discretization_mode([sampler], 0.5, "cpu"):
        points = sampler.forward()
    sampler.sampled_geometry.to_numpy()
    points = np.asarray(sampler.backend.to_numpy(points))
    # The axes size of the points in mesh mode, as the graph resolves it.
    axes = GeometryAxes(sampler.sampled_geometry, (AxesDim(len(points)),))
    config = DataConfiguration(axes, FeatureAxes(U))
    data = _field(points)[:, None]
    time = TimeAxis(T)
    plot = HeatmapPlot(data, config, position=X, color=U, controls=[time])
    n_times = len(sampler.sampled_geometry.factors[0].coordinates())
    assert plot.data.shape == (n_times, len(points) // n_times, 1)
    backend = Figure(plot, renderer=RecordingRenderer()).draw(embedded=[time])
    assert len(backend["embedded"][id(time)]) == n_times


class TestGridAxes:
    """A rectilinear grid is the product of its axes: a control steps
    through one axis, the others are drawn."""

    @staticmethod
    def _on_grid(shape=(4, 3, 2)):
        x = Variable("x", len(shape))
        lines = [np.linspace(0.0, 1.0, n) for n in shape]
        points = np.stack(np.meshgrid(*lines, indexing="ij"), axis=-1)
        geometry = GridGeometry(x, points)
        data = _field(points.reshape(-1, len(shape))).reshape(*shape, 1)
        return data, DataConfiguration(GeometryAxes(geometry), FeatureAxes(U)), x

    def test_a_slider_steps_through_one_axis(self):
        data, config, x = self._on_grid()
        x_1, x_2, x_3 = x.leaves
        layer = SliderAxis(x_3, init_state=1)
        plot = HeatmapPlot(data, config, position=(x_1, x_2), color=U, controls=[layer])
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.values, data[:, :, 1, 0].T)
        np.testing.assert_allclose(layer.coordinates, [0.0, 1.0])

    def test_positions_take_any_order(self):
        data, config, x = self._on_grid()
        x_1, x_2, x_3 = x.leaves
        plot = HeatmapPlot(data, config, x=x_2, y=x_1, color=U, controls=[SliderAxis(x_3)])
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.values, data[:, :, 0, 0])
        assert plot.scaled_channels == ("x", "y")

    def test_a_line_through_the_grid(self):
        data, config, x = self._on_grid()
        x_1, x_2, x_3 = x.leaves
        plot = LinePlot(
            data, config, x=x_2, y=U, controls=[SliderAxis(x_1, 3), SliderAxis(x_3, 1)]
        )
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.positions[:, 1], data[3, :, 1, 0])

    def test_auto_plot_draws_the_axes_not_controlled(self):
        data, config, x = self._on_grid()
        plot = auto_plot(data, config, controls=[SliderAxis(x.leaves[0])])
        assert isinstance(plot, HeatmapPlot)
        assert tuple(plot.domain.shape) == (3, 2)

    def test_a_slider_through_the_middle_axis_draws_the_outer_ones(self):
        data, config, x = self._on_grid()
        x_1, x_2, x_3 = x.leaves
        plot = HeatmapPlot(
            data, config, position=(x_1, x_3), color=U, controls=[SliderAxis(x_2, 2)]
        )
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.values, data[:, 2, :, 0].T)
        assert isinstance(auto_plot(data, config, controls=[SliderAxis(x_2)]), HeatmapPlot)

    def test_a_controlled_axis_bound_as_position_moves_the_slice_through_space(self):
        data, config, x = self._on_grid()
        x_1, x_2, x_3 = x.leaves
        layer = SliderAxis(x_2, init_state=1)
        plot = SurfacePlot(data, config, position=(x_1, x_2, x_3), color=U, controls=[layer])
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.positions[:, 1], 0.5)
        np.testing.assert_allclose(result.values, data[:, 1, :, 0].ravel())
        assert plot.scaled_channels == ("x", "y", "z")

    def test_auto_plot_moves_a_surface_through_space(self):
        data, config, x = self._on_grid()
        plot = auto_plot(data, config, plot_type=SurfacePlot, controls=[SliderAxis(x.leaves[2])])
        assert plot.n_positions == 3

    def test_auto_plot_names_the_missing_control_of_a_slice_in_space(self):
        data, config, _ = self._on_grid((4, 3))
        with pytest.raises(ValueError, match="no control is on one coordinate"):
            auto_plot(data, config, plot_type=SurfacePlot)

    def test_a_grid_without_controls_is_drawn_whole(self):
        data, config, _ = self._on_grid()
        assert isinstance(auto_plot(data, config), SurfacePlot)

