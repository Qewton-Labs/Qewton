"""Data plots on product geometries: a factor named by a binding or a
control gets tensor axes of its own."""

import numpy as np
import pytest

from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.visualization import (
    Figure,
    HeatmapPlot,
    LinePlot,
    SliderAxis,
    TimeAxis,
    VectorPlot,
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
