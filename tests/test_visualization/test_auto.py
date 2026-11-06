"""auto_plot beyond its matrix (test_matrix_auto.py): plot_type, passed
arguments, explicit controls and scales."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.visualization import (
    BarPlot,
    ColorScale,
    HeatmapPlot,
    LinePlot,
    Overlay,
    Row,
    ScatterPlot,
    Selector,
    SliderAxis,
    SurfacePlot,
    TimeAxis,
    VectorPlot,
    VectorScale,
    auto_layout,
    auto_plot,
)

T, M, X, Y = Variable("t", 1), Variable("m", 1), Variable("x", 2), Variable("y", 3)
U, P, V, W = Variable("u", 1), Variable("p", 1), Variable("v", 2), Variable("w", 3)


def _square_mesh():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)


def _data(*axes):
    config = DataConfiguration(*axes)
    return np.random.default_rng(0).random(config.shape), config


def _controls(plot):
    return [(type(c), c.variable_or_axes) for c in plot.controls]


class TestRules:
    """What the matrix (test_matrix_auto.py) does not cover."""

    def test_a_plot_type_restricts_the_choice(self):
        grid = GridGeometry.build_unit_grid(T, 5)
        plot = auto_plot(*_data(GeometryAxes(grid), FeatureAxes(U)), BarPlot)
        assert isinstance(plot, BarPlot)

    def test_a_plot_type_that_does_not_fit_raises(self):
        with pytest.raises(ValueError, match="LinePlot puts the domain on one axis, but it has 2 coordinates"):
            auto_plot(*_data(GeometryAxes(_square_mesh()), FeatureAxes(U)), LinePlot)

    def test_plot_arguments_are_passed_on(self):
        plot = auto_plot(
            *_data(GeometryAxes(_square_mesh()), FeatureAxes(U)), title="field"
        )
        assert plot.title == "field"

    def test_no_named_feature_variable_raises(self):
        with pytest.raises(ValueError, match="named Variable"):
            auto_plot(*_data(BatchAxes(3), FeatureAxes(shape=(2,))))


    def test_an_explicit_control_is_not_duplicated(self):
        batch = BatchAxes(3)
        time = TimeAxis(batch)
        plot = auto_plot(
            *_data(batch, GeometryAxes(_square_mesh()), FeatureAxes(U)), controls=[time]
        )
        assert plot.controls == [time]

class TestArguments:
    """Plot arguments take part in the choice; a mismatch names every
    plot type tried with its reason."""

    def test_a_height_makes_a_field_a_surface(self):
        plot = auto_plot(*_data(GeometryAxes(_square_mesh()), FeatureAxes(U)), z=U)
        assert isinstance(plot, SurfacePlot)

    def test_an_argument_no_plot_type_takes_names_every_reason(self):
        grid = GridGeometry.build_unit_grid(T, 5)
        with pytest.raises(ValueError) as caught:
            auto_plot(*_data(GeometryAxes(grid), FeatureAxes(U)), show_edges=True)
        message = str(caught.value)
        assert "on the geometry of t with the arguments show_edges" in message
        assert "LinePlot takes no show_edges" in message
        assert "HeatmapPlot draws 2 position components, not 1" in message


class TestScales:
    def test_a_color_scale_is_passed_on(self):
        scale = ColorScale(vmin=0.0)
        plot = auto_plot(
            *_data(GeometryAxes(_square_mesh()), FeatureAxes(U)), color_scale=scale
        )
        assert plot.color_scale is scale

    def test_a_vector_scale_is_passed_on(self):
        scale = VectorScale(length=0.1)
        plot = auto_plot(
            *_data(GeometryAxes(_square_mesh()), FeatureAxes(V)), vector_scale=scale
        )
        assert plot.vector_scale is scale

    def test_a_scale_for_a_channel_the_plot_does_not_bind_raises(self):
        with pytest.raises(ValueError, match="SurfacePlot takes no vector_scale"):
            auto_plot(
                *_data(GeometryAxes(_square_mesh()), FeatureAxes(U)),
                vector_scale=VectorScale(),
            )
