import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.product_mesh_geometry import ProductMeshGeometry
from qewton.visualization.auto import auto_plot
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.spec import FixedSpec, SliderSpec, TimeSpec
from qewton.visualization.product_view import product_view

T, M = Variable("t", 1), Variable("m", 1)
X, Y = Variable("x", 2), Variable("y", 3)
U = Variable("u", 1)


def _field(points: np.ndarray) -> np.ndarray:
    """A value that identifies each point, to check the restructuring."""
    weights = np.arange(1, points.shape[-1] + 1) * 10.0 ** np.arange(points.shape[-1])
    return (points * weights).sum(axis=-1, keepdims=True)


def _on(geometry, max_vertex_distance=0.5):
    mesh = geometry.create_mesh(max_vertex_distance)
    points = _points(mesh)
    config = DataConfiguration(GeometryAxes(mesh), FeatureAxes(U))
    return _field(points), config


def _points(geometry) -> np.ndarray:
    if isinstance(geometry, ProductMeshGeometry):
        return np.asarray(geometry.discretization_points)
    return np.asarray(geometry.mesh.vertices)


def _time():
    return Interval(T, 0.0, 2.0)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


class TestSplitByControls:
    @pytest.mark.parametrize("time_first", [True, False])
    def test_a_control_on_a_factor_gives_it_an_axis(self, time_first):
        geometry = _time() * _square() if time_first else _square() * _time()
        data, config = _on(geometry)
        data, config, _ = product_view(data, config, [TimeSpec(T)])

        factor_axes, geometry_axes, _ = config.axes
        assert isinstance(factor_axes, GeometryAxes)
        assert factor_axes.geometry.variable == T
        assert geometry_axes.geometry.variable == X
        assert data.shape == (5, 9, 1)

        times = np.asarray(factor_axes.geometry.mesh.vertices)
        space = _points(geometry_axes.geometry)
        for i, t in enumerate(times):
            coords = (
                np.hstack([np.full((9, 1), t), space])
                if time_first
                else np.hstack([space, np.full((9, 1), t)])
            )
            assert np.allclose(data[i], _field(coords))

    def test_several_controls(self):
        data, config = _on(_time() * Interval(M, 1.0, 3.0) * _square())
        data, config, _ = product_view(data, config, [TimeSpec(T), SliderSpec(M)])
        assert [type(a) for a in config.axes] == [
            GeometryAxes,
            GeometryAxes,
            GeometryAxes,
            FeatureAxes,
        ]
        assert data.shape == (5, 5, 9, 1)

    def test_remaining_factors_are_drawn_as_a_product(self):
        data, config = _on(_time() * Interval(M, 1.0, 3.0) * _square())
        data, config, _ = product_view(data, config, [TimeSpec(T)])
        drawn = config.axes[1].geometry
        assert isinstance(drawn, ProductMeshGeometry)
        assert [f.variable for f in drawn.factors] == [M, X]
        assert data.shape == (5, 45, 1)

    def test_a_single_bound_control_is_returned_as_a_list(self):
        control = TimeSpec(T)
        data, config = _on(_time() * _square())
        _, _, controls = product_view(data, config, control)
        assert controls == [control]

    def test_other_axes_keep_their_place(self):
        data, config = _on(_time() * _square())
        data = np.stack([data, 2 * data])
        config = DataConfiguration(BatchAxes(2), *config.axes)
        data, config, _ = product_view(data, config, [TimeSpec(T)])
        assert [type(a) for a in config.axes] == [
            BatchAxes,
            GeometryAxes,
            GeometryAxes,
            FeatureAxes,
        ]
        assert data.shape == (2, 5, 9, 1)
        assert np.allclose(data[1], 2 * data[0])


class TestWithoutSplit:
    def test_a_drawable_product_is_left_unchanged(self):
        data, config = _on(_time() * _square())
        new_data, new_config, _ = product_view(data, config)
        assert new_data is data
        assert new_config is config

    def test_non_product_data_is_left_unchanged(self):
        data, config = _on(_square())
        new_data, new_config, _ = product_view(data, config, [TimeSpec(T)])
        assert new_config is config
        assert new_data is data

    def test_unrelated_single_control_is_not_wrapped(self):
        control = FixedSpec(init_state=0)
        data, config = _on(_time() * _square())
        _, _, controls = product_view(data, config, control)
        assert controls is control


class TestDefaultSplit:
    def test_the_one_dimensional_factor_is_split_beyond_3d(self):
        data, config = _on(_time() * Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0))
        _, config, controls = product_view(data, config)
        assert config.axes[0].geometry.variable == T
        assert config.axes[1].geometry.variable == Y
        assert [type(c) for c in controls] == [SliderSpec]
        assert controls[0].variable_or_axes == T

    def test_a_control_class_is_used_for_the_default_control(self):
        data, config = _on(_time() * Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0))
        _, _, controls = product_view(data, config, TimeSpec)
        assert [type(c) for c in controls] == [TimeSpec]
        assert controls[0].variable_or_axes == T

    def test_ambiguous_factors_raise(self):
        data, config = _on(_time() * Interval(M, 1.0, 3.0) * _square())
        with pytest.raises(ValueError, match=r"Pass a control for 1 of"):
            product_view(data, config)

    def test_too_few_one_dimensional_factors_raise(self):
        data, config = _on(_square() * Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0), 1.0)
        with pytest.raises(ValueError, match="too few"):
            product_view(data, config)


class TestInvalidControls:
    def test_control_on_a_multi_dimensional_factor_raises(self):
        data, config = _on(_time() * _square())
        with pytest.raises(ValueError, match="1-dimensional factor"):
            product_view(data, config, [SliderSpec(X)])

    def test_control_on_a_component_of_a_factor_raises(self):
        data, config = _on(_time() * _square())
        with pytest.raises(ValueError, match="only a component"):
            product_view(data, config, [SliderSpec(X[0])])

    def test_nothing_left_to_draw_raises(self):
        data, config = _on(_time() * Interval(M, 1.0, 3.0))
        with pytest.raises(ValueError, match="nothing is left to draw"):
            product_view(data, config, [TimeSpec(T), SliderSpec(M)])


class TestSeveralGeometryAxes:
    def _two_axes(self):
        time = MeshGeometry(T, _time().create_mesh(0.5).mesh)
        square = MeshGeometry(X, _square().create_mesh(0.5).mesh)
        data = np.random.rand(5, 9, 1)
        config = DataConfiguration(GeometryAxes(time), GeometryAxes(square), FeatureAxes(U))
        return data, config

    def test_are_merged_without_controls(self):
        data, config = self._two_axes()
        new_data, new_config, _ = product_view(data, config)
        assert isinstance(new_config.geometry_axes.geometry, ProductMeshGeometry)
        assert np.allclose(new_data, data.reshape(45, 1))

    def test_a_control_splits_its_factor(self):
        data, config = self._two_axes()
        new_data, new_config, _ = product_view(data, config, [TimeSpec(T)])
        assert new_config.axes[0].geometry.variable == T
        assert np.allclose(new_data, data)

    def test_a_control_bound_to_one_of_the_axes_keeps_it(self):
        data, config = self._two_axes()
        time_axes = config.axes[0]
        _, new_config, _ = product_view(data, config, [SliderSpec(time_axes)])
        assert new_config.axes[0] is time_axes


class TestAutoPlot:
    def test_control_on_time_gives_an_animated_mesh_plot(self):
        data, config = _on(_time() * _square())
        plot = auto_plot(data, config, controls=[TimeSpec(T)])
        assert isinstance(plot, MeshFieldPlot)
        assert len(plot.controls) == 1
        assert plot.controls[0].values == list(range(5))

    def test_default_control_is_added_for_a_split_factor(self):
        data, config = _on(_time() * Box(Y, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0))
        plot = auto_plot(data, config)
        assert [type(c) for c in plot.controls] == [SliderSpec]
        assert plot.controls[0].name == "t"

    def test_time_control_resolves_to_the_time_axes_when_space_comes_first(self):
        data, config = _on(_square() * _time())
        plot = auto_plot(data, config, controls=[TimeSpec(T)])
        assert plot.drawn_geometry_axes.geometry.variable == X
        [(control, dim)] = plot._resolve_controls()
        assert plot.data_config.axes[dim].geometry.variable == T
