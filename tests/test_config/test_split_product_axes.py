"""Splitting the axes of a product geometry into axes of its factors, and
replacing axes of a DataConfiguration."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.product_mesh_geometry import (
    ProductDiscreteGeometry,
    ProductMeshGeometry,
)

T, M, X = Variable("t", 1), Variable("m", 1), Variable("x", 2)
U = Variable("u", 1)


def _time():
    return Interval(T, 0.0, 2.0)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


def _axes(geometry):
    return GeometryAxes(geometry.create_mesh(0.5))


def _variables(axes_list):
    return [a.geometry.variable for a in axes_list]


def _sizes(axes_list):
    return [tuple(d.size for d in a.shape) for a in axes_list]


class TestSplit:
    @pytest.mark.parametrize("time_first", [True, False])
    def test_a_named_factor_gets_axes_of_its_own(self, time_first):
        product = _time() * _square() if time_first else _square() * _time()
        split = _axes(product).split([T])
        expected = [T, X] if time_first else [X, T]
        assert _variables(split) == expected
        assert _sizes(split) == ([(5,), (9,)] if time_first else [(9,), (5,)])

    def test_the_points_are_ordered_first_factor_slowest(self):
        axes = _axes(_time() * _square())
        time, space = axes.split([T])
        points = np.asarray(axes.geometry.discretization_points).reshape(5, 9, 3)
        np.testing.assert_allclose(points[..., 0], np.repeat(time.geometry.coordinates(), 9, 1))
        np.testing.assert_allclose(points[2, :, 1:], space.geometry.coordinates())

    def test_several_named_factors(self):
        split = _axes(_time() * Interval(M, 1.0, 3.0) * _square()).split([T, M])
        assert _variables(split) == [T, M, X]

    def test_adjacent_other_factors_stay_together_as_a_product(self):
        split = _axes(_time() * Interval(M, 1.0, 3.0) * _square()).split([T])
        assert _variables(split)[0] == T
        merged = split[1].geometry
        assert isinstance(merged, ProductMeshGeometry)
        assert [f.variable for f in merged.factors] == [M, X]
        assert _sizes(split) == [(5,), (45,)]

    def test_other_factors_apart_from_each_other_stay_apart(self):
        split = _axes(_time() * Interval(M, 1.0, 3.0) * _square()).split([M])
        assert _variables(split) == [T, M, X]

    def test_point_set_factors_keep_their_shape(self):
        times = _time().discretize(np.array([[0.0], [1.0], [2.0]]))
        g1, g2 = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 5), indexing="ij")
        space = _square().discretize(np.stack([g1, g2], axis=-1))
        split = GeometryAxes(ProductDiscreteGeometry([times, space])).split([T])
        assert _variables(split) == [T, X]
        assert _sizes(split) == [(3,), (4, 5)]

    def test_a_variable_that_is_no_factor_raises(self):
        with pytest.raises(ValueError, match="no factor"):
            _axes(_time() * _square()).split([X[0]])

    def test_a_geometry_that_is_no_product_raises(self):
        with pytest.raises(ValueError, match="not a product"):
            _axes(_square()).split([X])


class TestReplace:
    def test_axes_are_replaced_in_place(self):
        product = _axes(_time() * _square())
        batch, features = BatchAxes(2), FeatureAxes(U)
        config = DataConfiguration(batch, product, features, dtype="float32")
        split = config.replace(product, product.split([T]))
        assert split.axes[0] is batch and split.axes[-1] is features
        assert split.shape == (2, 5, 9, 1)
        assert split.dtype == "float32"

    def test_the_number_of_elements_must_match(self):
        product = _axes(_time() * _square())
        config = DataConfiguration(product, FeatureAxes(U))
        with pytest.raises(ValueError, match="45 elements"):
            config.replace(product, [BatchAxes(44)])

    def test_axes_not_in_the_configuration_raise(self):
        config = DataConfiguration(BatchAxes(3), FeatureAxes(U))
        with pytest.raises(ValueError, match="not part"):
            config.replace(BatchAxes(3), [BatchAxes(3)])
