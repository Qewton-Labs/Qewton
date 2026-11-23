import inspect

import numpy as np
import pytest

from qewton.backends.base import ComputingBackend
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.config.variables import Variable
from qewton.config.devices import cpu, cuda, cuda_available


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_grid_geometry(backend: type[ComputingBackend]):
    T = Variable("t", 1)
    grid = backend.math.linspace(0, 2, 100).reshape(-1, 1)
    geom = GridGeometry(T, grid, backend=backend)
    assert geom.variable == T
    assert backend.math.allclose(geom.discretization_points, grid, atol=1.0e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gridgeometry_bounding_box_one_dimensional(backend: type[ComputingBackend]):
    T = Variable("x", 1)
    grid = backend.math.linspace(-1.0, 2.0, 4).reshape(-1, 1)
    geom = GridGeometry(T, grid, backend=backend)

    bb = geom.bounding_box()
    assert len(bb) == 2
    assert [float(bb[0]), float(bb[1])] == [-1.0, 2.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_gridgeometry_build_unit_grid_three_dimensional(backend: type[ComputingBackend]):
    T = Variable("x", 3)
    geom = GridGeometry.build_unit_grid(T, resolution=2, backend=backend)

    assert geom.discretization_points.shape == (2, 2, 2, 3)
    bb = [float(v) for v in geom.bounding_box()]
    assert bb == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_gridgeometry_sample_grid_from_discretization_returns_requested_number(
    backend: type[ComputingBackend],
):
    T = Variable("x", 2)
    axis = backend.math.linspace(0.0, 1.0, 2)
    grid = backend.math.stack(backend.math.meshgrid(axis, axis), axis=-1)
    geom = GridGeometry(T, grid, backend=backend)

    sampled = geom.sample_grid_from_discretization(3)
    assert sampled.shape[0] == 3
    assert bool(backend.math.all(backend.math.min(sampled, axis=0) >= 0.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gridgeometry_sample_grid_from_discretization_repeats_points_when_more_requested(
    backend: type[ComputingBackend],
):
    T = Variable("x", 2)
    axis = backend.math.linspace(0.0, 1.0, 2)
    grid = backend.math.stack(backend.math.meshgrid(axis, axis), axis=-1)
    geom = GridGeometry(T, grid, backend=backend)

    sampled = geom.sample_grid_from_discretization(6)
    assert sampled.shape[0] == 6
    assert bool(backend.math.all(backend.math.max(sampled, axis=0) <= 1.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gridgeometry_random_uniform_sampling_respects_filter(
    backend: type[ComputingBackend],
):
    T = Variable("x", 2)
    axis = backend.math.linspace(0.0, 1.0, 3)
    grid = backend.math.stack(backend.math.meshgrid(axis, axis), axis=-1)
    point_filter = backend.build_tensor(
        [
            [[True], [True], [False]],
            [[True], [True], [False]],
            [[False], [False], [False]],
        ]
    )
    geom = GridGeometry(T, grid, point_filter=point_filter, backend=backend)

    sampled = geom.sample_random_uniform_from_discretization(4)
    if sampled.ndim == 3:
        sampled = sampled[:, 0, :]

    assert sampled.shape[0] == 4
    assert bool(backend.math.all(sampled[..., 0] <= 0.5))
    assert bool(backend.math.all(sampled[..., 1] <= 0.5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gridgeometry_contains_identifies_point_membership(
    backend: type[ComputingBackend],
):
    T = Variable("x", 2)
    axis = backend.math.linspace(0.0, 1.0, 3)
    grid = backend.math.stack(backend.math.meshgrid(axis, axis), axis=-1)
    print(grid.shape, backend)
    geom = GridGeometry(T, grid, backend=backend)

    inside_point = geom.discretization_points.reshape(-1, 2)[1:2]
    outside_point = backend.build_tensor([[10.0, 10.0]])

    inside_mask = geom.contains(inside_point)
    outside_mask = geom.contains(outside_point)
    assert inside_mask.shape[0] == 1
    assert bool(inside_mask[0])
    assert not bool(outside_mask[0])


class TestFactors:
    @staticmethod
    def _grid(variable, indexing="ij"):
        lines = [np.linspace(0.0, 1.0, n) for n in (4, 3, 2)]
        points = np.stack(np.meshgrid(*lines, indexing=indexing), axis=-1)
        return GridGeometry(variable, points)

    def test_a_rectilinear_grid_is_the_product_of_its_axes(self):
        x = Variable("x", 3)
        grid = self._grid(x)
        assert [f.variable for f in grid.factors] == x.leaves
        assert [tuple(f.shape) for f in grid.factors] == [(4,), (3,), (2,)]

    def test_a_factor_takes_the_leaf_of_the_coordinate_along_its_axis(self):
        x = Variable("x", 3)
        grid = self._grid(x, indexing="xy")  # axis 0 runs along x_2
        assert [f.variable.name for f in grid.factors] == ["x_2", "x_1", "x_3"]

    def test_a_subproduct_is_the_grid_over_the_axes_of_its_factors(self):
        t, x = Variable("t", 1), Variable("x", 2)
        grid = self._grid(t * x)
        _, x_1, x_2 = grid.factors
        plane = grid.subproduct([x_1, x_2])
        assert plane.variable == x
        assert tuple(plane.shape) == (3, 2)
        assert plane.axis_coordinates()[1].tolist() == [0.0, 1.0]

    def test_a_subproduct_of_no_node_composes_the_leaves(self):
        x = Variable("x", 3)
        grid = self._grid(x)
        x_1, _, x_3 = grid.factors
        assert grid.subproduct([x_1, x_3]).variable == x.leaves[0] * x.leaves[2]

    def test_a_line_has_no_factors(self):
        assert self._grid(Variable("x", 3)).factors[0].factors is None

    def test_no_factors_for_a_rotated_grid(self):
        u = np.linspace(0.0, 1.0, 3)
        a, b = np.meshgrid(u, u, indexing="ij")
        points = np.stack([a + b, a - b], axis=-1)
        assert GridGeometry(Variable("x", 2), points).factors is None
