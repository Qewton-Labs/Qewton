import inspect
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
