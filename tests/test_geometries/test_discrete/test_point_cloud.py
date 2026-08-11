import inspect
import pytest

from qewton.backends.base import ComputingBackend
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.config.variables import Variable
from qewton.config.devices import cpu, cuda


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)
devices = [cpu, cuda(0)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_point_cloud(backend: type[ComputingBackend]):
    T = Variable("t", 1)
    points = backend.build_tensor([[1.0], [0.0], [2.0]])
    pc = PointCloud(T, points, backend=backend)
    assert pc.variable == T
    assert backend.math.all(pc.discretization_points == points)


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_cloud_move_points_preserves_point_values(backend: type[ComputingBackend]):
    T = Variable("x", 2)
    points = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    pc = PointCloud(T, points, backend=backend)

    pc._move_points(cpu)

    assert pc.discretization_points.shape == points.shape
    assert float(pc.discretization_points[0][0]) == 1.0
    assert float(pc.discretization_points[1][1]) == 4.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_cloud_bounding_box_and_volume(backend: type[ComputingBackend]):
    T = Variable("x", 2)
    points = backend.build_tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 5.0]])
    pc = PointCloud(T, points, backend=backend)

    bounding_box = pc.bounding_box()

    assert float(bounding_box[0]) == 0.0
    assert float(bounding_box[1]) == 3.0
    assert float(bounding_box[2]) == 2.0
    assert float(bounding_box[3]) == 5.0
    assert float(pc._get_volume()) == 9.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_cloud_create_boundary_raises(backend: type[ComputingBackend]):
    T = Variable("x", 2)
    points = backend.build_tensor([[1.0, 2.0]])
    pc = PointCloud(T, points, backend=backend)

    with pytest.raises(NotImplementedError):
        pc.create_boundary()


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_cloud_sample_random_uniform_from_discretization(
    backend: type[ComputingBackend],
):
    T = Variable("x", 1)
    points = backend.build_tensor([[1.0], [2.0], [3.0]])
    pc = PointCloud(T, points, backend=backend)

    sampled = pc.sample_random_uniform_from_discretization(2)
    assert len(sampled) == 2
    assert float(sampled[0][0]) in (1.0, 2.0, 3.0)
    assert float(sampled[1][0]) in (1.0, 2.0, 3.0)

    sampled_alias = pc.sample_random_uniform(2)
    assert len(sampled_alias) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_cloud_sample_grid_from_discretization_with_more_points(
    backend: type[ComputingBackend],
):
    T = Variable("x", 1)
    points = backend.build_tensor([[1.0], [2.0], [3.0]])
    pc = PointCloud(T, points, backend=backend)

    grid_points = pc.sample_grid_from_discretization(5)
    assert len(grid_points) == 5
    assert float(grid_points[0][0]) == 1.0
    assert float(grid_points[1][0]) == 2.0
    assert float(grid_points[2][0]) == 3.0
    assert all(float(grid_points[i][0]) in (1.0, 2.0, 3.0) for i in range(5))

    grid_alias = pc.sample_grid(5)
    assert len(grid_alias) == 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_cloud_contains_identifies_membership(backend: type[ComputingBackend]):
    T = Variable("x", 2)
    points = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    pc = PointCloud(T, points, backend=backend)

    contained_points = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    outside_points = backend.build_tensor([[0.0, 0.0], [3.0, 4.0]])

    contained_result = pc.contains(contained_points)
    assert float(contained_result[0][0]) == 1.0
    assert float(contained_result[1][0]) == 1.0

    outside_result = pc.contains(outside_points)
    assert float(outside_result[0][0]) == 0.0
    assert float(outside_result[1][0]) == 1.0
