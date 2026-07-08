import inspect
import math
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domain_operations.intersection import (
    IntersectionGeometry,
)
from qewton.geometries.continuous.domains_2d.parallelogram import Parallelogram
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.continuous.domains_3d.sphere import Sphere
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
Y = Variable("y", 2)
X = Variable("x", 3)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_intersect_domain_2d(backend):
    parallel = Parallelogram(Y, [0, 0], [1, 0], [0, 1], backend=backend)
    circle = Circle(Y, [0, 0], 0.5, backend=backend)
    intersect_domain = parallel & circle

    assert isinstance(intersect_domain, IntersectionGeometry)
    assert intersect_domain.variable == Y
    assert intersect_domain.backend == backend
    assert intersect_domain.geometry_a == parallel
    assert intersect_domain.geometry_b == circle


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_intersect_domain_3d(backend):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    sphere = Sphere(X, [0, 0, 0], 0.5, backend=backend)
    intersect_domain = box & sphere

    assert isinstance(intersect_domain, IntersectionGeometry)
    assert intersect_domain.variable == X
    assert intersect_domain.backend == backend
    assert intersect_domain.geometry_a == box
    assert intersect_domain.geometry_b == sphere


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_intersect_domain_2d(backend):
    parallel = Parallelogram(Y, [0, 0], [1, 0], [0, 1], backend=backend)
    circle = Circle(Y, [0, 0], 0.5, backend=backend)
    intersect_domain = parallel & circle

    inside = intersect_domain.contains([[0.1, 0.2]])
    assert backend.math.all(inside)

    outside = intersect_domain.contains([[1.1, 0.5], [-0.2, -0.2]])
    expected = backend.build_tensor([[False], [False]])
    assert backend.math.all(outside == expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_intersect_domain_3d(backend):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    sphere = Sphere(X, [0, 0, 0], 0.5, backend=backend)
    intersect_domain = box & sphere

    inside = intersect_domain.contains([[0.0, 0.25, 0.2]])
    assert backend.math.all(inside)

    outside = intersect_domain.contains([[1.1, 0.5, 0.5], [-0.2, 0.1, 0.1]])
    expected = backend.build_tensor([[False], [False]])
    assert backend.math.all(outside == expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform_intersect_2d(backend, device):
    parallel = Parallelogram(Y, [0, 0], [1, 0], [0, 1], backend=backend)
    circle = Circle(Y, [0, 0], 0.5, backend=backend)
    intersect_domain = parallel & circle

    points = intersect_domain.sample_random_uniform(20, device=device)
    assert points.shape == (20, 2)
    assert backend.math.all(intersect_domain.contains(points))
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform_intersect_3d(backend, device):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    sphere = Sphere(X, [0, 0, 0], 0.5, backend=backend)
    intersect_domain = box & sphere

    points = intersect_domain.sample_random_uniform(20, device=device)
    assert points.shape == (20, 3)
    assert backend.math.all(intersect_domain.contains(points))
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_intersect_2d(backend, device):
    parallel = Parallelogram(Y, [0, 0], [1, 0], [0, 1], backend=backend)
    circle = Circle(Y, [0, 0], 0.5, backend=backend)
    intersect_domain = parallel & circle

    pts = intersect_domain.sample_grid(16, device=device)
    assert pts.shape == (16, 2)
    assert backend.math.all(intersect_domain.contains(pts))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_intersect_3d(backend, device):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    sphere = Sphere(X, [0, 0, 0], 0.5, backend=backend)
    intersect_domain = box & sphere

    pts = intersect_domain.sample_grid(24, device=device)
    assert pts.shape == (24, 3)
    assert backend.math.all(intersect_domain.contains(pts))


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_intersect_2d(backend):
    parallel = Parallelogram(Y, [0, 0], [1, 0], [0, 1], backend=backend)
    circle = Circle(Y, [0, 0], 0.5, backend=backend)
    boundary = (parallel & circle).boundary

    assert backend.math.all(boundary.contains([[0.0, 0.5], [0.4, 0.0]]))
    assert float(boundary.contains([[0.2, 0.2]]).reshape(-1, 1)[0][0]) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_intersect_3d(backend):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    sphere = Sphere(X, [0, 0, 0], 0.5, backend=backend)
    boundary = (box & sphere).boundary

    assert backend.math.all(boundary.contains([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]))
    assert float(boundary.contains([[0.2, 0.1, 0.1]]).reshape(-1, 1)[0][0]) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_intersect_2d(backend, device):
    parallel = Parallelogram(Y, [0, 0], [1, 0], [0, 1], backend=backend)
    circle = Circle(Y, [0, 0], 0.5, backend=backend)
    boundary = (parallel & circle).boundary

    pts, normals = boundary.sample_random_uniform(10, device=device, include_normals=True)
    assert pts.shape == (10, 2)
    assert normals.shape == (10, 2)
    for n in normals:
        length = math.sqrt(float(n[0]) ** 2 + float(n[1]) ** 2)
        assert pytest.approx(1.0, rel=1e-3) == length


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_intersect_3d(backend, device):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    sphere = Sphere(X, [0, 0, 0], 0.5, backend=backend)
    boundary = (box & sphere).boundary

    pts, normals = boundary.sample_random_uniform(12, device=device, include_normals=True)
    assert pts.shape == (12, 3)
    assert normals.shape == (12, 3)
    for n in normals:
        length = math.sqrt(float(n[0]) ** 2 + float(n[1]) ** 2 + float(n[2]) ** 2)
        assert pytest.approx(1.0, rel=1e-3) == length
