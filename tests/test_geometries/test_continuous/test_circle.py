import inspect
import math
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_2d.circle import Circle
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
X = Variable("x", 2)
devices = [cpu, cuda(0)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_circle(backend):
    circle = Circle(X, [0, 0], 1.5, backend=backend)
    assert backend.math.all(circle.center == backend.build_tensor([0.0, 0.0]))
    assert circle.radius == 1.5
    assert circle.variable == X
    assert circle.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    circle = Circle(X, [0, 0], 1.5, backend=backend)
    points = circle.sample_random_uniform(10, device=device)
    assert points.shape == (10, 2)
    for p in points:
        assert p[0] ** 2 + p[1] ** 2 <= 1.5**2
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains(backend):
    circle = Circle(X, [0, 0], 1.0, backend=backend)
    points = backend.build_tensor(
        [[0.5, 0.5], [1.0, 0.0], [1.1, 0.0], [0.0, 1.2], [0.0, 0.0]]
    )
    inside = circle.contains(points)
    expected = backend.build_tensor([[True], [True], [False], [False], [True]])
    assert backend.math.all(inside == expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid(backend, device):
    circle = Circle(X, [0, 0], 1.0, backend=backend)
    points = circle.sample_grid(10, device=device)
    assert points.shape == (10, 2)
    # Check if they are inside
    for p in points:
        assert float(p[0]) ** 2 + float(p[1]) ** 2 <= 1.00001


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box(backend):
    circle = Circle(X, [1.0, -1.0], 2.0, backend=backend)
    bbox = circle.bounding_box()
    expected = backend.build_tensor([-1.0, 3.0, -3.0, 1.0])
    assert backend.math.allclose(bbox, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_volume(backend):
    radius = 2.0
    circle = Circle(X, [0, 0], radius, backend=backend)
    assert circle.volume() == pytest.approx(math.pi * radius**2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_mesh(backend):
    circle = Circle(X, [0, 0], 1.0, backend=backend)
    mesh_geo = circle.create_mesh(max_vertex_distance=0.5)
    assert len(mesh_geo.mesh.vertices) > 0
    assert len(mesh_geo.mesh.cells) > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary(backend):
    circle = Circle(X, [0, 0], 1.0, backend=backend)
    boundary = circle.boundary
    assert boundary.volume() == pytest.approx(2 * math.pi * 1.0)

    points = backend.build_tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    inside = boundary.contains(points)
    expected = backend.build_tensor([[True], [True], [False]])
    assert backend.math.all(inside == expected)

    # Normal test
    test_points = backend.build_tensor([[1.0, 0.0], [0.0, -1.0]])
    normals = boundary.normal(test_points)
    expected_normals = backend.build_tensor([[1.0, 0.0], [0.0, -1.0]])
    assert backend.math.allclose(normals, expected_normals)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_sampling(backend, device):
    radius = 1.5
    circle = Circle(X, [0, 0], radius, backend=backend)
    boundary = circle.boundary

    # sampling random
    res = boundary.sample_random_uniform(10, device=device)
    assert res.shape == (10, 2)
    for p in res:
        dist = (float(p[0]) ** 2 + float(p[1]) ** 2) ** 0.5
        assert dist == pytest.approx(radius)

    # sampling grid
    points = boundary.sample_grid(10, device=device)
    assert points.shape == (10, 2)
    for p in points:
        dist = (float(p[0]) ** 2 + float(p[1]) ** 2) ** 0.5
        assert dist == pytest.approx(radius)

    # sampling with normals
    pts, ns = boundary.sample_random_uniform(5, device=device, include_normals=True)
    assert pts.shape == (5, 2)
    assert ns.shape == (5, 2)
