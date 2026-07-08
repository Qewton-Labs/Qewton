import inspect
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_1d.interval import Interval
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
T = Variable("t", 1)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_interval(backend):
    interval = Interval(T, 0, 1, backend=backend)
    assert interval.lower_bound == 0
    assert interval.upper_bound == 1
    assert interval.variable == T
    assert interval.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    interval = Interval(T, 0, 1, backend=backend)
    points = interval.sample_random_uniform(10, device=device)
    assert points.shape == (10, 1)
    for p in points:
        assert p >= 0 and p <= 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains(backend):
    interval = Interval(T, 0, 1, backend=backend)
    points = backend.build_tensor([[0.5], [-0.1], [1.1], [0.0], [1.0]])
    inside = interval.contains(points)
    expected = backend.build_tensor([[True], [False], [False], [True], [True]])
    assert backend.math.all(inside == expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid(backend, device):
    interval = Interval(T, 0, 1, backend=backend)
    points = interval.sample_grid(3, device=device)
    assert points.shape == (3, 1)
    expected = backend.build_tensor([[0.25], [0.5], [0.75]])
    if issubclass(backend, DeepLearningBackend):
        expected = backend.to(expected, device=device)
    assert backend.math.allclose(points, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box(backend):
    interval = Interval(T, -2, 5, backend=backend)
    bbox = interval.bounding_box()
    assert backend.math.all(bbox == backend.build_tensor([-2.0, 5.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_volume(backend):
    interval = Interval(T, 0, 10, backend=backend)
    assert interval.volume() == 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_mesh(backend):
    interval = Interval(T, 0, 1, backend=backend)
    mesh_geo = interval.create_mesh(max_vertex_distance=0.5)
    # size=1, max_dist=0.5 -> ceil(1/0.5)=2 -> num_vertices = 1 + 2 = 3
    assert len(mesh_geo.mesh.vertices) == 3
    assert len(mesh_geo.mesh.cells) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary(backend):
    interval = Interval(T, 0, 1, backend=backend)
    boundary = interval.boundary
    assert boundary.volume() == 2

    points = backend.build_tensor([[0.0], [1.0], [0.5]])
    inside = boundary.contains(points)
    expected = backend.build_tensor([[True], [True], [False]])
    assert backend.math.all(inside == expected)

    # Normal test
    normals = boundary.normal(backend.build_tensor([[0.0], [1.0]]))
    expected_normals = backend.build_tensor([[-1.0], [1.0]])
    assert backend.math.all(normals == expected_normals)


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_boundary_points(backend):
    interval = Interval(T, 0, 1, backend=backend)
    lb = interval.boundary_left
    rb = interval.boundary_right

    assert lb.volume() == 1
    assert rb.volume() == 1

    normals_lb = lb.normal(backend.build_tensor([[0.0]]))
    assert backend.math.all(normals_lb == backend.build_tensor([[-1.0]]))

    normals_rb = rb.normal(backend.build_tensor([[1.0]]))
    assert backend.math.all(normals_rb == backend.build_tensor([[1.0]]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_sampling(backend, device):
    interval = Interval(T, 0, 1, backend=backend)
    boundary = interval.boundary

    # sampling random
    res = boundary.sample_random_uniform(10, device=device)
    assert res.shape == (10, 1)
    for p in res[:, 0]:
        assert float(p) == 0.0 or float(p) == 1.0

    # sampling grid
    points = boundary.sample_grid(2, device=device)
    assert points.shape == (2, 1)
    p_vals = [float(p) for p in points[:, 0]]
    assert 0.0 in p_vals and 1.0 in p_vals


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_mesh(backend):
    interval = Interval(T, 0, 1, backend=backend)
    boundary = interval.boundary

    # create_mesh
    mesh_geo = boundary.create_mesh()
    assert len(mesh_geo.mesh.vertices) == 2
    assert len(mesh_geo.mesh.cells) == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_single_boundary_point_logic(backend, device):
    interval = Interval(T, 0, 1, backend=backend)
    lb = interval.boundary_left

    # contains
    points = backend.build_tensor([[0.0], [1.0], [0.5]])
    inside = lb.contains(points)
    expected = backend.build_tensor([[True], [False], [False]])
    assert backend.math.all(inside == expected)

    # sampling with normals
    pts, ns = lb.sample_random_uniform(5, device=device, include_normals=True)
    assert pts.shape == (5, 1)
    assert ns.shape == (5, 1)
    assert backend.math.all(pts == 0.0)
    assert backend.math.all(ns == -1.0)
