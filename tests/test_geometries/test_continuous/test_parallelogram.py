import inspect
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_2d.parallelogram import Parallelogram
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
X = Variable("x", 2)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_parallelogram(backend):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 5.0], backend=backend)
    assert backend.math.all(para.origin == backend.build_tensor([0.0, 0.0]))
    assert backend.math.all(para.corner_1 == backend.build_tensor([1.0, 0.0]))
    assert backend.math.all(para.corner_2 == backend.build_tensor([0.0, 5.0]))
    assert para.variable == X
    assert para.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 5.0], backend=backend)
    points = para.sample_random_uniform(10, device=device)
    assert points.shape == (10, 2)
    for p in points:
        assert 0.0 <= float(p[0]) <= 1.0
        assert 0.0 <= float(p[1]) <= 5.0
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains(backend):
    # Origin [0,0], Corner 1 [2,0], Corner 2 [1,1]
    para = Parallelogram(X, [0, 0], [2.0, 0.0], [1.0, 1.0], backend=backend)
    points = backend.build_tensor(
        [[1.0, 0.5], [0.5, 0.5], [2.5, 0.5], [1.0, 1.1], [-0.1, 0.0], [3.0, 1.0]]
    )
    inside = para.contains(points)
    expected = backend.build_tensor([[True], [True], [True], [False], [False], [True]])
    assert backend.math.all(inside == expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box(backend):
    para = Parallelogram(X, [1.0, 1.0], [3.0, 1.0], [1.0, 4.0], backend=backend)
    bbox = para.bounding_box()
    expected = backend.build_tensor([1.0, 3.0, 1.0, 4.0])
    assert backend.math.allclose(bbox, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_volume(backend):
    para = Parallelogram(X, [0, 0], [2.0, 0.0], [0.0, 3.0], backend=backend)
    assert para.volume() == pytest.approx(6.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_mesh(backend):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    mesh_geo = para.create_mesh(max_vertex_distance=0.6)
    assert len(mesh_geo.mesh.vertices) == 9
    assert len(mesh_geo.mesh.cells) == 8


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_create_mesh_on_device(backend, device):
    """Regression: create_mesh() used to build its internal linspace/
    triangle-index tensors on cpu regardless of `device`, mismatching the
    already-moved origin/corner_1/corner_2 the moment a non-cpu device was
    requested."""
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    mesh_geo = para.create_mesh(max_vertex_distance=0.6, device=device)
    assert len(mesh_geo.mesh.vertices) == 9
    assert len(mesh_geo.mesh.cells) == 8


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid(backend, device):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    points = para.sample_grid(4, device=device)
    assert points.shape == (4, 2)
    for p in points:
        assert float(p[0]) in [0.0, 1.0]
        assert float(p[1]) in [0.0, 1.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary(backend):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    boundary = para.boundary
    assert boundary.volume() == pytest.approx(4.0)

    points = backend.build_tensor(
        [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.5, 0.5]]
    )
    inside = boundary.contains(points)
    expected = backend.build_tensor([[True], [True], [True], [True], [False]])
    assert backend.math.all(inside == expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_sampling(backend, device):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    boundary = para.boundary

    pts = boundary.sample_random_uniform(20, device=device)
    assert pts.shape == (20, 2)
    # check points on boundary
    assert backend.math.all(boundary.contains(pts))

    pts_grid = boundary.sample_grid(4, device=device)
    assert pts_grid.shape == (4, 2)

    pts, ns = boundary.sample_random_uniform(5, device=device, include_normals=True)
    assert pts.shape == (5, 2)
    assert ns.shape == (5, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    para = Parallelogram(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    boundary = para.boundary

    # Normals: [0,-1], [1,0], [0,1], [-1,0] for bottom, right, top, left edges
    test_pts = backend.build_tensor([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]])
    ns = boundary.normal(test_pts)
    expected = backend.build_tensor([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert backend.math.allclose(ns, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_degenerate(backend):
    # Collinear points for origin and corners
    with pytest.raises(ValueError, match="collinear"):
        para = Parallelogram(X, [0, 0], [1, 0], [2, 0], backend=backend)
        # contains calls solve_barycentric which raises the error
        para.contains(backend.build_tensor([[0.5, 0.0]]))
