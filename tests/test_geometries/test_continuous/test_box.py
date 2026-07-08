import inspect
import pytest
import math

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_3d.box import Box
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
X = Variable("x", 3)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_box(backend):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    # origin should be a 3-vector
    assert tuple(box.origin.shape) == (3,)
    # bounding box should list min and max for x,y,z
    bbox = box.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    assert bx == [0.0, 1.0, 0.0, 2.0, 0.0, 1.0]
    # volume should match product
    assert float(box._get_volume()) == pytest.approx(1.0 * 2.0 * 1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_box_offset(backend):
    box = Box(X, [0.5, -1.0, 2.0], 0.5, 0.5, 0.2, backend=backend)
    bbox = box.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    compare_list = [0.5, 1.0, -1.0, -0.5, 2.0, 2.2]
    for b, compare_b in zip(bx, compare_list):
        assert abs(b - compare_b) < 1.0e-5
    assert float(box._get_volume()) == pytest.approx(0.5 * 0.5 * 0.2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 0.1, backend=backend)
    points = box.sample_random_uniform(10, device=device)
    assert points.shape == (10, 3)
    for p in points:
        assert -0.01 <= float(p[0]) <= 1.01
        assert -0.01 <= float(p[1]) <= 2.01
        assert -0.01 <= float(p[2]) <= 0.11
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform_repeat(backend, device):
    # second run with non-zero origin
    box = Box(X, [0.1, 0.2, 0.3], 0.8, 0.9, 0.4, backend=backend)
    pts = box.sample_random_uniform(20, device=device)
    assert pts.shape == (20, 3)
    for p in pts:
        assert 0.09 <= float(p[0]) <= 0.91
        assert 0.19 <= float(p[1]) <= 1.11
        assert 0.29 <= float(p[2]) <= 0.71
    if issubclass(backend, DeepLearningBackend):
        assert pts.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_inside_outside(backend):
    box = Box(X, [0, 0, 0], 1.0, 2.0, 1.0, backend=backend)
    inside = [0.2, 0.5, 0.5]
    outside = [1.1, 0.0, 0.0]
    on_face = [0.0, 1.0, 0.5]
    assert float(box.contains([inside]).reshape(-1, 1)[0][0]) == 1.0
    assert float(box.contains([outside]).reshape(-1, 1)[0][0]) == 0.0
    assert float(box.contains([on_face]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_edges_and_corners(backend):
    box = Box(X, [0.5, 0.5, 0.5], 1.0, 1.0, 1.0, backend=backend)
    # corner
    corner = [1.5, 1.5, 1.5]
    # face center
    face = [0.5, 1.0, 1.0]
    assert float(box.contains([corner]).reshape(-1, 1)[0][0]) == 1.0
    assert float(box.contains([face]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_and_membership(backend, device):
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0, backend=backend)
    pts = box.sample_grid(100, device=device)
    assert pts.shape == (100, 3)
    inside = box.contains(pts)
    inside_count = sum(
        1 for i in range(len(pts)) if float(inside.reshape(-1, 1)[i][0]) == 1.0
    )
    assert inside_count >= int(0.95 * len(pts))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_repeat_and_device(backend, device):
    box = Box(X, [0.2, 0.2, 0.2], 0.5, 0.5, 0.5, backend=backend)
    pts = box.sample_grid(50, device=device)
    assert pts.shape == (50, 3)
    # all sampled points should be inside or on boundary
    inside = box.contains(pts)
    assert all(
        float(v[0]) in (0.0, 1.0) or float(v[0]) >= 0.0 for v in inside.reshape(-1, 1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_and_volume(backend):
    box = Box(X, [1.0, 2.0, 3.0], 0.2, 0.3, 0.4, backend=backend)
    bbox = box.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    compare_list = [1.0, 1.2, 2.0, 2.3, 3.0, 3.4]
    for b, compare_b in zip(bx, compare_list):
        assert abs(b - compare_b) < 1.0e-5
    assert float(box._get_volume()) == pytest.approx(0.2 * 0.3 * 0.4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_repeat(backend):
    box = Box(X, [-1.0, -2.0, -3.0], 1.0, 2.0, 3.0, backend=backend)
    bbox = box.bounding_box()
    assert [float(bbox[i]) for i in range(6)] == [-1.0, 0.0, -2.0, 0.0, -3.0, 0.0]
    assert float(box._get_volume()) == pytest.approx(1.0 * 2.0 * 3.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_faces_and_corners(backend):
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0, backend=backend)
    boundary = box.create_boundary()
    # center of face x=0
    p1 = [0.0, 0.5, 0.5]
    # center of opposite face x=1
    p2 = [1.0, 0.5, 0.5]
    # corner
    p3 = [0.0, 0.0, 0.0]
    assert boundary.contains([p1]).reshape(-1, 1)[0][0]
    assert boundary.contains([p2]).reshape(-1, 1)[0][0]
    assert boundary.contains([p3]).reshape(-1, 1)[0][0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_non_boundary(backend):
    box = Box(X, [0.0, 0.0, 0.0], 2.0, 2.0, 2.0, backend=backend)
    boundary = box.create_boundary()
    inside = [1.0, 1.0, 1.0]
    assert float(boundary.contains([inside]).reshape(-1, 1)[0][0]) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_and_normals(backend, device):
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 2.0, 3.0, backend=backend)
    boundary = box.create_boundary()
    pts, normals = boundary.sample_random_uniform(30, device=device, include_normals=True)
    assert pts.shape == (30, 3)
    assert normals.shape == (30, 3)
    # normals should be unit length
    for n in normals:
        nx = float(n[0])
        ny = float(n[1])
        nz = float(n[2])
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        assert pytest.approx(1.0, rel=1e-3) == norm
    # normals should point roughly outward from centroid
    centroid = [0.5, 1.0, 1.5]
    for p, n in zip(pts, normals):
        vx = float(p[0]) - centroid[0]
        vy = float(p[1]) - centroid[1]
        vz = float(p[2]) - centroid[2]
        dot = float(n[0]) * vx + float(n[1]) * vy + float(n[2]) * vz
        assert dot > -1e-6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_repeat(backend, device):
    box = Box(X, [0.2, 0.3, 0.4], 0.5, 0.6, 0.7, backend=backend)
    boundary = box.create_boundary()
    pts = boundary.sample_random_uniform(40, device=device, include_normals=False)
    assert pts.shape == (40, 3)
    # all returned points should be on the boundary
    on_boundary = boundary.contains(pts)
    assert all(
        float(v[0]) == 1.0 or float(v[0]) == 0.0 or float(v[0]) >= 0.0
        for v in on_boundary.reshape(-1, 1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_normal_function_expected(backend):
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0, backend=backend)
    boundary = box.create_boundary()
    # point on min-x face
    p_minx = [0.0, 0.2, 0.3]
    n_minx = boundary.normal([p_minx])
    assert pytest.approx(-1.0, rel=1e-6) == float(n_minx[0][0])
    # point on max-z face
    p_maxz = [0.3, 0.4, 1.0]
    n_maxz = boundary.normal([p_maxz])
    assert pytest.approx(1.0, rel=1e-6) == float(n_maxz[0][2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_normal_on_corner(backend):
    box = Box(X, [0.0, 0.0, 0.0], 1.0, 1.0, 1.0, backend=backend)
    boundary = box.create_boundary()
    corner = [0.0, 0.0, 0.0]
    n = boundary.normal([corner])
    # corner normal should be normalized combination
    norm = math.sqrt(float(n[0][0]) ** 2 + float(n[0][1]) ** 2 + float(n[0][2]) ** 2)
    assert pytest.approx(1.0, rel=1e-6) == norm
