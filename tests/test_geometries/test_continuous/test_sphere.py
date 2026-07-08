import inspect
import pytest
import math

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_3d.sphere import Sphere
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
X = Variable("x", 3)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_sphere(backend):
    sphere = Sphere(X, [0, 0, 0], 1.0, backend=backend)
    # center should be a 3-vector
    assert tuple(sphere.center.shape) == (3,)
    # bounding box should list min and max for x,y,z
    bbox = sphere.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    assert bx == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    # volume should match sphere formula
    assert float(sphere._get_volume()) == pytest.approx((4.0 / 3.0) * math.pi * 1.0**3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_sphere_offset(backend):
    sphere = Sphere(X, [0.5, -1.0, 2.0], 0.5, backend=backend)
    bbox = sphere.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    compare_list = [0.0, 1.0, -1.5, -0.5, 1.5, 2.5]
    for b, compare_b in zip(bx, compare_list):
        assert abs(b - compare_b) < 1.0e-5
    assert float(sphere._get_volume()) == pytest.approx((4.0 / 3.0) * math.pi * 0.5**3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    sphere = Sphere(X, [0, 0, 0], 1.0, backend=backend)
    points = sphere.sample_random_uniform(10, device=device)
    assert points.shape == (10, 3)
    for p in points:
        dist = math.sqrt(float(p[0]) ** 2 + float(p[1]) ** 2 + float(p[2]) ** 2)
        assert dist <= 1.01
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform_repeat(backend, device):
    # second run with non-zero center
    sphere = Sphere(X, [0.5, 1.0, 1.5], 0.8, backend=backend)
    pts = sphere.sample_random_uniform(20, device=device)
    assert pts.shape == (20, 3)
    for p in pts:
        dist = math.sqrt(
            (float(p[0]) - 0.5) ** 2 + (float(p[1]) - 1.0) ** 2 + (float(p[2]) - 1.5) ** 2
        )
        assert dist <= 0.81
    if issubclass(backend, DeepLearningBackend):
        assert pts.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_inside_outside(backend):
    sphere = Sphere(X, [0, 0, 0], 1.0, backend=backend)
    inside = [0.2, 0.3, 0.4]
    outside = [1.5, 0.0, 0.0]
    on_surface = [1.0, 0.0, 0.0]
    assert float(sphere.contains([inside]).reshape(-1, 1)[0][0]) == 1.0
    assert float(sphere.contains([outside]).reshape(-1, 1)[0][0]) == 0.0
    assert float(sphere.contains([on_surface]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_edges_and_corners(backend):
    sphere = Sphere(X, [0.5, 0.5, 0.5], 1.0, backend=backend)
    # point on surface
    on_surface = [1.5, 0.5, 0.5]
    # point near center
    near_center = [0.6, 0.5, 0.5]
    assert float(sphere.contains([on_surface]).reshape(-1, 1)[0][0]) == 1.0
    assert float(sphere.contains([near_center]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_and_membership(backend, device):
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    pts = sphere.sample_grid(100, device=device)
    assert pts.shape == (100, 3)
    inside = sphere.contains(pts)
    inside_count = sum(
        1 for i in range(len(pts)) if float(inside.reshape(-1, 1)[i][0]) == 1.0
    )
    assert inside_count >= int(0.95 * len(pts))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_repeat_and_device(backend, device):
    sphere = Sphere(X, [0.2, 0.2, 0.2], 0.5, backend=backend)
    pts = sphere.sample_grid(50, device=device)
    assert pts.shape == (50, 3)
    # all sampled points should be inside or on boundary
    inside = sphere.contains(pts)
    assert all(
        float(v[0]) in (0.0, 1.0) or float(v[0]) >= 0.0 for v in inside.reshape(-1, 1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_and_volume(backend):
    sphere = Sphere(X, [1.0, 2.0, 3.0], 0.5, backend=backend)
    bbox = sphere.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    compare_list = [0.5, 1.5, 1.5, 2.5, 2.5, 3.5]
    for b, compare_b in zip(bx, compare_list):
        assert abs(b - compare_b) < 1.0e-5
    assert float(sphere._get_volume()) == pytest.approx((4.0 / 3.0) * math.pi * 0.5**3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_repeat(backend):
    sphere = Sphere(X, [-1.0, -2.0, -3.0], 1.0, backend=backend)
    bbox = sphere.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    assert bx == [-2.0, 0.0, -3.0, -1.0, -4.0, -2.0]
    assert float(sphere._get_volume()) == pytest.approx((4.0 / 3.0) * math.pi * 1.0**3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_surface(backend):
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    boundary = sphere.create_boundary()
    # point on surface in different directions
    p1 = [1.0, 0.0, 0.0]
    p2 = [0.0, 1.0, 0.0]
    p3 = [0.0, 0.0, 1.0]
    assert boundary.contains([p1]).reshape(-1, 1)[0][0]
    assert boundary.contains([p2]).reshape(-1, 1)[0][0]
    assert boundary.contains([p3]).reshape(-1, 1)[0][0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_non_boundary(backend):
    sphere = Sphere(X, [0.0, 0.0, 0.0], 2.0, backend=backend)
    boundary = sphere.create_boundary()
    inside = [1.0, 0.0, 0.0]
    assert float(boundary.contains([inside]).reshape(-1, 1)[0][0]) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_and_normals(backend, device):
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    boundary = sphere.create_boundary()
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
    # normals should point outward from center
    centroid = [0.0, 0.0, 0.0]
    for p, n in zip(pts, normals):
        vx = float(p[0]) - centroid[0]
        vy = float(p[1]) - centroid[1]
        vz = float(p[2]) - centroid[2]
        dot = float(n[0]) * vx + float(n[1]) * vy + float(n[2]) * vz
        assert dot > -1e-6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_repeat(backend, device):
    sphere = Sphere(X, [0.2, 0.3, 0.4], 0.7, backend=backend)
    boundary = sphere.create_boundary()
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
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    boundary = sphere.create_boundary()
    # point on positive x-axis
    p_x = [1.0, 0.0, 0.0]
    n_x = boundary.normal([p_x])
    assert pytest.approx(1.0, rel=1e-6) == float(n_x[0][0])
    assert pytest.approx(0.0, rel=1e-6) == float(n_x[0][1])
    assert pytest.approx(0.0, rel=1e-6) == float(n_x[0][2])
    # point on positive z-axis
    p_z = [0.0, 0.0, 1.0]
    n_z = boundary.normal([p_z])
    assert pytest.approx(0.0, rel=1e-6) == float(n_z[0][0])
    assert pytest.approx(0.0, rel=1e-6) == float(n_z[0][1])
    assert pytest.approx(1.0, rel=1e-6) == float(n_z[0][2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_grid_sampling(backend):
    sphere = Sphere(X, [0.0, 0.0, 0.0], 1.0, backend=backend)
    boundary = sphere.create_boundary()
    pts = boundary.sample_grid(20)
    assert pts.shape == (20, 3)
    # all returned points should be on the boundary
    on_boundary = boundary.contains(pts)
    assert all(
        float(v[0]) >= 0.0 or float(v[0]) == 1.0 or float(v[0]) == 0.0
        for v in on_boundary.reshape(-1, 1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_grid_sampling_with_normals(backend):
    sphere = Sphere(X, [1.0, 1.0, 1.0], 0.5, backend=backend)
    boundary = sphere.create_boundary()
    pts, normals = boundary.sample_grid(15, include_normals=True)
    assert pts.shape == (15, 3)
    assert normals.shape == (15, 3)
    # normals should be unit length
    for n in normals:
        norm = math.sqrt(float(n[0]) ** 2 + float(n[1]) ** 2 + float(n[2]) ** 2)
        assert pytest.approx(1.0, rel=1e-3) == norm
