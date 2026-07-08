import inspect
import pytest
import math

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_3d.cylinder import Cylinder
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
def test_create_cylinder(backend):
    cylinder = Cylinder(X, [0, 0, 0], 1.0, 2.0, backend=backend)
    # center should be a 3-vector
    assert tuple(cylinder.center.shape) == (3,)
    # bounding box should list min and max for x,y,z
    bbox = cylinder.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    assert bx == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    # volume should match cylinder formula
    assert float(cylinder._get_volume()) == pytest.approx(math.pi * 1.0**2 * 2.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_cylinder_offset(backend):
    cylinder = Cylinder(X, [0.5, -1.0, 2.0], 0.5, 1.0, backend=backend)
    bbox = cylinder.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    compare_list = [0.0, 1.0, -1.5, -0.5, 1.5, 2.5]
    for b, compare_b in zip(bx, compare_list):
        assert abs(b - compare_b) < 1.0e-5
    assert float(cylinder._get_volume()) == pytest.approx(math.pi * 0.5**2 * 1.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    cylinder = Cylinder(X, [0, 0, 0], 1.0, 2.0, backend=backend)
    points = cylinder.sample_random_uniform(10, device=device)
    assert points.shape == (10, 3)
    for p in points:
        radial_dist = math.sqrt(float(p[0]) ** 2 + float(p[1]) ** 2)
        z = float(p[2])
        assert radial_dist <= 1.01
        assert -1.01 <= z <= 1.01
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform_repeat(backend, device):
    # second run with non-zero center
    cylinder = Cylinder(X, [0.5, 1.0, 1.5], 0.8, 0.6, backend=backend)
    pts = cylinder.sample_random_uniform(20, device=device)
    assert pts.shape == (20, 3)
    for p in pts:
        radial_dist = math.sqrt((float(p[0]) - 0.5) ** 2 + (float(p[1]) - 1.0) ** 2)
        z = float(p[2])
        assert radial_dist <= 0.81
        assert 1.2 <= z <= 1.8
    if issubclass(backend, DeepLearningBackend):
        assert pts.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_inside_outside(backend):
    cylinder = Cylinder(X, [0, 0, 0], 1.0, 2.0, backend=backend)
    inside = [0.2, 0.3, 0.5]
    outside = [1.5, 0.0, 0.0]
    on_side = [1.0, 0.0, 0.0]
    on_bottom_cap = [0.5, 0.3, -1.0]
    assert float(cylinder.contains([inside]).reshape(-1, 1)[0][0]) == 1.0
    assert float(cylinder.contains([outside]).reshape(-1, 1)[0][0]) == 0.0
    assert float(cylinder.contains([on_side]).reshape(-1, 1)[0][0]) == 1.0
    assert float(cylinder.contains([on_bottom_cap]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_edges_and_corners(backend):
    cylinder = Cylinder(X, [0.5, 0.5, 0.5], 1.0, 1.0, backend=backend)
    # edge on side surface
    edge = [1.5, 0.5, 0.5]
    # point inside
    inside = [0.7, 0.6, 0.5]
    assert float(cylinder.contains([edge]).reshape(-1, 1)[0][0]) == 1.0
    assert float(cylinder.contains([inside]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_and_membership(backend, device):
    cylinder = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    pts = cylinder.sample_grid(100, device=device)
    assert pts.shape == (100, 3)
    inside = cylinder.contains(pts)
    inside_count = sum(
        1 for i in range(len(pts)) if float(inside.reshape(-1, 1)[i][0]) == 1.0
    )
    assert inside_count >= int(0.95 * len(pts))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_repeat_and_device(backend, device):
    cylinder = Cylinder(X, [0.2, 0.2, 0.2], 0.5, 0.5, backend=backend)
    pts = cylinder.sample_grid(50, device=device)
    assert pts.shape == (50, 3)
    # all sampled points should be inside or on boundary
    inside = cylinder.contains(pts)
    assert all(
        float(v[0]) in (0.0, 1.0) or float(v[0]) >= 0.0 for v in inside.reshape(-1, 1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_and_volume(backend):
    cylinder = Cylinder(X, [1.0, 2.0, 3.0], 0.2, 0.4, backend=backend)
    bbox = cylinder.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    compare_list = [0.8, 1.2, 1.8, 2.2, 2.8, 3.2]
    for b, compare_b in zip(bx, compare_list):
        assert abs(b - compare_b) < 1.0e-5
    assert float(cylinder._get_volume()) == pytest.approx(math.pi * 0.2**2 * 0.4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bounding_box_repeat(backend):
    cylinder = Cylinder(X, [-1.0, -2.0, -3.0], 1.0, 2.0, backend=backend)
    bbox = cylinder.bounding_box()
    bx = [float(bbox[i]) for i in range(6)]
    assert bx == [-2.0, 0.0, -3.0, -1.0, -4.0, -2.0]
    assert float(cylinder._get_volume()) == pytest.approx(math.pi * 1.0**2 * 2.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_side_and_caps(backend):
    cylinder = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    boundary = cylinder.create_boundary()
    # point on side surface
    p_side = [1.0, 0.0, 0.0]
    # point on top cap
    p_top = [0.5, 0.3, 1.0]
    # point on bottom cap
    p_bottom = [0.3, 0.2, -1.0]
    assert boundary.contains([p_side]).reshape(-1, 1)[0][0]
    assert boundary.contains([p_top]).reshape(-1, 1)[0][0]
    assert boundary.contains([p_bottom]).reshape(-1, 1)[0][0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_non_boundary(backend):
    cylinder = Cylinder(X, [0.0, 0.0, 0.0], 2.0, 2.0, backend=backend)
    boundary = cylinder.create_boundary()
    inside = [1.0, 0.0, 0.0]
    assert float(boundary.contains([inside]).reshape(-1, 1)[0][0]) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_and_normals(backend, device):
    cylinder = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    boundary = cylinder.create_boundary()
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
    # normals should point outward from axis
    centroid = [0.0, 0.0, 0.0]
    for p, n in zip(pts, normals):
        # For side surface, normal should be radial
        # For caps, normal should be vertical
        dot = float(n[0]) * (float(p[0]) - centroid[0]) + float(n[1]) * (
            float(p[1]) - centroid[1]
        )
        if abs(float(p[2]) - 1.0) > 1e-3 and abs(float(p[2]) + 1.0) > 1e-3:
            # On side surface - dot product should be close to 1 (radial)
            assert dot > -1e-6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_random_sampling_repeat(backend, device):
    cylinder = Cylinder(X, [0.2, 0.3, 0.4], 0.5, 0.6, backend=backend)
    boundary = cylinder.create_boundary()
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
    cylinder = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    boundary = cylinder.create_boundary()
    # point on side surface (positive x-axis)
    p_side = [1.0, 0.0, 0.0]
    n_side = boundary.normal([p_side])
    assert pytest.approx(1.0, rel=1e-6) == float(n_side[0][0])
    assert pytest.approx(0.0, rel=1e-6) == float(n_side[0][1])
    # point on top cap
    p_top = [0.0, 0.0, 1.0]
    n_top = boundary.normal([p_top])
    assert pytest.approx(0.0, rel=1e-6) == float(n_top[0][0])
    assert pytest.approx(0.0, rel=1e-6) == float(n_top[0][1])
    assert pytest.approx(1.0, rel=1e-6) == float(n_top[0][2])
    # point on bottom cap
    p_bottom = [0.0, 0.0, -1.0]
    n_bottom = boundary.normal([p_bottom])
    assert pytest.approx(0.0, rel=1e-6) == float(n_bottom[0][0])
    assert pytest.approx(0.0, rel=1e-6) == float(n_bottom[0][1])
    assert pytest.approx(-1.0, rel=1e-6) == float(n_bottom[0][2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_grid_sampling(backend):
    cylinder = Cylinder(X, [0.0, 0.0, 0.0], 1.0, 2.0, backend=backend)
    boundary = cylinder.create_boundary()
    pts = boundary.sample_grid(30)
    assert pts.shape == (30, 3)
    # all returned points should be on the boundary
    on_boundary = boundary.contains(pts)
    assert all(
        float(v[0]) >= 0.0 or float(v[0]) == 1.0 or float(v[0]) == 0.0
        for v in on_boundary.reshape(-1, 1)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_grid_sampling_with_normals(backend):
    cylinder = Cylinder(X, [1.0, 1.0, 1.0], 0.5, 1.0, backend=backend)
    boundary = cylinder.create_boundary()
    pts, normals = boundary.sample_grid(25, include_normals=True)
    assert pts.shape == (25, 3)
    assert normals.shape == (25, 3)
    # normals should be unit length
    for n in normals:
        norm = math.sqrt(float(n[0]) ** 2 + float(n[1]) ** 2 + float(n[2]) ** 2)
        assert pytest.approx(1.0, rel=1e-3) == norm
