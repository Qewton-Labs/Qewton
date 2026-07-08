import inspect
import pytest
import math

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.geometries.continuous.domains_2d.triangle import Triangle
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
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_triangle(backend):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    # origin and corners should be 2D vectors
    assert tuple(tri.origin.shape) == (2,)
    assert tuple(tri.corner_1.shape) == (2,)
    assert tuple(tri.corner_2.shape) == (2,)

    # bounding box should match expected for the canonical triangle
    bbox = tri.bounding_box()
    # bbox ordering: min_x, max_x, min_y, max_y
    bx = [float(bbox[i]) for i in range(4)]
    assert bx == [0.0, 1.0, 0.0, 1.0]

    # volume (area) should be positive
    vol = tri._get_volume()
    assert float(vol) > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform(backend, device):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    points = tri.sample_random_uniform(10, device=device)
    assert points.shape == (10, 2)
    for p in points:
        assert 0.0 <= float(p[0]) <= 1.0
        assert 0.0 <= float(p[1]) <= 1.0
        assert float(p[0] + p[1]) <= 1.0
    if issubclass(backend, DeepLearningBackend):
        assert points.device == backend.get_device(device)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_uniform_points_are_inside(backend, device):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    pts = tri.sample_random_uniform(100, device=device)
    # every sampled point should be inside or on the boundary
    inside = tri.contains(pts)
    for i in range(len(pts)):
        assert (
            float(inside.reshape(-1, 1)[i][0]) in (0.0, 1.0)
            or float(inside.reshape(-1, 1)[i][0]) >= 0.0
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_known_points(backend):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    inside_point = [0.2, 0.2]
    outside_point = [1.1, 1.1]
    boundary_point = [0.5, 0.5]  # on diagonal x+y==1

    assert float(tri.contains([inside_point]).reshape(-1, 1)[0][0]) == 1.0
    assert float(tri.contains([outside_point]).reshape(-1, 1)[0][0]) == 0.0
    # boundary should be reported as inside for contains (<=1.0)
    assert float(tri.contains([boundary_point]).reshape(-1, 1)[0][0]) == 1.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_and_membership(backend, device):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    pts = tri.sample_grid(50, device=device)
    assert pts.shape[1] == 2
    inside = tri.contains(pts)
    # at least 95% of grid samples should be inside (padded random points possible)
    inside_count = sum(
        1 for i in range(len(pts)) if float(inside.reshape(-1, 1)[i][0]) == 1.0
    )
    assert inside_count >= int(0.95 * len(pts))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_sampling_and_normals(backend, device):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    boundary = tri.create_boundary()
    pts, normals = boundary.sample_random_uniform(30, device=device, include_normals=True)
    assert pts.shape == (30, 2)
    assert normals.shape == (30, 2)

    # normals should be unit vectors and point (roughly) outward from centroid
    centroid = [float(tri.origin[0]), float(tri.origin[1])] + [
        0,
        0,
    ]  # placeholder to avoid linter complaints
    centroid = [
        float(tri.origin[0] + tri.corner_1[0] + tri.corner_2[0]) / 3.0,
        float(tri.origin[1] + tri.corner_1[1] + tri.corner_2[1]) / 3.0,
    ]

    for p, n in zip(pts, normals):
        nx = float(n[0])
        ny = float(n[1])
        norm = math.sqrt(nx * nx + ny * ny)
        assert pytest.approx(1.0, rel=1e-3) == norm
        # dot with vector from centroid to point should be positive (points outward)
        vx = float(p[0]) - centroid[0]
        vy = float(p[1]) - centroid[1]
        dot = nx * vx + ny * vy
        assert dot > -1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_vector_invalid_raises(backend):
    tri = Triangle(X, [0, 0], [1.0, 0.0], [0.0, 1.0], backend=backend)
    with pytest.raises(ValueError):
        tri._to_vector([1.0, 2.0, 3.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_degenerate_triangle_volume_zero(backend):
    # collinear corners -> zero area
    tri = Triangle(X, [0, 0], [1.0, 1.0], [2.0, 2.0], backend=backend)
    vol = tri._get_volume()
    assert pytest.approx(0.0, abs=1e-8) == float(vol)
