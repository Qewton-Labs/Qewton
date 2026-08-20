import inspect
import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.point_sampler import PointSampler
from qewton.data.dataloaders.sampler.product_sampler import ProductSampler
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle


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


class DummyPointSampler(PointSampler):
    def __init__(self, geometry, n_points, backend, points=None):
        super().__init__(geometry, n_points, backend=backend)
        self._points = points

    def sample_points(self):
        if self._points is None:
            points = self.backend.build_tensor([[0.25], [0.75]])
        else:
            points = self.backend.build_tensor(self._points)
        points = self.backend.to(points, device=self._device)
        return points, None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_random_uniform_sampler_returns_points_in_interval(backend, device):
    interval = Interval(T, 0, 1, backend=backend)
    sampler = RandomUniformSampler(interval, 5, backend=backend)
    sampler.to(device)

    points = sampler.forward()

    assert points.shape == (5, 1)
    for point in points:
        assert -1e-6 <= float(point[0]) <= 1 + 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_grid_sampler_returns_expected_grid_points(backend, device):
    interval = Interval(T, 0, 1, backend=backend)
    sampler = GridSampler(interval, 3, backend=backend)
    sampler.to(device)

    points = sampler.forward()

    assert points.shape == (3, 1)
    expected = backend.build_tensor([[0.25], [0.5], [0.75]])
    if issubclass(backend, DeepLearningBackend):
        expected = backend.to(expected, device=device)
    assert backend.math.allclose(points, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_point_sampler_cache_and_clear_cache(backend):
    interval = Interval(T, 0, 1, backend=backend)
    sampler = DummyPointSampler(interval, 2, backend=backend)

    assert len(sampler) == 2
    assert sampler.forward().shape == (2, 1)

    sampler.cache(1)
    assert sampler.created_cache is True
    assert sampler.point_cache is not None

    cached_points = sampler.forward()
    assert cached_points.shape == (2, 1)

    sampler.clear_cache()
    assert sampler.created_cache is False
    assert sampler.point_cache is None
    assert sampler.normal_cache is None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_product_sampler_combines_two_samplers(backend, device):
    interval_t = Interval(T, 0, 1, backend=backend)
    interval_u = Interval(Variable("u", 1), 0, 1, backend=backend)

    sampler_t = DummyPointSampler(
        interval_t,
        2,
        backend=backend,
        points=[[0.25], [0.75]],
    )
    sampler_u = DummyPointSampler(
        interval_u,
        2,
        backend=backend,
        points=[[0.1], [0.9]],
    )
    sampler = sampler_t * sampler_u
    assert isinstance(sampler, ProductSampler)
    assert len(sampler) == 4
    sampler.to(device)

    points = sampler.forward()
    assert points.shape == (2, 2, 2)
    expected = backend.build_tensor(
        [[[0.25, 0.1], [0.25, 0.9]], [[0.75, 0.1], [0.75, 0.9]]]
    )
    expected = backend.to(expected, device=device)
    assert backend.math.allclose(points, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_product_sampler_rejects_same_variable(backend):
    interval = Interval(T, 0, 1, backend=backend)
    sampler_a = DummyPointSampler(interval, 2, backend=backend)
    sampler_b = DummyPointSampler(interval, 2, backend=backend)

    with pytest.raises(AssertionError):
        sampler_a * sampler_b


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_product_sampler_cache_preserves_cached_batches(backend, device):
    interval_t = Interval(T, 0, 1, backend=backend)
    interval_u = Interval(Variable("u", 1), 0, 1, backend=backend)

    sampler_t = DummyPointSampler(
        interval_t,
        2,
        backend=backend,
        points=[[0.25], [0.75]],
    )
    sampler_u = DummyPointSampler(
        interval_u,
        2,
        backend=backend,
        points=[[0.1], [0.9]],
    )
    sampler = sampler_t * sampler_u
    sampler.cache(1)
    sampler.to(device)

    first = sampler.forward()
    second = sampler.forward()
    assert backend.math.allclose(first, second)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_grid_sampler_with_filter_in_square(backend, device):
    """Test grid sampling in a square with a filter that removes some points."""
    xy = Variable("xy", 2)

    square = Rectangle(xy, origin=[0, 0], width=1, height=1, backend=backend)

    def filter_func(x_coord: xy):
        return x_coord[..., :1] > 0.5

    sampler = GridSampler(square, 1000, filter_fn=filter_func, backend=backend)
    sampler.to(device)

    points = sampler.forward()
    assert points.shape == (1000, 2)

    x_coords = points[..., 0]
    assert backend.math.all(x_coords > 0.5 - 1e-6)
    assert backend.math.all(points >= 0 - 1e-6)
    assert backend.math.all(points <= 1 + 1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_random_uniform_sampler_with_filter_in_square(backend, device):
    """Test sampling in a square (rectangle) with a filter that removes some points."""
    xy = Variable("xy", 2)

    # Create a square from (0,0) to (1,1)
    square = Rectangle(xy, origin=[0, 0], width=1, height=1, backend=backend)

    def filter_func(x_coord: xy):
        return x_coord[..., :1] > 0.5

    sampler = RandomUniformSampler(square, 10, filter_fn=filter_func, backend=backend)
    sampler.to(device)

    points = sampler.forward()

    # Check shape: should have 10 points with 2 coordinates
    assert points.shape == (10, 2)

    # Check that all points satisfy the filter condition (x > 0.5)
    x_coords = points[..., 0]
    assert backend.math.all(x_coords > 0.5 - 1e-6)

    # Check that all points are still within the square
    assert backend.math.all(points >= 0 - 1e-6)
    assert backend.math.all(points <= 1 + 1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_random_uniform_sampler_with_filter_on_boundary(backend, device):
    """Test sampling on the boundary of a square with a filter."""
    xy = Variable("xy", 2)

    # Create a square from (0,0) to (1,1)
    square = Rectangle(xy, origin=[0, 0], width=1, height=1, backend=backend)

    # Get the boundary of the square
    boundary = square.boundary

    # Define a filter: keep only points on the left or right edge
    # On the boundary: (x, y) where x=0 or x=1 or y=0 or y=1
    # Filter: keep points where x is close to 0 or x is close to 1
    def filter_func(xy_coord: xy):
        x_coord = xy_coord[..., 0]
        return (x_coord < 0.1) | (x_coord > 0.9)

    sampler = RandomUniformSampler(boundary, 15, filter_fn=filter_func, backend=backend)
    sampler.to(device)

    points = sampler.forward()

    # Check shape: should have 15 points with 2 coordinates
    assert points.shape == (15, 2)

    # Check that all points satisfy the filter condition
    x_coords = points[..., 0]
    filter_result = (x_coords < 0.1 + 1e-6) | (x_coords > 0.9 - 1e-6)
    assert backend.math.all(filter_result)

    # Check that all points are on the boundary
    # A point is on boundary if x=0 or x=1 or y=0 or y=1
    on_left = backend.math.abs(x_coords) < 1e-6
    on_right = backend.math.abs(x_coords - 1) < 1e-6
    y_coords = points[..., 1]
    on_bottom = backend.math.abs(y_coords) < 1e-6
    on_top = backend.math.abs(y_coords - 1) < 1e-6
    on_boundary = on_left | on_right | on_bottom | on_top
    assert backend.math.all(on_boundary)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_random_uniform_sampler_with_filter_on_boundary_with_normals(backend, device):
    """Test boundary sampling with normals and a filter on x > 0.5."""
    xy = Variable("xy", 2)

    square = Rectangle(xy, origin=[0, 0], width=1, height=1, backend=backend)
    boundary = square.boundary

    def filter_func(xy_coord: xy):
        x_coord = xy_coord[..., 0]
        return x_coord > 0.5

    sampler = RandomUniformSampler(
        boundary,
        12,
        filter_fn=filter_func,
        compute_normals=True,
        backend=backend,
    )
    sampler.to(device)

    points, normals = sampler.forward()

    assert points.shape == (12, 2)
    assert normals.shape == (12, 2)

    x_coords = points[..., 0]
    assert backend.math.all(x_coords > 0.5 - 1e-6)

    # Normals should be unit length for boundary sampling.
    normal_lengths = backend.math.sqrt(backend.math.sum(normals**2, axis=1))
    assert backend.math.all(backend.math.abs(normal_lengths - 1.0) < 1e-6)

    # All points should remain on the boundary.
    y_coords = points[..., 1]
    on_left = backend.math.abs(x_coords) < 1e-6
    on_right = backend.math.abs(x_coords - 1) < 1e-6
    on_bottom = backend.math.abs(y_coords) < 1e-6
    on_top = backend.math.abs(y_coords - 1) < 1e-6
    assert backend.math.all(on_left | on_right | on_bottom | on_top)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_random_uniform_sampler_with_filter_on_product_square_interval(backend, device):
    """Test sampling in a product of a square and an interval with a filter."""
    xy = Variable("xy", 2)
    t = Variable("t", 1)

    # Create a square from (0,0) to (1,1)
    square = Rectangle(xy, origin=[0, 0], width=1, height=1, backend=backend)

    # Create an interval from 0 to 1
    interval = Interval(t, 0, 1, backend=backend)

    # Define a filter on the product: keep points where y > 0.4 and t > 0.5
    def filter_func(xy_coord: xy, t_coord: t):
        y_coord = xy_coord[..., 1:]
        return (y_coord > 0.4) & (t_coord > 0.5)

    # Create a sampler with filter on the product geometry
    product_geom = square * interval
    product_sampler_filtered = RandomUniformSampler(
        product_geom, 12, filter_fn=filter_func, backend=backend
    )
    product_sampler_filtered.to(device)

    points = product_sampler_filtered.forward()

    # Check shape: should have 12 points with 3 coordinates (x, y, t)
    assert points.shape == (12, 3)

    # Check that all points satisfy the filter conditions
    y_coords = points[..., 1]
    t_coords = points[..., 2]
    assert backend.math.all(y_coords > 0.4 - 1e-6)
    assert backend.math.all(t_coords > 0.5 - 1e-6)

    # Check that all points are within the expected bounds
    assert backend.math.all(points[:, 0] >= 0 - 1e-6)  # x
    assert backend.math.all(points[:, 0] <= 1 + 1e-6)  # x
    assert backend.math.all(points[:, 1] >= 0 - 1e-6)  # y
    assert backend.math.all(points[:, 1] <= 1 + 1e-6)  # y
    assert backend.math.all(points[:, 2] >= 0 - 1e-6)  # t
    assert backend.math.all(points[:, 2] <= 1 + 1e-6)  # t


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_random_uniform_sampler_with_filter_in_circle(backend, device):
    """Test sampling in a circle with a filter that keeps only points in upper half."""
    xy = Variable("xy", 2)

    # Create a circle with center at (0.5, 0.5) and radius 0.5
    circle = Circle(xy, center=[0.5, 0.5], radius=0.5, backend=backend)

    # Define a filter: keep only points in the upper half (y > 0.5)
    def filter_func(xy_coord: xy):
        y_coord = xy_coord[..., 1:]
        return y_coord > 0.5

    sampler = RandomUniformSampler(circle, 20, filter_fn=filter_func, backend=backend)
    sampler.to(device)

    points = sampler.forward()

    # Check shape: should have 20 points with 2 coordinates
    assert points.shape == (20, 2)

    # Check that all points satisfy the filter condition (y > 0.5)
    y_coords = points[..., 1]
    assert backend.math.all(y_coords > 0.5 - 1e-6)

    # Check that all points are within the circle
    # Distance from center (0.5, 0.5) should be <= 0.5
    center = backend.build_tensor([0.5, 0.5])
    if issubclass(backend, DeepLearningBackend):
        center = backend.to(center, device=device)
    distances = backend.math.sqrt(backend.math.sum((points - center) ** 2, axis=1))
    assert backend.math.all(distances <= 0.5 + 1e-6)
