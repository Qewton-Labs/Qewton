import inspect

import pytest

from qewton.backends.base import ComputingBackend, DeepLearningBackend
from qewton.config.devices import cpu, cuda
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.point_sampler import PointSampler, ProductSampler
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_1d.interval import Interval


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)
T = Variable("t", 1)
devices = [cpu, cuda(0)]


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
