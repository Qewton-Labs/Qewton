"""Volumes, uniform sampling and operators of union, cut and intersection."""

import math

import numpy as np
import pytest

from qewton.backends import NumPyBackend, TorchBackend
from qewton.config.variables import Variable
from qewton.geometries.continuous.domain_operations.union import UnionGeometry
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle

X = Variable("x", 2)
BACKENDS = [TorchBackend, NumPyBackend]


def _square(backend):
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0, backend=backend)


def _circle(center, radius, backend):
    return Circle(X, center, radius, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "build, exact",
    [
        # a half circle sticks out of the square
        (lambda b: _square(b) + _circle([1.0, 0.5], 0.3, b), 1 + math.pi * 0.09 / 2),
        (lambda b: _square(b) - _circle([0.5, 0.5], 0.3, b), 1 - math.pi * 0.09),
        (lambda b: _square(b) & _circle([1.0, 0.5], 0.5, b), math.pi * 0.25 / 2),
    ],
    ids=["union", "cut", "intersection"],
)
def test_volume_is_estimated_accurately(backend, build, exact):
    assert build(backend).volume() == pytest.approx(exact, rel=0.01)


def test_volume_estimate_is_reproducible():
    first = (_square(TorchBackend) + _circle([1.0, 0.5], 0.3, TorchBackend)).volume()
    second = (_square(TorchBackend) + _circle([1.0, 0.5], 0.3, TorchBackend)).volume()
    assert first == second


def test_set_volume_overrides_the_estimate():
    union = _square(TorchBackend) + _circle([1.0, 0.5], 0.3, TorchBackend)
    union.set_volume(2.0)
    assert union.volume() == 2.0


def test_contained_cut_volume_is_exact():
    cut = _square(TorchBackend) - _circle([0.5, 0.5], 0.3, TorchBackend)
    cut.contained = True
    assert cut.volume() == pytest.approx(1 - math.pi * 0.09, rel=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_union_samples_uniformly(backend):
    square = _square(backend)
    circle = _circle([1.0, 0.5], 0.3, backend)
    points = (square + circle).sample_random_uniform(200_000)
    assert backend.math.shape(points) == (200_000, 2)

    in_square = np.asarray(backend.to_numpy(square.contains(points))).reshape(-1)
    density_square = in_square.sum() / 1.0
    density_circle_part = (~in_square).sum() / (math.pi * 0.09 / 2)
    assert density_circle_part == pytest.approx(density_square, rel=0.03)


def test_or_is_the_union():
    union = _square(TorchBackend) | _circle([1.0, 0.5], 0.3, TorchBackend)
    assert isinstance(union, UnionGeometry)


def test_union_of_unsupported_geometries_raises():
    mesh = _square(TorchBackend).create_mesh(0.5)
    with pytest.raises(NotImplementedError):
        _ = mesh + mesh
    with pytest.raises(NotImplementedError):
        _ = mesh | mesh


def test_intersection_boundary_uses_the_cached_boundaries():
    square, circle = _square(TorchBackend), _circle([1.0, 0.5], 0.5, TorchBackend)
    boundary = (square & circle).boundary
    points = boundary.sample_random_uniform(50)

    def fail():
        raise AssertionError("create_boundary called again")

    square.create_boundary = fail
    circle.create_boundary = fail
    boundary.contains(points)
    boundary.normal(points)
