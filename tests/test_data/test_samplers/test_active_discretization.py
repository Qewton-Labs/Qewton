import numpy as np
import pytest

from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.point_cloud import PointCloud


def _sampler():
    X = Variable("x", 2)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    return GridSampler(square, 10)


def _point_cloud(variable, points):
    return PointCloud(variable, np.asarray(points, dtype=np.float32))


class TestActiveDiscretizationValidation:
    def test_rejects_a_mismatched_variable_dimension(self):
        sampler = _sampler()
        geometry = _point_cloud(Variable("y", 3), [[0.1, 0.1, 0.1]])
        with pytest.raises(ValueError, match="dim"):
            sampler.active_discretization = geometry

    def test_rejects_a_bounding_box_that_does_not_overlap_the_domain(self):
        """A reference far outside the domain's bounding box usually means a
        units/normalization mismatch, not a genuinely out-of-domain
        reference."""
        sampler = _sampler()
        geometry = _point_cloud(Variable("y", 2), [[100.0, 100.0], [101.0, 101.0]])
        with pytest.raises(ValueError, match="overlap"):
            sampler.active_discretization = geometry

    def test_rejects_points_outside_the_domain_despite_an_overlapping_box(self):
        """Bounding boxes can overlap while individual points still fall
        outside a non-box-shaped (or just smaller) domain - the stricter
        contains() check catches that."""
        sampler = _sampler()
        geometry = _point_cloud(Variable("y", 2), [[0.5, 0.5], [1.5, 0.5]])
        with pytest.raises(ValueError, match="outside this sampler's domain"):
            sampler.active_discretization = geometry

    def test_accepts_a_geometry_fully_inside_the_domain(self):
        sampler = _sampler()
        geometry = _point_cloud(Variable("y", 2), [[0.2, 0.2], [0.8, 0.8]])
        sampler.active_discretization = geometry
        assert sampler.active_discretization is geometry

    def test_none_clears_it_without_validation(self):
        sampler = _sampler()
        sampler.active_discretization = _point_cloud(Variable("y", 2), [[0.2, 0.2]])
        sampler.active_discretization = None
        assert sampler.active_discretization is None


class TestActiveDiscretizationForward:
    def test_forward_returns_the_injected_points(self):
        sampler = _sampler()
        points = [[0.2, 0.2], [0.8, 0.8], [0.5, 0.1]]
        geometry = _point_cloud(Variable("y", 2), points)
        sampler.active_discretization = geometry
        result = sampler.forward()
        assert np.allclose(np.asarray(result), np.asarray(points, dtype=np.float32))

    def test_forward_updates_the_sampled_geometrys_current_discretization(self):
        sampler = _sampler()
        points = [[0.2, 0.2], [0.8, 0.8]]
        geometry = _point_cloud(Variable("y", 2), points)
        sampler.active_discretization = geometry
        sampler.forward()
        current = np.asarray(sampler.sampled_geometry._current_points)
        assert np.allclose(current, np.asarray(points, dtype=np.float32))

    def test_takes_priority_over_mesh_mode(self):
        sampler = _sampler()
        sampler.set_mesh_mode(max_vertex_distance=0.5)
        points = [[0.2, 0.2], [0.8, 0.8]]
        geometry = _point_cloud(Variable("y", 2), points)
        sampler.active_discretization = geometry
        result = sampler.forward()
        assert np.allclose(np.asarray(result), np.asarray(points, dtype=np.float32))

    def test_compute_normals_is_not_supported(self):
        X = Variable("x", 2)
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
        boundary = square.boundary
        sampler = RandomUniformSampler(boundary, 5, compute_normals=True)
        geometry = _point_cloud(Variable("y", 2), [[0.2, 0.0], [0.8, 0.0]])
        sampler.active_discretization = geometry
        with pytest.raises(NotImplementedError, match="normals"):
            sampler.forward()
