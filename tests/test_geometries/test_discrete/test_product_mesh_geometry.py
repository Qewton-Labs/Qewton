import math

import numpy as np
import pytest

from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.product_mesh_geometry import ProductMeshGeometry


T, M, K = Variable("t", 1), Variable("m", 1), Variable("k", 1)
X = Variable("x", 2)


def _time():
    return Interval(T, 0.0, 2.0)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


def _simplex_volume(mesh) -> float:
    vertices = np.asarray(mesh.vertices)
    cells = np.asarray(mesh.cells)
    edges = vertices[cells[:, 1:]] - vertices[cells[:, :1]]
    return np.abs(np.linalg.det(edges)).sum() / math.factorial(cells.shape[1] - 1)


def _facet_counts(mesh) -> np.ndarray:
    cells = np.asarray(mesh.cells)
    facets = np.concatenate(
        [np.sort(np.delete(cells, i, axis=1), axis=1) for i in range(cells.shape[1])]
    )
    _, counts = np.unique(facets, axis=0, return_counts=True)
    return counts


class TestFactors:
    def test_nested_products_are_flattened_in_order(self):
        a, b, c = _time(), Interval(M, 0.0, 1.0), _square()
        assert (a * b * c).factors == [a, b, c]
        assert (a * (b * c)).factors == [a, b, c]


class TestCreateMesh:
    def test_returns_a_product_mesh_of_the_factor_meshes(self):
        mesh = (_time() * _square()).create_mesh(0.5)
        assert isinstance(mesh, ProductMeshGeometry)
        assert mesh.factor_shape == (5, 9)
        assert mesh.discretization_points.shape == (45, 3)

    def test_vertex_distance_per_variable(self):
        mesh = (_time() * _square()).create_mesh({T: 0.25, X: 0.5})
        assert mesh.factor_shape == (9, 9)

    def test_variables_missing_from_the_dict_use_none(self):
        mesh = (_time() * _square()).create_mesh({T: 0.25})
        assert mesh.factor_shape == (9, 4)

    def test_points_are_in_product_order_first_factor_slowest(self):
        mesh = (_time() * _square()).create_mesh({T: 1.0, X: 1.0})
        points = np.asarray(mesh.discretization_points).reshape(3, 4, 3)
        assert np.allclose(points[:, 0, 0], [0.0, 1.0, 2.0])
        assert np.allclose(points[0, :, 1:], points[2, :, 1:])


class TestProductCells:
    @pytest.mark.parametrize(
        "geometry, volume",
        [
            (Interval(T, 0.0, 2.0) * Interval(M, 0.0, 3.0), 6.0),
            (Interval(T, 0.0, 2.0) * Rectangle(X, [0.0, 0.0], 1.0, 1.0), 2.0),
            (Rectangle(X, [0.0, 0.0], 1.0, 1.0) * Interval(T, 0.0, 2.0), 2.0),
            (
                Interval(T, 0.0, 2.0) * Interval(M, 0.0, 1.0) * Interval(K, 0.0, 3.0),
                6.0,
            ),
        ],
    )
    def test_cells_fill_the_product_volume(self, geometry, volume):
        mesh = geometry.create_mesh(0.5).mesh
        assert np.isclose(_simplex_volume(mesh), volume)

    def test_mesh_is_conforming(self):
        # Every facet is shared by at most two cells, interior ones by exactly two.
        mesh = (_time() * _square()).create_mesh(0.5).mesh
        counts = _facet_counts(mesh)
        assert set(np.unique(counts)) == {1, 2}
        # Boundary: 2 caps of 2x2 squares and 4 sides of 2x4 rectangles,
        # each rectangle split into 2 triangles.
        assert np.sum(counts == 1) == 2 * (2 * 2 * 2) + 4 * (2 * 4 * 2)

    def test_boundary_factor_gives_a_surface(self):
        mesh = (_time() * _square().boundary).create_mesh(0.5).mesh
        assert np.asarray(mesh.vertices).shape[1] == 3
        assert np.asarray(mesh.cells).shape[1] == 3

    def test_point_factor_gives_a_slice(self):
        mesh = (_time().boundary_left * _square()).create_mesh(0.5).mesh
        vertices = np.asarray(mesh.vertices)
        assert np.allclose(vertices[:, 0], 0.0)
        assert np.asarray(mesh.cells).shape == (8, 3)

    def test_more_than_three_dimensions_raise(self):
        product = (_time() * Interval(M, 0.0, 1.0) * _square()).create_mesh(1.0)
        with pytest.raises(ValueError, match="up to 3D"):
            product.mesh


class TestBoundingBox:
    def test_product_geometry_concatenates_factor_boxes(self):
        box = np.asarray((_time() * _square()).bounding_box())
        assert np.allclose(box, [0.0, 2.0, 0.0, 1.0, 0.0, 1.0])

    def test_product_mesh_uses_its_points(self):
        box = np.asarray((_time() * _square()).create_mesh(0.5).bounding_box())
        assert np.allclose(box, [0.0, 2.0, 0.0, 1.0, 0.0, 1.0])
