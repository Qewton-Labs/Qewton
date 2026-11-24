"""Structure, topological dimension, coordinates and cells of discrete
geometries, as used to draw them."""

import numpy as np
import pytest

from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.point_cloud import PointCloud

T, X, Y = Variable("t", 1), Variable("x", 2), Variable("y", 3)


def _square_mesh():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)


# name: (factory, structure, topological_dim, dim)
CASES = {
    "point_cloud": (lambda: PointCloud(X, np.random.rand(7, 2)), Structure.POINTS, 0, 2),
    "mesh_1d": (lambda: Interval(T, 0.0, 1.0).create_mesh(0.25), Structure.MESH, 1, 1),
    "boundary_2d": (
        lambda: MeshGeometry(X, _square_mesh().mesh.get_boundary_mesh()),
        Structure.MESH,
        1,
        2,
    ),
    "mesh_2d": (_square_mesh, Structure.MESH, 2, 2),
    "mesh_3d": (
        lambda: Box(Y, [0.0] * 3, 1.0, 1.0, 1.0).create_mesh(0.6),
        Structure.MESH,
        3,
        3,
    ),
    "grid_1d": (lambda: GridGeometry.build_unit_grid(T, 5), Structure.GRID, 1, 1),
    "grid_2d": (lambda: GridGeometry.build_unit_grid(X, 4), Structure.GRID, 2, 2),
    "grid_3d": (lambda: GridGeometry.build_unit_grid(Y, 3), Structure.GRID, 3, 3),
    "index_grid": (
        lambda: IndexGridGeometry(Variable("i", 1), (6,)),
        Structure.GRID,
        1,
        1,
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_structure_and_dimensions(name):
    factory, structure, topological_dim, dim = CASES[name]
    geometry = factory()
    assert geometry.structure is structure
    assert geometry.topological_dim == topological_dim
    assert geometry.dim == dim


@pytest.mark.parametrize("name", CASES)
def test_coordinates_have_one_row_per_point(name):
    geometry = CASES[name][0]()
    coordinates = geometry.coordinates()
    assert isinstance(coordinates, np.ndarray)
    assert coordinates.shape == (int(np.prod(geometry.shape)), geometry.dim)


@pytest.mark.parametrize("name", CASES)
def test_cells_are_simplices_indexing_the_points(name):
    geometry = CASES[name][0]()
    cells = geometry.cells
    if geometry.structure is Structure.POINTS:
        assert cells is None
        return
    assert cells.shape[1] == geometry.topological_dim + 1
    assert cells.min() >= 0 and cells.max() < len(geometry.coordinates())


def test_grid_cells_fill_every_hexahedron_with_six_tetrahedra():
    geometry = GridGeometry.build_unit_grid(Y, 3)
    points = geometry.coordinates()
    tetrahedra = points[geometry.cells]
    edges = tetrahedra[:, 1:] - tetrahedra[:, :1]
    volumes = np.abs(np.linalg.det(edges)) / 6
    assert len(volumes) == 8 * 6
    assert volumes.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["grid_3d", "mesh_3d"])
def test_the_boundary_of_a_volume_is_closed_triangles(name):
    geometry = CASES[name][0]()
    boundary = geometry.boundary_cells
    assert boundary.shape[1] == 3
    edges = np.sort(
        np.concatenate([boundary[:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 0))]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)  # every edge of a closed surface joins two triangles


def test_the_boundary_of_a_unit_grid_lies_on_the_faces_of_the_cube():
    geometry = GridGeometry.build_unit_grid(Y, 3)
    corners = geometry.coordinates()[geometry.boundary_cells]
    on_face = [
        any(np.allclose(triangle[:, k], v) for k in range(3) for v in (0.0, 1.0))
        for triangle in corners
    ]
    assert all(on_face)
    assert len(corners) == 6 * 4 * 2


def test_a_point_cloud_has_no_boundary():
    with pytest.raises(ValueError):
        CASES["point_cloud"][0]().boundary_cells


def test_grid_cells_cover_every_quad_with_two_triangles():
    grid = GridGeometry.build_unit_grid(X, 4)
    assert grid.cells.shape == (2 * 3 * 3, 3)


def test_axis_coordinates_follow_the_varying_component():
    # meshgrid(x, y) varies x along the second grid axis.
    x, y = np.linspace(0, 1, 3), np.linspace(10, 20, 4)
    points = np.stack(np.meshgrid(x, y), axis=-1)  # shape (4, 3, 2)
    grid = GridGeometry(X, points)
    assert grid.axis_components == (1, 0)
    np.testing.assert_allclose(grid.axis_coordinates()[0], y)
    np.testing.assert_allclose(grid.axis_coordinates()[1], x)


def test_rotated_grid_is_drawn_as_a_mesh():
    u, v = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij")
    angle = 0.3
    points = np.stack(
        [np.cos(angle) * u - np.sin(angle) * v, np.sin(angle) * u + np.cos(angle) * v],
        axis=-1,
    )
    grid = GridGeometry(X, points)
    assert grid.structure is Structure.MESH
    with pytest.raises(ValueError, match="not aligned"):
        grid.axis_coordinates()


@pytest.mark.parametrize("name", ["point_cloud", "mesh_2d"])
def test_axis_coordinates_raise_without_a_grid(name):
    with pytest.raises(ValueError, match="not a rectilinear grid"):
        CASES[name][0]().axis_coordinates()
