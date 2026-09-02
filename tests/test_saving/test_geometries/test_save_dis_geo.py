import inspect
import pytest

from qewton.config.saving.callables import save, load
from qewton.backends.base import ComputingBackend
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.config.variables import Variable


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_point_cloud(tmp_path, backend):
    T = Variable("x", 2)
    points = backend.build_tensor([[0.0, 1.0], [2.0, 3.0]])
    point_cloud = PointCloud(T, points, backend=backend)

    save(point_cloud, tmp_path / "point_cloud_save", replace=True)
    loaded_point_cloud = load(tmp_path / "point_cloud_save")

    assert isinstance(loaded_point_cloud, PointCloud)
    assert loaded_point_cloud.variable == point_cloud.variable
    assert backend.math.allclose(
        loaded_point_cloud.discretization_points, point_cloud.discretization_points
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_point_cloud_with_discretization_of(tmp_path, backend):
    T = Variable("x", 2)
    circle = Circle(T, [0.0, 0.0], 1.0, backend=backend)
    points = backend.build_tensor([[0.0, 1.0], [2.0, 3.0]])
    point_cloud = PointCloud(T, points, discretization_of=circle, backend=backend)

    save(point_cloud, tmp_path / "point_cloud_save", replace=True)
    loaded_point_cloud = load(tmp_path / "point_cloud_save")

    assert isinstance(loaded_point_cloud, PointCloud)
    assert isinstance(loaded_point_cloud.discretization_of, Circle)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_grid_geometry(tmp_path, backend):
    T = Variable("x", 2)
    point_grid = backend.build_tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )
    point_filter = backend.build_tensor([[[True], [False]], [[True], [True]]])
    grid = GridGeometry(T, point_grid, point_filter=point_filter, backend=backend)

    save(grid, tmp_path / "grid_geometry_save", replace=True)
    loaded_grid = load(tmp_path / "grid_geometry_save")

    assert isinstance(loaded_grid, GridGeometry)
    assert loaded_grid.variable == grid.variable
    assert backend.math.allclose(
        loaded_grid.discretization_points, grid.discretization_points
    )
    assert backend.math.all(loaded_grid.point_filter == grid.point_filter)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_grid_geometry_with_discretization_of(tmp_path, backend):
    T = Variable("x", 2)
    circle = Circle(T, [0.0, 0.0], 1.0, backend=backend)
    point_grid = backend.build_tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )
    grid = GridGeometry(T, point_grid, discretization_of=circle, backend=backend)

    save(grid, tmp_path / "grid_geometry_save", replace=True)
    loaded_grid = load(tmp_path / "grid_geometry_save")

    assert isinstance(loaded_grid, GridGeometry)
    assert isinstance(loaded_grid.discretization_of, Circle)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_mesh_geometry(tmp_path, backend):
    T = Variable("x", 2)
    mesh = Mesh(
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        cells=[[0, 1, 2], [0, 2, 3]],
        backend=backend,
    )
    mesh_geometry = MeshGeometry(T, mesh, backend=backend)

    save(mesh_geometry, tmp_path / "mesh_geometry_save", replace=True)
    loaded_mesh_geometry = load(tmp_path / "mesh_geometry_save")

    assert isinstance(loaded_mesh_geometry, MeshGeometry)
    assert loaded_mesh_geometry.variable == mesh_geometry.variable
    assert backend.math.allclose(
        loaded_mesh_geometry.mesh.vertices, mesh_geometry.mesh.vertices
    )
    assert backend.math.all(loaded_mesh_geometry.mesh.cells == mesh_geometry.mesh.cells)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_mesh_geometry_with_discretization_of(tmp_path, backend):
    T = Variable("x", 2)
    circle = Circle(T, [0.0, 0.0], 1.0, backend=backend)
    mesh = Mesh(
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        cells=[[0, 1, 2], [0, 2, 3]],
        backend=backend,
    )
    mesh_geometry = MeshGeometry(T, mesh, discretization_of=circle, backend=backend)

    save(mesh_geometry, tmp_path / "mesh_geometry_save", replace=True)
    loaded_mesh_geometry = load(tmp_path / "mesh_geometry_save")

    assert isinstance(loaded_mesh_geometry, MeshGeometry)
    assert isinstance(loaded_mesh_geometry.discretization_of, Circle)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_mesh_internal(tmp_path, backend):
    mesh = Mesh(
        vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        cells=[[0, 1, 2]],
        cell_markers=[3],
        faces=[[0, 1], [1, 2], [0, 2]],
        face_markers=[1, 2, 3],
        marker_labels={"boundary": (1, 1)},
        backend=backend,
    )

    save(mesh, tmp_path / "mesh_save", replace=True)
    loaded_mesh = load(tmp_path / "mesh_save")

    assert backend.math.allclose(loaded_mesh.vertices, mesh.vertices)
    assert backend.math.all(loaded_mesh.cells == mesh.cells)
    assert backend.math.all(loaded_mesh.cell_markers == mesh.cell_markers)
    assert backend.math.all(loaded_mesh.faces == mesh.faces)
    assert backend.math.all(loaded_mesh.face_markers == mesh.face_markers)
    assert loaded_mesh.marker_labels == mesh.marker_labels
