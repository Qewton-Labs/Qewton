import inspect
import pytest

from qewton.backends.base import ComputingBackend
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry, MeshBoundaryGeometry
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
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


@pytest.mark.parametrize("backend", BACKENDS)
def test_meshgeometry_bounding_box_and_volume(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)

    bb = mg.bounding_box()
    # bounding box should have 4 values: xmin,xmax,ymin,ymax
    assert len(bb) == 4

    vol = mg._get_volume()
    s = backend.math.sum(mesh.compute_cell_volumes())
    try:
        expected = float(s)
    except Exception:
        expected = s.item() if hasattr(s, "item") else s
    try:
        got = float(vol)
    except Exception:
        got = vol.item() if hasattr(vol, "item") else vol
    assert pytest.approx(expected, rel=1e-6) == got


@pytest.mark.parametrize("backend", BACKENDS)
def test_meshgeometry_boundary_and_create_boundary(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)

    b_prop = mg.create_boundary()
    assert isinstance(b_prop, MeshBoundaryGeometry)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sampling_methods_return_expected_shapes(backend, device):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)

    pts1 = mg.sample_random_uniform_from_discretization(5, device=device)
    assert pts1.shape[0] == 5

    pts2 = mg.sample_grid_from_discretization(3, device=device)
    assert pts2.shape[0] == 3

    pts3 = mg.sample_random_uniform(4, device=device)
    assert pts3.shape[0] == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_grid_uses_contains_and_returns_requested_number(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)

    pts = mg.sample_grid(5)
    assert pts.shape[0] == 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_contains_point_and_cell_based(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)

    # point-based (len(points) < len(cells))
    p_inside = mg.contains(mg.sample_random_uniform_from_discretization(1))
    assert p_inside.shape[0] == 1

    # cell-based (len(points) >= len(cells)) - use multiple points
    many_pts = mg.sample_random_uniform(10)
    inside_mask = mg.contains(many_pts)
    assert inside_mask.shape[0] == 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_submesh_returns_meshgeometry(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    cell_markers = [0, 1]
    mesh = Mesh(
        vertices=vertices, cells=cells, cell_markers=cell_markers, backend=backend
    )
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)

    sub = mg.get_submesh(1)
    assert isinstance(sub, MeshGeometry)
    assert sub.mesh.cells.shape[0] == 1


# MeshBoundaryGeometry tests
@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_bounding_box_and_volume(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)
    mb = MeshBoundaryGeometry(mg)

    bb = mb.bounding_box()
    assert len(bb) == 4

    vol = mb._get_volume()
    s = backend.math.sum(mb.mesh.compute_cell_volumes())
    try:
        expected = float(s)
    except Exception:
        expected = s.item() if hasattr(s, "item") else s
    try:
        got = float(vol)
    except Exception:
        got = vol.item() if hasattr(vol, "item") else vol
    assert pytest.approx(expected, rel=1e-6) == got


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_contains_point_and_cell_based(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)
    mb = MeshBoundaryGeometry(mg)

    # point-based
    mb._build_face_bbox()
    pts = mb.sample_random_uniform_from_discretization(1)
    inside, idx = mb._contains_point_based_search(pts)
    assert inside.shape[0] == 1

    # cell-based: generate many points on boundary via sample_grid
    pts2, _ = mb.sample_grid_from_discretization(5, include_normals=True)
    inside2, idx2 = mb._contains_cell_based_search(pts2)
    assert inside2.shape[0] == pts2.shape[0]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_sampling_and_normals(backend, device):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)
    mb = MeshBoundaryGeometry(mg)

    pts, normals = mb.sample_random_uniform_from_discretization(
        3, device=device, include_normals=True
    )
    assert pts.shape[0] == 3
    # normals may be None if implementation chooses so; when provided, should match points
    if normals is not None:
        assert normals.shape[0] == 3

    pts2, normals2 = mb.sample_random_uniform(4, device=device, include_normals=True)
    assert pts2.shape[0] == 4
    if normals2 is not None:
        assert normals2.shape[0] == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_normal_method_and_get_submesh(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    faces = [[0, 1], [1, 2]]
    face_markers = [0, 0]
    mesh = Mesh(
        vertices=vertices,
        cells=cells,
        faces=faces,
        face_markers=face_markers,
        backend=backend,
    )
    var = Variable("x", 2)
    mg = MeshGeometry(variable=var, mesh=mesh, backend=backend)
    mb = MeshBoundaryGeometry(mg)

    # normal should return array same length as input points
    pts = mb.sample_random_uniform_from_discretization(2)
    normals = mb.normal(pts)
    assert normals.shape[0] == pts.shape[0]

    sub = mb.get_submesh(0)
    assert isinstance(sub, MeshBoundaryGeometry)
