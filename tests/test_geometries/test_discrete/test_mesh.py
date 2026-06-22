import inspect
import pytest

from qewton.backends.base import ComputingBackend
from qewton.geometries.discrete.mesh import Mesh
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
T = Variable("t", 1)
devices = [cpu, cuda(0)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_triangle_mesh(backend):
    # Create a simple 2D triangle mesh and check basics
    vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    cells = [[0, 1, 2]]
    m = Mesh(vertices=vertices, cells=cells, backend=backend)

    assert m.vertex_count == 3
    assert m.topological_dim == 3
    # boundary faces should be three edges
    assert m.boundary_faces.shape[0] == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_tetrahedron(backend):
    vertices3 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cells3 = [[0, 1, 2, 3]]
    m3 = Mesh(vertices=vertices3, cells=cells3, backend=backend)
    assert m3.vertex_count == 4
    assert m3.topological_dim == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_cell_volumes(backend):
    # Build a square triangulated into two triangles
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)

    vols = mesh.compute_cell_volumes()
    s = backend.math.sum(vols)
    try:
        total_area = float(s)
    except Exception:
        total_area = s.item() if hasattr(s, "item") else s
    assert pytest.approx(1.0, rel=1e-6) == total_area


@pytest.mark.parametrize("backend", BACKENDS)
def test_probability_weights_sum_to_one(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)
    p = mesh.compute_cell_probability_weights()
    psum = backend.math.sum(p)
    try:
        psum_f = float(psum)
    except Exception:
        psum_f = psum.item() if hasattr(psum, "item") else psum
    assert pytest.approx(1.0, rel=1e-6) == psum_f


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_submesh_and_boundary(backend):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    cell_markers = [0, 1]
    mesh = Mesh(
        vertices=vertices, cells=cells, cell_markers=cell_markers, backend=backend
    )

    sub = mesh.get_submesh(1)
    assert sub.cells.shape[0] == 1

    b = mesh.get_boundary_mesh()
    assert b.vertex_count == 4
    assert b.cells.shape[0] >= 4


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_from_vertices(backend, device):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)

    pts, idx = mesh.sample_random_from_vertices(5, device=device)
    assert pts.shape[0] == 5


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_from_vertices_small(backend, device):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)

    pts2, idx2 = mesh.sample_grid_from_vertices(2, device=device)
    assert pts2.shape[0] == 2


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_grid_from_vertices_large(backend, device):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)

    pts3, idx3 = mesh.sample_grid_from_vertices(10, device=device)
    assert pts3.shape[0] == 10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_sample_random_inside(backend, device):
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)

    inside_pts, chosen_cells = mesh.sample_random_inside(10, device=device)
    assert inside_pts.shape[0] == 10
    assert chosen_cells.shape[0] == 10
