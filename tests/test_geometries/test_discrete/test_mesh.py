import inspect
import pytest

from qewton.backends.base import ComputingBackend
from qewton.geometries.discrete.meshes.mesh import Mesh
from qewton.config.variables import Variable
from qewton.config.devices import cpu, cuda, cuda_available


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_triangle_mesh(backend):
    # Create a simple 2D triangle mesh and check basics
    vertices = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    cells = [[0, 1, 2]]
    m = Mesh(vertices=vertices, cells=cells, backend=backend)

    assert m.vertex_count == 3
    # boundary faces should be three edges
    assert m.boundary_faces.shape[0] == 3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_boundary_facets_and_normals_on_device(backend, device):
    """Regression: _find_boundary_facets()/_compute_normals() used to build
    their own bookkeeping tensors (cell_ids, missing_vertex_ids,
    boundary_normals_at_vertex, the 1D-case normals) on cpu regardless of
    where vertices/cells actually live, mismatching them the moment a
    non-cpu device was used."""
    vertices = backend.build_tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device
    )
    cells = backend.build_tensor([[0, 1, 2], [1, 3, 2]], device=device)
    m = Mesh(vertices=vertices, cells=cells, backend=backend, device=device)

    assert m.vertex_count == 4
    assert m.boundary_faces.shape[0] == 4
    assert m.boundary_normals.shape[0] == 4
    assert m.boundary_normals_at_vertex.shape[0] == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_tetrahedron(backend):
    vertices3 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    cells3 = [[0, 1, 2, 3]]
    m3 = Mesh(vertices=vertices3, cells=cells3, backend=backend)
    assert m3.vertex_count == 4


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_normals_point_outwards(backend):
    # Unit square triangulated into two triangles
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    mesh = Mesh(vertices=vertices, cells=cells, backend=backend)

    # there should be four boundary faces (the square edges)
    assert mesh.boundary_faces.shape[0] == 4

    # normals should point outwards (away from the center of the domain)
    center = backend.build_tensor([0.5, 0.5])
    centroids = backend.math.mean(mesh.vertices[mesh.boundary_faces], axis=1)

    inward_vectors = center - centroids
    dots = backend.math.sum(mesh.boundary_normals * inward_vectors, axis=1)

    for d in dots:
        try:
            dv = float(d)
        except Exception:
            dv = d.item() if hasattr(d, "item") else d
        assert dv < 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_simple_face_marking_strategy(backend):
    # Provide explicit face markers and ensure they are transferred
    vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    cells = [[0, 1, 2], [0, 2, 3]]
    # define all boundary faces explicitly and give markers
    faces = [[0, 1], [1, 2], [2, 3], [3, 0]]
    face_markers = [10, 11, 12, 13]

    mesh = Mesh(
        vertices=vertices,
        cells=cells,
        faces=faces,
        face_markers=face_markers,
        backend=backend,
    )

    b = mesh.get_boundary_mesh()
    assert b.cell_markers is not None
    # all boundary faces mapped to cell_markers
    try:
        markers = [int(x) for x in b.cell_markers]
    except Exception:
        markers = [
            int(x.item()) if hasattr(x, "item") else int(x) for x in b.cell_markers
        ]

    # Every marker should be one of the provided markers (no default marker used)
    print(face_markers, markers)
    assert all(m in face_markers for m in markers)


def _edge_list(backend, edges):
    return [tuple(int(i) for i in e) for e in backend.to_numpy(edges)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
def test_edges_of_cells(backend, device):
    # Two triangles share the edge (1, 2), which must appear only once.
    m = Mesh(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[0, 1, 2], [1, 3, 2]],
        backend=backend,
        device=device,
    )
    assert _edge_list(backend, m.edges) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    assert m.edges.dtype == m.cells.dtype
    assert backend.math.allclose(
        backend.to(m.edges, cpu), backend.to(backend.move_to_same_device(m.edges, m.cells), cpu)
    )
    assert m.edges is m.edges  # computed once


@pytest.mark.parametrize("backend", BACKENDS)
def test_edges_of_intervals_and_tetrahedra(backend):
    m1 = Mesh([[0.0], [0.5], [1.0]], [[1, 0], [1, 2]], backend=backend)
    assert _edge_list(backend, m1.edges) == [(0, 1), (1, 2)]

    m3 = Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 1, 2, 3]],
        backend=backend,
    )
    assert len(_edge_list(backend, m3.edges)) == 6


@pytest.mark.parametrize("backend", BACKENDS)
def test_set_edges(backend):
    # A mesh without cells, e.g. data that only provides its connectivity.
    m = Mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [], backend=backend)
    assert tuple(m.edges.shape) == (0, 2)

    # Both directions and self-loops are reduced to each edge once.
    m.set_edges([[1, 0], [0, 1], [2, 1], [1, 2], [2, 2]])
    assert _edge_list(backend, m.edges) == [(0, 1), (1, 2)]

    with pytest.raises(ValueError, match="vertices 0 to 2"):
        m.set_edges([[0, 3]])
    with pytest.raises(ValueError, match=r"shape \[E, 2\]"):
        m.set_edges([[0, 1, 2], [1, 2, 0]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_set_edges_discards_hierarchy(backend):
    m = Mesh(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[0, 1, 2], [1, 3, 2]],
        backend=backend,
    )
    levels = m.hierarchy(2)
    assert m.hierarchy(2) is levels
    m.set_edges([[0, 1], [2, 3]])
    assert m.hierarchy(2) is not levels
    assert m.hierarchy(2)[0].edge_count == 2
