import inspect

import numpy as np
import pytest

from qewton.backends.base import ComputingBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.geometries.discrete.meshes import Mesh, Level, build_hierarchy


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


def unit_square(n: int, backend, device=cpu) -> Mesh:
    """A triangulation of [0, 1]^2 with (n + 1)^2 vertices."""
    xs = np.linspace(0.0, 1.0, n + 1)
    vertices = [[x, y] for y in xs for x in xs]
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return Mesh(vertices, cells, backend=backend, device=device)


def edge_set(backend, edges) -> set:
    return {tuple(int(i) for i in e) for e in backend.to_numpy(edges)}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("random_order", [True, False])
def test_levels_are_consistent(backend, random_order):
    mesh = unit_square(16, backend)
    levels = mesh.hierarchy(4, random_order)

    assert len(levels) == 4
    assert all(isinstance(level, Level) for level in levels)
    assert levels[0].vertices is mesh.vertices
    assert edge_set(backend, levels[0].edges) == edge_set(backend, mesh.edges)
    assert levels[-1].cluster is None and levels[-1].cluster_size is None
    assert levels[-1].coarse_vertex_count == 0

    for fine, coarse in zip(levels[:-1], levels[1:]):
        cluster = backend.to_numpy(fine.cluster)
        positions = backend.to_numpy(fine.vertices)
        # Every fine vertex belongs to one coarse vertex, and every coarse
        # vertex has at least one fine vertex.
        assert cluster.shape == (fine.vertex_count,)
        assert fine.coarse_vertex_count == coarse.vertex_count < fine.vertex_count
        assert np.array_equal(
            np.bincount(cluster, minlength=coarse.vertex_count),
            backend.to_numpy(fine.cluster_size),
        )
        assert backend.to_numpy(fine.cluster_size).min() >= 1
        # A coarse vertex lies at the mean of its aggregate.
        for c in range(coarse.vertex_count):
            assert np.allclose(
                positions[cluster == c].mean(axis=0),
                backend.to_numpy(coarse.vertices)[c],
                atol=1e-6,
            )
        # Coarse vertices are connected iff a fine edge runs between them.
        expected = {
            tuple(sorted((int(cluster[a]), int(cluster[b]))))
            for a, b in edge_set(backend, fine.edges)
            if cluster[a] != cluster[b]
        }
        assert edge_set(backend, coarse.edges) == expected

    for level in levels:
        degree = np.bincount(
            backend.to_numpy(level.edges).ravel(), minlength=level.vertex_count
        )
        assert np.array_equal(degree, backend.to_numpy(level.degree))


@pytest.mark.parametrize("backend", BACKENDS)
def test_hierarchy_is_cached_per_depth_and_order(backend):
    mesh = unit_square(4, backend)
    assert mesh.hierarchy(3) is mesh.hierarchy(3, random_order=True)
    assert mesh.hierarchy(3, random_order=False) is mesh.hierarchy(3, random_order=False)
    assert mesh.hierarchy(3, random_order=False) is not mesh.hierarchy(3)
    assert len(mesh.hierarchy(2)) == 2
    assert mesh.hierarchy(1)[0].cluster is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_index_order_follows_the_numbering(backend):
    # In index order, the first aggregate is vertex 0 with its neighbours.
    mesh = unit_square(8, backend)
    cluster = backend.to_numpy(mesh.hierarchy(2, random_order=False)[0].cluster)
    first = {0} | {int(v) for e in backend.to_numpy(mesh.edges) if 0 in e for v in e}
    assert set(np.flatnonzero(cluster == 0).tolist()) >= first


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_hierarchy_from_graph(backend):
    # Without a mesh, e.g. from data given as vertices and an edge index;
    # edges in both directions and self-loops are allowed.
    vertices = backend.build_tensor([[0.0], [1.0], [2.0], [3.0]])
    edges = backend.build_tensor([[0, 1], [1, 0], [1, 2], [2, 3], [3, 3]])
    levels = build_hierarchy(vertices, edges, 2, backend)
    assert levels[0].edge_count == 3
    assert backend.to_numpy(levels[0].degree).tolist() == [1, 2, 2, 1]
    assert 1 <= levels[1].vertex_count < 4

    with pytest.raises(ValueError, match="at least one level"):
        build_hierarchy(vertices, edges, 0, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("random_order", [True, False])
def test_hierarchy_on_device(backend, device, random_order):
    mesh = unit_square(4, backend)
    levels = mesh.hierarchy(3, random_order)
    mesh.to(device)
    for level in levels:
        for tensor in (level.vertices, level.edges, level.degree):
            assert backend.math.allclose(
                backend.to(backend.move_to_same_device(tensor, mesh.vertices), cpu),
                backend.to(tensor, cpu),
            )

    built_on_device = unit_square(4, backend, device=device).hierarchy(3, random_order)
    for a, b in zip(levels, built_on_device):
        assert edge_set(backend, a.edges) == edge_set(backend, b.edges)
