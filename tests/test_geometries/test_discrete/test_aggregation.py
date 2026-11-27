import inspect

import numpy as np
import pytest

from qewton.backends.base import ComputingBackend
from qewton.config.devices import cpu, cuda, cuda_available
from qewton.config.dtypes import Int64
from qewton.geometries.discrete.meshes.aggregation import (
    greedy_aggregation,
    parallel_aggregation,
    sequential_aggregation,
    vertex_priority,
)
from qewton.geometries.discrete.meshes.hierarchy import unique_edges


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)
devices = [cpu, cuda(0)] if cuda_available() else [cpu]


def grid_edges(n: int) -> np.ndarray:
    """The edges of a triangulation of an (n + 1) x (n + 1) grid of vertices,
    numbered row by row."""
    index = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, 1:].ravel(), index[1:, :-1].ravel()
    pairs = [(a, b), (b, c), (a, c), (a, d), (d, c)]
    return np.concatenate([np.stack(pair, axis=1) for pair in pairs])


def sequential_greedy(edges: np.ndarray, n: int, order: np.ndarray) -> np.ndarray:
    """Greedy aggregation as written in the module docstring, one vertex at a
    time in `order`, with aggregates numbered by their roots' indices."""
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    root_of = np.full(n, -1)
    for v in order:
        if root_of[v] < 0 and all(root_of[u] < 0 for u in neighbours[v]):
            root_of[v] = v
            root_of[neighbours[v]] = v
    number = {r: i for i, r in enumerate(sorted(set(root_of[root_of >= 0])))}
    phase1 = np.array([number[r] if r >= 0 else -1 for r in root_of])
    cluster = phase1.copy()
    for v in np.flatnonzero(phase1 < 0):
        joined = [phase1[u] for u in neighbours[v] if phase1[u] >= 0]
        cluster[v] = np.bincount(joined).argmax()
    return cluster


def graphs(n_random: int = 100) -> list[tuple[np.ndarray, int]]:
    """Grids and random graphs, including isolated vertices, self-loops,
    repeated edges and several components."""
    rng = np.random.default_rng(0)
    result = [(grid_edges(n), (n + 1) ** 2) for n in (1, 3, 16)]
    for _ in range(n_random):
        n = int(rng.integers(1, 40))
        result.append((rng.integers(0, n, size=(int(rng.integers(0, 3 * n)), 2)), n))
    return result


def tensor_edges(backend, edges: np.ndarray, device=cpu):
    edges = backend.build_tensor(edges.reshape(-1, 2), dtype=Int64)
    return unique_edges(backend.to(edges, device), backend)


def aggregate(backend, edges: np.ndarray, n: int, random_order: bool, device=cpu) -> np.ndarray:
    edges = tensor_edges(backend, edges, device)
    return backend.to_numpy(greedy_aggregation(edges, n, backend, random_order))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("random_order", [True, False])
def test_aggregation_equals_sequential_greedy(backend, random_order):
    any_tensor = backend.build_tensor([0], dtype=Int64)
    for edges, n in graphs():
        unique = backend.to_numpy(tensor_edges(backend, edges))
        if random_order:
            visit = np.argsort(backend.to_numpy(vertex_priority(n, backend, any_tensor)))
        else:
            visit = np.arange(n)
        expected = sequential_greedy(unique, n, visit)
        assert np.array_equal(aggregate(backend, edges, n, random_order), expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_parallel_aggregation_in_index_order_equals_sequential(backend):
    # Both implementations give the same aggregates for the same order, only
    # the parallel one needs many rounds in this order.
    for edges, n in graphs(n_random=20):
        edges = tensor_edges(backend, edges)
        index = backend.cast_dtype(backend.math.arange(n), Int64)
        assert np.array_equal(
            backend.to_numpy(parallel_aggregation(edges, n, backend, index)),
            backend.to_numpy(sequential_aggregation(edges, n, backend)),
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_index_order_walks_along_a_path(backend):
    # Along the path 0 - 1 - ... - 8: {0, 1}, then 3 with {2, 4}, then 6 with
    # {5, 7}; vertex 8 is left over and joins its neighbour's aggregate.
    edges = np.array([[i, i + 1] for i in range(8)])
    assert aggregate(backend, edges, 9, random_order=False).tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize("random_order", [True, False])
def test_aggregation_is_the_same_on_all_backends(random_order):
    clusters = [aggregate(backend, grid_edges(12), 169, random_order) for backend in BACKENDS]
    assert all(np.array_equal(clusters[0], cluster) for cluster in clusters)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("random_order", [True, False])
@pytest.mark.parametrize("device", devices)
def test_aggregation_on_device(backend, random_order, device):
    edges = tensor_edges(backend, grid_edges(8), device)
    cluster = greedy_aggregation(edges, 81, backend, random_order)
    assert getattr(cluster, "device", None) == getattr(edges, "device", None)
    assert np.array_equal(backend.to_numpy(cluster), aggregate(backend, grid_edges(8), 81, random_order))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("random_order", [True, False])
def test_aggregation_keeps_components_apart(backend, random_order):
    # Two triangles without an edge between them (e.g. across a slit), and an
    # isolated vertex.
    edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])
    assert aggregate(backend, edges, 7, random_order).tolist() == [0, 0, 0, 1, 1, 1, 2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_vertex_priority_is_a_permutation(backend):
    priority = backend.to_numpy(
        vertex_priority(1000, backend, backend.build_tensor([0], dtype=Int64))
    )
    assert sorted(priority.tolist()) == list(range(1000))
    assert priority.tolist() != list(range(1000))
