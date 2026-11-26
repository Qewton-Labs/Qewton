"""Nested coarse graphs of a mesh, for multiscale (multigrid-like) models.

The coarse levels are built by greedy aggregation, as in algebraic multigrid:

1. Walk over the vertices. If a vertex *and all its neighbours* are not yet
   assigned, they form a new aggregate (one vertex of the coarser level).
2. Every vertex left over joins the neighbouring aggregate it touches most often.

A coarse vertex lies at the mean of its aggregate, and two coarse vertices are
connected if any edge runs between their aggregates. On a 2D triangle mesh an
aggregate holds about 7-9 vertices, e.g. 1089 -> ~140 -> ~20 -> ~3 vertices.

Only the connectivity of the graph is used, never Euclidean distances. Since no
mesh edge crosses a hole or a slit of the domain, no aggregate or coarse edge
does either.
"""

from __future__ import annotations
from typing import Generic

import numpy as np

from qewton.backends.base import TensorType, ComputingBackend
from qewton.config.dtypes import Float32, Int32
from qewton.config.saving.saving import Serializable


class Level(Serializable, Generic[TensorType]):
    """One graph of a mesh hierarchy, see :meth:`Mesh.hierarchy` and
    :func:`build_hierarchy`.

    Args:
        vertices (TensorType): The vertex positions, shape [N, d].
        edges (TensorType): The undirected edges, shape [E, 2], each once with
            the smaller vertex index first.
        degree (TensorType): The number of neighbours of each vertex, shape [N].
        cluster (TensorType | None): For each vertex, the index of the vertex
            of the next coarser level it belongs to, shape [N]. None on the
            coarsest level.
        cluster_size (TensorType | None): The number of vertices of this level
            in each vertex of the next coarser level, shape [N_coarse]. None on
            the coarsest level.
        backend (type[ComputingBackend[TensorType]]): The backend of the tensors.
    """

    def __init__(
        self,
        vertices: TensorType,
        edges: TensorType,
        degree: TensorType,
        cluster: TensorType | None,
        cluster_size: TensorType | None,
        backend: type[ComputingBackend[TensorType]],
    ) -> None:
        self.backend = backend
        self.vertices = vertices
        self.edges = edges
        self.degree = degree
        self.cluster = cluster
        self.cluster_size = cluster_size

    @property
    def vertex_count(self) -> int:
        """The number of vertices of this level."""
        return int(self.vertices.shape[0])

    @property
    def edge_count(self) -> int:
        """The number of (undirected) edges of this level."""
        return int(self.edges.shape[0])

    @property
    def coarse_vertex_count(self) -> int:
        """The number of vertices of the next coarser level, 0 on the coarsest."""
        return 0 if self.cluster_size is None else int(self.cluster_size.shape[0])

    def to(self, device) -> Level:
        """Move all data of this level to `device`.

        Returns:
            Level: This level.
        """
        for name in ("vertices", "edges", "degree", "cluster", "cluster_size"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, self.backend.to(value, device))
        return self

    def __repr__(self) -> str:
        return f"Level({self.vertex_count} vertices, {self.edge_count} edges)"


def build_hierarchy(
    vertices: TensorType,
    edges: TensorType,
    n_levels: int,
    backend: type[ComputingBackend[TensorType]],
) -> list[Level]:
    """Build `n_levels` nested graphs by repeated greedy aggregation.

    Level 0 is the given graph (sharing the `vertices` tensor); every further
    level is the coarsening of the previous one. All tensors are on the device
    of `vertices`.

    Args:
        vertices (TensorType): The vertex positions, shape [N, d].
        edges (TensorType): The edges, shape [E, 2]. Each edge may appear in
            one or both directions; self-loops are ignored.
        n_levels (int): The number of levels, including the given graph.
        backend (type[ComputingBackend[TensorType]]): The backend of the tensors.

    Returns:
        list[Level]: The levels, from the given graph to the coarsest.
    """
    if n_levels < 1:
        raise ValueError(f"A hierarchy needs at least one level, got {n_levels}.")

    def tensor(data: np.ndarray, dtype) -> TensorType:
        return backend.move_to_same_device(backend.build_tensor(data, dtype=dtype), vertices)

    positions = np.asarray(backend.to_numpy(vertices), dtype=np.float64)
    graph_edges = unique_edges(np.asarray(backend.to_numpy(edges)).reshape(-1, 2))
    levels = []
    for i in range(n_levels):
        n = positions.shape[0]
        coarsest = i == n_levels - 1
        cluster = None if coarsest else greedy_aggregation(graph_edges, n)
        cluster_size = None if cluster is None else np.bincount(cluster)
        levels.append(
            Level(
                vertices=vertices if i == 0 else tensor(positions, Float32),
                edges=tensor(graph_edges, Int32),
                degree=tensor(np.bincount(graph_edges.ravel(), minlength=n), Int32),
                cluster=None if cluster is None else tensor(cluster, Int32),
                cluster_size=None if cluster_size is None else tensor(cluster_size, Int32),
                backend=backend,
            )
        )
        if coarsest:
            break
        positions, graph_edges = coarsen(positions, graph_edges, cluster)
    return levels


def unique_edges(edges: np.ndarray) -> np.ndarray:
    """The undirected edges of `edges` [E, 2], each once with the smaller vertex
    index first, sorted, and without self-loops."""
    edges = np.sort(edges.astype(np.int64), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0).reshape(-1, 2)


def neighbours(edges: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    """The adjacency of an undirected graph in compressed form: the neighbours
    of vertex v are `nbrs[ptr[v]:ptr[v + 1]]`.

    Args:
        edges (np.ndarray): The undirected edges, shape [E, 2], each once.
        vertex_count (int): The number of vertices.

    Returns:
        tuple[np.ndarray, np.ndarray]: nbrs [2E] and ptr [vertex_count + 1].
    """
    senders = np.concatenate([edges[:, 0], edges[:, 1]])
    receivers = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(senders, kind="stable")
    ptr = np.searchsorted(senders[order], np.arange(vertex_count + 1))
    return receivers[order], ptr


def greedy_aggregation(edges: np.ndarray, vertex_count: int) -> np.ndarray:
    """Assign every vertex to an aggregate, see the module docstring.

    Args:
        edges (np.ndarray): The undirected edges, shape [E, 2], each once.
        vertex_count (int): The number of vertices.

    Returns:
        np.ndarray: The aggregate of each vertex, shape [vertex_count], with
            the aggregates numbered 0, 1, ... in the order they were formed.
    """
    nbrs, ptr = neighbours(edges, vertex_count)
    cluster = np.full(vertex_count, -1, dtype=np.int64)
    count = 0
    # Phase 1: a vertex whose neighbourhood is still free forms a new aggregate
    # with it. An isolated vertex forms an aggregate of its own.
    for v in range(vertex_count):
        nb = nbrs[ptr[v] : ptr[v + 1]]
        if cluster[v] < 0 and np.all(cluster[nb] < 0):
            cluster[v] = count
            cluster[nb] = count
            count += 1
    # Phase 2: a vertex left over had an assigned neighbour when phase 1
    # reached it, so it always has an aggregate to join.
    phase1 = cluster.copy()
    for v in np.flatnonzero(cluster < 0):
        candidates = phase1[nbrs[ptr[v] : ptr[v + 1]]]
        cluster[v] = np.bincount(candidates[candidates >= 0]).argmax()
    return cluster


def coarsen(
    positions: np.ndarray, edges: np.ndarray, cluster: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The coarse graph of an aggregation.

    Args:
        positions (np.ndarray): The vertex positions, shape [N, d].
        edges (np.ndarray): The undirected edges, shape [E, 2], each once.
        cluster (np.ndarray): The aggregate of each vertex, shape [N].

    Returns:
        tuple[np.ndarray, np.ndarray]: The positions of the coarse vertices
            (the mean of their aggregate), shape [N_coarse, d], and the coarse
            edges, shape [E_coarse, 2], each once.
    """
    n_coarse = int(cluster.max()) + 1
    coarse_positions = np.zeros((n_coarse, positions.shape[1]))
    np.add.at(coarse_positions, cluster, positions)
    coarse_positions /= np.bincount(cluster, minlength=n_coarse)[:, None]
    return coarse_positions, unique_edges(cluster[edges])
