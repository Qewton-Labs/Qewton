"""Nested coarse graphs of a mesh, for multigrid-like models.

Every coarse level is built by greedy aggregation of the previous one, as in
algebraic multigrid, see :mod:`~qewton.geometries.discrete.meshes.aggregation`.
A coarse vertex lies at the mean of its aggregate, and two coarse vertices are
connected if any edge runs between their aggregates. On a 2D triangle mesh an
aggregate holds about 8-10 vertices, e.g. 1089 -> ~110 -> ~15 -> ~3 vertices.

Since no aggregate crosses a hole or a slit of the domain, no coarse edge does
either.
"""

from __future__ import annotations
from typing import Generic

from qewton.backends.base import TensorType, ComputingBackend
from qewton.config.dtypes import Float32, Int32, Int64
from qewton.config.saving.saving import Serializable
from qewton.geometries.discrete.meshes.aggregation import greedy_aggregation


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
    random_order: bool = True,
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
        random_order (bool): Whether the aggregation visits the vertices in a
            fixed pseudo-random order (fast), or else by increasing index (as
            in the multiscale prototype), see
            :func:`~qewton.geometries.discrete.meshes.aggregation.greedy_aggregation`.

    Returns:
        list[Level]: The levels, from the given graph to the coarsest.
    """
    if n_levels < 1:
        raise ValueError(f"A hierarchy needs at least one level, got {n_levels}.")
    b_math = backend.math
    edges = backend.move_to_same_device(backend.cast_dtype(edges, Int64), vertices)
    positions, graph_edges = vertices, unique_edges(edges, backend)
    levels = []
    for i in range(n_levels):
        n = int(positions.shape[0])
        coarsest = i == n_levels - 1
        cluster = None if coarsest else greedy_aggregation(graph_edges, n, backend, random_order)
        degree = b_math.bincount(b_math.reshape(graph_edges, (-1,)), minlength=n)
        levels.append(
            Level(
                vertices=positions,
                edges=backend.cast_dtype(graph_edges, Int32),
                degree=backend.cast_dtype(degree, Int32),
                cluster=None if cluster is None else backend.cast_dtype(cluster, Int32),
                cluster_size=(
                    None
                    if cluster is None
                    else backend.cast_dtype(b_math.bincount(cluster), Int32)
                ),
                backend=backend,
            )
        )
        if not coarsest:
            positions, graph_edges = coarsen(positions, graph_edges, cluster, backend)
    return levels


def unique_edges(edges: TensorType, backend: type[ComputingBackend]) -> TensorType:
    """The undirected edges of `edges` [E, 2], each once with the smaller vertex
    index first, sorted, and without self-loops."""
    b_math = backend.math
    edges = b_math.sort(b_math.reshape(edges, (-1, 2)), axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return b_math.reshape(b_math.unique(edges, axis=0), (-1, 2))


def coarsen(
    positions: TensorType,
    edges: TensorType,
    cluster: TensorType,
    backend: type[ComputingBackend],
) -> tuple[TensorType, TensorType]:
    """The coarse graph of an aggregation.

    Args:
        positions (TensorType): The vertex positions, shape [N, d].
        edges (TensorType): The undirected edges, shape [E, 2], each once.
        cluster (TensorType): The aggregate of each vertex, shape [N].
        backend (type[ComputingBackend]): The backend of the tensors.

    Returns:
        tuple[TensorType, TensorType]: The positions of the coarse vertices
            (the mean of their aggregate), shape [N_coarse, d], and the coarse
            edges, shape [E_coarse, 2], each once.
    """
    b_math = backend.math
    size = b_math.bincount(cluster)
    sums = b_math.stack(
        [
            b_math.bincount(cluster, weights=positions[:, k], minlength=size.shape[0])
            for k in range(positions.shape[1])
        ],
        axis=1,
    )
    coarse_positions = backend.cast_dtype(sums / size[:, None], Float32)
    return coarse_positions, unique_edges(cluster[edges], backend)
