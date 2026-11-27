"""Greedy aggregation of the vertices of a graph, as in algebraic multigrid:

1. Visit the vertices in a fixed order. If a vertex *and all its neighbours*
   are not yet assigned, they form a new aggregate (one vertex of the coarser
   level), with the vertex as its root.
2. Every vertex left over joins the neighbouring aggregate it touches most
   often, the one with the smaller number on a tie.

Two visiting orders are available, both giving the same aggregates in every
run:

- ``random_order=True`` (default): a fixed pseudo-random order, see
  :func:`vertex_priority`. The roots are then found in a few rounds of
  operations on the whole graph (:func:`parallel_aggregation`), fast also on
  a GPU.
- ``random_order=False``: by increasing vertex index, one vertex at a time on
  the CPU (:func:`sequential_aggregation`), as in the multiscale prototype. On
  a mesh numbered row by row, the aggregates follow the rows.

Only the connectivity of the graph is used, never Euclidean distances. Since no
mesh edge crosses a hole or a slit of the domain, no aggregate does either.
"""

from __future__ import annotations

from qewton.backends.base import TensorType, ComputingBackend
from qewton.config.devices import cpu
from qewton.config.dtypes import Bool, Float32, Int64


def greedy_aggregation(
    edges: TensorType,
    vertex_count: int,
    backend: type[ComputingBackend],
    random_order: bool = True,
) -> TensorType:
    """Assign every vertex to an aggregate, see the module docstring.

    Args:
        edges (TensorType): The undirected edges, shape [E, 2], each once, as
            integers of 64 bits.
        vertex_count (int): The number of vertices.
        backend (type[ComputingBackend]): The backend of `edges`.
        random_order (bool): Whether the vertices are visited in a fixed
            pseudo-random order, or else by increasing index.

    Returns:
        TensorType: The aggregate of each vertex, shape [vertex_count], on the
            device of `edges`, with the aggregates numbered 0, 1, ... in the
            order of their roots' indices.
    """
    if not random_order:
        return sequential_aggregation(edges, vertex_count, backend)
    priority = vertex_priority(vertex_count, backend, edges)
    return parallel_aggregation(edges, vertex_count, backend, priority)


def sequential_aggregation(
    edges: TensorType, vertex_count: int, backend: type[ComputingBackend]
) -> TensorType:
    """Greedy aggregation visiting the vertices by increasing index, one at a
    time. Runs on the CPU, since one step per vertex is slow on a GPU: the
    edges are moved there, and the result is moved back to their device.

    Args:
        edges (TensorType): The undirected edges, shape [E, 2], each once, as
            integers of 64 bits.
        vertex_count (int): The number of vertices.
        backend (type[ComputingBackend]): The backend of `edges`.

    Returns:
        TensorType: The aggregate of each vertex, shape [vertex_count], on the
            device of `edges`, with the aggregates numbered 0, 1, ... in the
            order they were formed.
    """
    b_math = backend.math
    neighbourhoods = Neighbourhoods(backend.to(edges, cpu), vertex_count, backend)
    # Sorted by centre, N[v] are the members from start[v] to start[v + 1].
    members = neighbourhoods.members[b_math.argsort(neighbourhoods.centres)]
    start = backend.to_numpy(neighbourhoods.start).tolist() + [int(members.shape[0])]

    def neighbourhood(v: int) -> TensorType:
        return members[start[v] : start[v + 1]]

    cluster = b_math.full_like(neighbourhoods.index, -1)
    count = 0
    # Phase 1: a vertex whose neighbourhood (itself included) is still free
    # forms a new aggregate with it. An isolated vertex forms one of its own.
    for v in range(vertex_count):
        if bool(b_math.all(cluster[neighbourhood(v)] < 0)):
            cluster[neighbourhood(v)] = count
            count += 1
    # Phase 2: a vertex left over had an assigned neighbour when phase 1
    # reached it, so it always has an aggregate to join.
    phase1 = b_math.copy(cluster)
    for v in backend.to_numpy(neighbourhoods.index[cluster < 0]).tolist():
        candidates = phase1[neighbourhood(v)]
        cluster[v] = b_math.argmax(b_math.bincount(candidates[candidates >= 0]))
    return backend.move_to_same_device(cluster, edges)


class Neighbourhoods:
    """The closed neighbourhoods N[v] = {v} + the neighbours of v of an
    undirected graph, to reduce values over all of them at once.

    Stored as one (centre v, member u) pair for every u in N[v].

    Args:
        edges (TensorType): The undirected edges, shape [E, 2], each once, as
            integers of 64 bits.
        vertex_count (int): The number of vertices.
        backend (type[ComputingBackend]): The backend of `edges`.
    """

    def __init__(
        self, edges: TensorType, vertex_count: int, backend: type[ComputingBackend]
    ) -> None:
        b_math = backend.math
        self.backend = backend
        self.vertex_count = vertex_count
        #: The vertex indices 0, ..., N - 1, on the device of `edges`.
        self.index = backend.move_to_same_device(
            backend.cast_dtype(b_math.arange(vertex_count), Int64), edges
        )
        self.centres = b_math.concatenate([edges[:, 0], edges[:, 1], self.index])
        self.members = b_math.concatenate([edges[:, 1], edges[:, 0], self.index])
        # Where each neighbourhood starts once the pairs are sorted by centre.
        sizes = b_math.bincount(self.centres, minlength=vertex_count)
        self.start = b_math.cumsum(sizes) - sizes

    def min(self, values: TensorType, hops: int = 1) -> TensorType:
        """The smallest of `values`, integers in [0, N], within `hops` edges
        of every vertex."""
        bound = self.vertex_count + 1
        for _ in range(hops):
            # Sorted by centre and then by value, every neighbourhood starts
            # with its smallest value.
            keys = self.backend.math.sort(self.centres * bound + values[self.members])
            values = keys[self.start] - self.index * bound
        return values

    def any(self, mask: TensorType, hops: int = 1) -> TensorType:
        """Whether `mask` holds for any vertex within `hops` edges of every
        vertex."""
        for _ in range(hops):
            hits = self.backend.math.bincount(
                self.centres,
                weights=self.backend.cast_dtype(mask[self.members], Float32),
                minlength=self.vertex_count,
            )
            mask = hits > 0
        return mask


def parallel_aggregation(
    edges: TensorType,
    vertex_count: int,
    backend: type[ComputingBackend],
    priority: TensorType,
) -> TensorType:
    """Greedy aggregation visiting the vertices by increasing `priority`.

    Instead of one vertex at a time, :func:`find_roots` finds the same roots
    in a few rounds of operations on the whole graph, if the order is
    shuffled, see :func:`vertex_priority`.

    Args:
        edges (TensorType): The undirected edges, shape [E, 2], each once, as
            integers of 64 bits.
        vertex_count (int): The number of vertices.
        backend (type[ComputingBackend]): The backend of `edges`.
        priority (TensorType): The position of each vertex in the visiting
            order, a permutation of 0, ..., N - 1 of 64-bit integers.

    Returns:
        TensorType: The aggregate of each vertex, shape [vertex_count], with
            the aggregates numbered 0, 1, ... in the order of their roots'
            indices.
    """
    neighbourhoods = Neighbourhoods(edges, vertex_count, backend)
    root = find_roots(neighbourhoods, priority)
    cluster = assign_to_roots(neighbourhoods, root)
    return join_most_frequent(neighbourhoods, cluster)


def find_roots(neighbourhoods: Neighbourhoods, priority: TensorType) -> TensorType:
    """The roots of phase 1, visiting the vertices by increasing `priority`.

    Visited one at a time, a vertex becomes a root if no root lies within two
    edges of it, since only then are it and all its neighbours unassigned.
    So, in every round, an undecided vertex that comes first among the
    undecided vertices within two edges becomes a root: all vertices nearby
    that come before it are decided not to be roots. Every vertex within two
    edges of a new root is then decided not to be one.

    Returns:
        TensorType: Whether each vertex is a root, shape [N].
    """
    backend = neighbourhoods.backend
    b_math = backend.math
    n = neighbourhoods.vertex_count
    undecided = backend.cast_dtype(b_math.ones_like(priority), Bool)
    root = ~undecided
    while bool(b_math.any(undecided)):
        # Decided vertices come after all others, with priority N.
        candidates = b_math.where(undecided, priority, n)
        new_roots = undecided & (neighbourhoods.min(candidates, hops=2) == priority)
        root = root | new_roots
        undecided = undecided & ~neighbourhoods.any(new_roots, hops=2)
    return root


def assign_to_roots(neighbourhoods: Neighbourhoods, root: TensorType) -> TensorType:
    """The aggregates of phase 1: every root with its neighbours, numbered in
    the order of the roots' indices.

    Returns:
        TensorType: The aggregate of each vertex, or N for a vertex left over,
            shape [N]. Roots are at least three edges apart, so no vertex is
            next to two of them.
    """
    backend = neighbourhoods.backend
    b_math = backend.math
    number = b_math.cumsum(backend.cast_dtype(root, Int64)) - 1
    return neighbourhoods.min(b_math.where(root, number, neighbourhoods.vertex_count))


def join_most_frequent(neighbourhoods: Neighbourhoods, cluster: TensorType) -> TensorType:
    """Phase 2: every vertex left over (with aggregate N in `cluster`) joins
    the aggregate most frequent among its neighbours, the smallest on a tie.

    Returns:
        TensorType: `cluster`, with the vertices left over assigned.
    """
    b_math = neighbourhoods.backend.math
    centres, members = neighbourhoods.centres, neighbourhoods.members
    left = cluster == neighbourhoods.vertex_count
    # Every neighbour in an aggregate votes for it. Each vertex left over has
    # such a neighbour, else it would have become a root.
    is_vote = left[centres] & ~left[members]
    if not bool(b_math.any(is_vote)):
        return cluster
    votes = b_math.stack([centres[is_vote], cluster[members[is_vote]]], axis=1)
    votes, count = b_math.unique(votes, axis=0, return_counts=True)
    # Sorted by vertex, then by most votes, then by aggregate, the first row of
    # every vertex holds the aggregate it joins. (The rows are distinct, so
    # `unique` only sorts them.)
    ranking = b_math.stack([votes[:, 0], -count, votes[:, 1]], axis=1)
    ranking = b_math.unique(ranking, axis=0)
    _, first = b_math.unique(ranking[:, 0], return_index=True)
    cluster[ranking[first, 0]] = ranking[first, 2]
    return cluster


def vertex_priority(
    vertex_count: int, backend: type[ComputingBackend], like: TensorType
) -> TensorType:
    """The position of every vertex in the order in which
    :func:`parallel_aggregation` visits them: sorted by a hash of the index, so
    pseudo-random, but the same on every backend and device.

    In this order, :func:`find_roots` needs 4-6 rounds for 2D meshes of
    10^3 to 10^6 vertices. In the order of the indices, a mesh numbered row by
    row needs many more (171 for 66k vertices), since the roots are then found
    one after another along the rows.

    Args:
        vertex_count (int): The number of vertices.
        backend (type[ComputingBackend]): The backend of the result.
        like (TensorType): A tensor on the device of the result.

    Returns:
        TensorType: The position of each vertex, a permutation of 0, ..., N - 1
            of 64-bit integers.
    """
    b_math = backend.math
    index = backend.move_to_same_device(
        backend.cast_dtype(b_math.arange(vertex_count), Int64), like
    )
    # Ties of the hash are broken by the index.
    order = b_math.argsort(hash32(index, backend) * vertex_count + index)
    return b_math.argsort(order)


def hash32(x: TensorType, backend: type[ComputingBackend]) -> TensorType:
    """A fixed pseudo-random hash of 64-bit integers in [0, 2^32), with values
    in [0, 2^32)."""
    b_math = backend.math
    for _ in range(2):
        x = b_math.bitwise_xor(x, b_math.right_shift(x, 16)) * 0x45D9F3B
        x = b_math.bitwise_and(x, 0xFFFFFFFF)
    return b_math.bitwise_xor(x, b_math.right_shift(x, 16))
