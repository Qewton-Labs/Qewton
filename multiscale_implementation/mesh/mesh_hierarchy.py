from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

# ## 2. The mesh hierarchy (the "multiscale" part)
#
# Greedy aggregation (from algebraic multigrid):
# 1. Walk over the nodes. If a node *and all its neighbours* are unassigned,
#    they form a new aggregate (= one coarse node).
# 2. Leftover nodes join the neighbouring aggregate they touch most often.
# Coarse node position = mean of its aggregate; two coarse nodes are connected
# if any fine edge runs between their aggregates.
#
# On your mesh one aggregate holds ~7-9 nodes: 1089 -> ~140 -> ~20 -> ~3.
#
# Important for holes: the coarsening only uses *graph connectivity*, never
# Euclidean distance. Since no mesh edge crosses a hole, no aggregate or coarse
# edge can jump across it either. (A "connect everything within radius r"
# coarsening would leak information across thin holes or slits.)


@dataclass
class Level:  # is a container for one graph level
    pos: torch.Tensor  # [N, 2] node positions on this level
    edge_index: torch.Tensor  # [2, E] (sender, receiver), both directions present
    edge_feat: torch.Tensor  # [E, 3] (dx, dy, |d|) / mean edge length of this level
    deg: torch.Tensor  # [N] number of incoming edges (>= 1), for mean aggregation
    cluster: Optional[torch.Tensor]  # [N] index of the coarse node each node belongs to
    n_coarse: int  # number of nodes on the next coarser level
    cluster_size: Optional[torch.Tensor]  # [n_coarse] aggregate sizes, for mean pooling

    @property
    def n_nodes(self) -> int:
        return self.pos.shape[0]

    def to(self, device) -> "Level":
        mv = lambda t: None if t is None else t.to(device)
        return Level(
            mv(self.pos),
            mv(self.edge_index),
            mv(self.edge_feat),
            mv(self.deg),
            mv(self.cluster),
            self.n_coarse,
            mv(self.cluster_size),
        )


# symmetrize makes every edge go both ways, removes self-loops, and removes duplicates.
def symmetrize(ei: np.ndarray) -> np.ndarray:
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    ei = ei[:, ei[0] != ei[1]]  # no self loops but we wanna keep self loops
    return np.unique(ei, axis=1)


# greedy_aggregation decides which fine nodes merge into one coarse node, using an algebraic-multigrid idea
def greedy_aggregation(ei: np.ndarray, n: int) -> np.ndarray:
    """Returns agg[n]: the aggregate (coarse node) id of every node.

    Args:
        ei: edge index array of shape (2, m) where m is the number of edges.
        n: number of nodes in the graph.
    Returns:
        agg: array of shape (n,) where agg[i] is the aggregate id of node i.
    """
    order = np.argsort(ei[0], kind="stable")  # order edges by source node
    nbrs = ei[1][order]  # ordered list of neighbours for each node
    ptr = np.searchsorted(
        ei[0][order], np.arange(n + 1)
    )  # ptr[i] is the index in nbrs where the neighbours of node i start
    agg = -np.ones(n, dtype=np.int64)
    n_agg = 0
    # Phase 1: go through the nodes. If a node and all its neighbours are still unassigned, they form a new group.
    for v in range(n):
        nb = nbrs[ptr[v] : ptr[v + 1]]  # retrieve the neighbours of node v
        if agg[v] < 0 and np.all(agg[nb] < 0):
            agg[v] = n_agg
            agg[nb] = n_agg
            n_agg += 1
    # Phase 2: each leftover node joins the neighbouring group it touches most often.
    # (A node only stays unassigned in phase 1 if a neighbour was already
    # assigned, so a candidate always exists.)
    phase1 = agg.copy()
    for v in np.flatnonzero(agg < 0):
        cand = phase1[nbrs[ptr[v] : ptr[v + 1]]]
        agg[v] = np.bincount(cand[cand >= 0]).argmax()
    return agg


# edge_features computes (dx, dy, length) for each edge, divided by the mean edge length
def edge_features(pos: np.ndarray, ei: np.ndarray) -> np.ndarray:
    d = pos[ei[1]] - pos[ei[0]]  # vector sender -> receiver
    length = np.linalg.norm(d, axis=1, keepdims=True)
    avg_length = (
        length.mean() if len(length) else 1.0
    )  # keeps features O(1) on every level
    return np.concatenate([d, length], axis=1) / avg_length


# build_hierarchy repeats this for n_levels levels:
# 1) Compute the node degrees and the aggregation.
# 2) Store the current level.
# 3) Build the next coarser graph:
#   * the new position is the mean position of each group,
#   * two coarse nodes are connected if any fine edge runs between their groups
#     (agg[ei] maps the edges, then self-loops and duplicates are removed).
def build_hierarchy(
    coords: torch.Tensor, edge_index: torch.Tensor, n_levels: int
) -> List[Level]:
    pos = coords.numpy().astype(np.float64)
    ei = symmetrize(edge_index.numpy())
    levels = []
    for lvl in range(n_levels):
        n = pos.shape[0]  # N number of nodes, pos.shape = (N, 2)
        deg = np.maximum(np.bincount(ei[1], minlength=n), 1)
        last = lvl == n_levels - 1
        if not last:
            agg = greedy_aggregation(ei, n)
            n_c = int(agg.max()) + 1  # number of coarse nodes
            size_c = np.bincount(
                agg, minlength=n_c
            )  # number of fine nodes in each coarse node
        levels.append(
            Level(
                pos=torch.tensor(pos, dtype=torch.float32),
                edge_index=torch.tensor(ei, dtype=torch.long),
                edge_feat=torch.tensor(edge_features(pos, ei), dtype=torch.float32),
                deg=torch.tensor(deg, dtype=torch.float32),
                cluster=None if last else torch.tensor(agg),
                n_coarse=0 if last else n_c,
                cluster_size=(
                    None if last else torch.tensor(size_c, dtype=torch.float32)
                ),
            )
        )
        if last:
            break
        # build the next (coarser) graph
        pos = (
            np.stack(
                [np.bincount(agg, weights=pos[:, k], minlength=n_c) for k in range(2)],
                1,
            )
            / size_c[:, None]
        )
        ei_c = agg[ei]
        ei = np.unique(
            ei_c[:, ei_c[0] != ei_c[1]], axis=1
        )  # remove self-loops and duplicates
    return levels


# describe_hierarchy prints the sizes
def describe_hierarchy(levels: List[Level]) -> str:
    return " -> ".join(
        f"{l.n_nodes} nodes/{l.edge_index.shape[1]} edges" for l in levels
    )


# plot_hierarchy draws the levels as a picture
def plot_hierarchy(levels: List[Level], path: str):
    import matplotlib

    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4))
    for i, (ax, l) in enumerate(zip(np.atleast_1d(axes), levels)):
        p, ei = l.pos.numpy(), l.edge_index.numpy()
        ax.add_collection(
            LineCollection(np.stack([p[ei[0]], p[ei[1]]], 1), lw=0.4, color="0.5")
        )
        if i > 0:  # colour fine nodes of the previous level by aggregate
            prev = levels[i - 1]
            ax.scatter(
                *prev.pos.numpy().T,
                c=prev.cluster.numpy() % 20,
                cmap="tab20",
                s=4,
                alpha=0.6,
            )
        ax.scatter(*p.T, s=12 if i else 3, color="k", zorder=3)
        ax.set_title(f"level {i}: {l.n_nodes} nodes")
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    # plt.close(fig)
