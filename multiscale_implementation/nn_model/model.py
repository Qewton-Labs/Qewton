import torch
import torch.nn as nn
from ..data_loading.normalization import Scales
from ..mesh.mesh_hierarchy import Level
from .basic_building_blocks import MPBlock, MLP
from typing import List

# **pool** (fine -> coarse): mean over the aggregate.
# **unpool** (coarse -> fine): each fine node gets a copy of its coarse node.
# On the way up, the coarse information (smooth, global part of the solution)
# is merged with the skip connection (local detail) by an MLP.


def pool(h, level: Level):
    out = h.new_zeros(h.shape[0], level.n_coarse, h.shape[2]).index_add(
        1, level.cluster, h
    )
    return out / level.cluster_size[None, :, None]


def unpool(h_coarse, level: Level):
    return h_coarse[:, level.cluster]


class MultiscaleGNN(nn.Module):
    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        hidden: int = 16,
        n_levels: int = 4,
        mp_per_level: int = 2,
        out_dim: int = 1,
    ):
        """Multiscale GNN with message-passing blocks at each level of a hierarchy.
        Args:
            in_dim_node: number of input features per node
            in_dim_edge: number of input features per edge
            hidden: number of hidden features per node and edge
            n_levels: number of levels in the hierarchy (>= 2)
            mp_per_level: number of message-passing blocks per level
            out_dim: number of output features per node
        """
        super().__init__()
        self.n_levels = n_levels
        blocks = lambda: nn.ModuleList([MPBlock(hidden) for _ in range(mp_per_level)])
        self.node_enc = MLP(in_dim_node, hidden, hidden)
        self.edge_enc = nn.ModuleList(
            [MLP(in_dim_edge, hidden, hidden) for _ in range(n_levels)]
        )
        self.down = nn.ModuleList([blocks() for _ in range(n_levels - 1)])
        self.bottom = blocks()
        self.merge = nn.ModuleList(
            [MLP(2 * hidden, hidden, hidden) for _ in range(n_levels - 1)]
        )  # skip connection merge
        self.up = nn.ModuleList([blocks() for _ in range(n_levels - 1)])
        self.decoder = MLP(hidden, out_dim, hidden, layernorm=False)

    def forward(self, x, levels: List[Level]):
        assert len(levels) == self.n_levels, "hierarchy depth must match the model"
        B = x.shape[0]
        h = self.node_enc(x)  # [B, N0, H]
        skips = []
        for l in range(self.n_levels - 1):  # ---- down path
            e = self.edge_enc[l](levels[l].edge_feat).expand(B, -1, -1)
            for blk in self.down[l]:
                h, e = blk(h, e, levels[l])
            skips.append((h, e))
            h = pool(h, levels[l])
        e = self.edge_enc[-1](levels[-1].edge_feat).expand(B, -1, -1)
        for blk in self.bottom:  # ---- coarsest level
            h, e = blk(h, e, levels[-1])
        for l in reversed(range(self.n_levels - 1)):  # ---- up path
            h_skip, e = skips[l]
            h = self.merge[l](torch.cat([h_skip, unpool(h, levels[l])], dim=-1))
            for blk in self.up[l]:
                h, e = blk(h, e, levels[l])
        return self.decoder(h)  # [B, N0, out_dim]


##  o -----> o
##  \       /
##   o -> o
##   \   /
##     o


def predict(model, x, g, bmask, levels, scales: Scales):
    u_int = model(x, levels) * scales.u
    # print("shapes", u_int.shape, g.shape, bmask.shape, x.shape)
    return torch.where(
        bmask.unsqueeze(-1), g.unsqueeze(-1), u_int
    )  # if bmask is True, return g (boundary value), else return u_int (predicted interior value)


def relative_l2(pred, true, interior):
    """Per-sample relative L2 error over interior nodes, shape [B]."""
    # interior multiplication masks out boundary nodes, which are not included in the loss.
    pred = pred.squeeze(-1)
    num = ((pred - true) ** 2 * interior).sum(-1)
    den = (true**2 * interior).sum(-1).clamp_min(1e-12)
    return (num / den).sqrt()
