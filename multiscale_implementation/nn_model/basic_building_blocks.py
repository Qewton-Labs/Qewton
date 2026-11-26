import torch
import torch.nn as nn
from ..mesh.mesh_hierarchy import Level

# 3. The model
#
# **Message-passing block**, with latent h_i per node and e_ij per edge:
#
#     e_ij <- e_ij + MLP_e([e_ij, h_i, h_j])     edge update (what flows from j to i)
#     h_i  <- h_i  + MLP_n([h_i, mean_j e_ij])   node update (collect incoming messages)
#
# Residual updates make deep stacks easy to train. The *mean* (not sum) of
# messages makes the block behave the same for nodes with 3 or 8 neighbours.
# A batch holds several samples on the *same* mesh: tensors are [B, N, H],
# `h[:, src]` gathers senders, `index_add(1, dst, ...)` sums at receivers.


# %%
class MLP(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, hidden: int, layernorm: bool = True):
        layers = [nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out)]
        if layernorm:
            layers.append(nn.LayerNorm(d_out))
        super().__init__(*layers)


class MPBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden, hidden, hidden)
        self.node_mlp = MLP(2 * hidden, hidden, hidden)

    def forward(self, h, e, level: Level):
        src, dst = level.edge_index
        e = e + self.edge_mlp(torch.cat([e, h[:, src], h[:, dst]], dim=-1))
        msg = torch.zeros_like(h).index_add(1, dst, e) / level.deg[None, :, None]
        h = h + self.node_mlp(torch.cat([h, msg], dim=-1))
        return h, e
