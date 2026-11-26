from dataclasses import dataclass
from typing import List

import torch


# Normalisation
# Networks train best when inputs and outputs are in range [-1,1]. We divide f, g and u
# each by one dataset-wide RMS value (computed on the training set only).
# We deliberately only *scale* and do not shift: the problem is linear, so
# f = 0, g = 0 must give u = 0, and subtracting a mean would break that.


@dataclass
class Scales:
    f: float
    g: float
    u: float

def rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt().clamp_min(1e-8))

def compute_scales(samples: List[dict]) -> Scales:
    f = torch.cat([s["f"] for s in samples])
    g = torch.cat([s["g"][s["bmask"]] for s in samples])
    u = torch.cat([s["u"][~s["bmask"]] for s in samples])
    return Scales(f=rms(f), g=rms(g), u=rms(u))
#f, g and u are each divided by their root-mean-square value, computed on the training set only.

#node_features builds the input vector for each node: [f, g, is_boundary, x, y]
def node_features(s: dict, scales: Scales, use_pos: bool) -> torch.Tensor:
    """Per-node input vector: [f, g (0 inside), is_boundary, (x, y)]."""
    feats = [s["f"] / scales.f, s["g"] / scales.g, s["bmask"].float()]
    if use_pos: #use_pos=True
        feats += [s["coords"][:, 0], s["coords"][:, 1]]
    return torch.stack(feats, dim=-1)

