import glob
import hashlib
import os
from typing import List

import numpy as np
import torch

# The architecture is a graph U-Net.
# It goes down through coarser and coarser graphs,
# then back up, with skip connections between matching levels.

# Each sample file is a dict with `coords [N,2]`, `edge_index [2,E]`,
# `boundary_mask [N,1]`, `boundary_values [N,1]`, `f_values [N,1]`,
# `solution [N,1]`. We only use g on boundary nodes.

# Samples on the same mesh share one graph hierarchy, so we fingerprint the mesh
# (coordinates + connectivity). A dataset may later contain several meshes
# (other resolutions, holes, ...) and each gets its own hierarchy automatically.


# when we go coarser mesh, it's expensive to compute which nodes and edges gonna connect with the coarser mesh.
# So we compute the hierarchy once per mesh and share it across samples.


# Samples on the same mesh get the same key, so they can share one graph hierarchy.
def mesh_key(coords: torch.Tensor, edge_index: torch.Tensor) -> str:
    h = hashlib.sha1()
    h.update(coords.numpy().astype(np.float32).tobytes())
    h.update(edge_index.numpy().astype(np.int64).tobytes())
    return h.hexdigest()[:16]


def load_sample(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=True)
    coords = d["coords"].float()
    edge_index = d["edge_index"].long()
    bmask = d["boundary_mask"].reshape(-1).bool()
    g = (
        d["boundary_values"].reshape(-1).float() * bmask
    )  # g – boundary values, multiplied by bmask so they are zero inside the domain
    f = d["f_values"].reshape(-1).float()
    u = d["solution"].reshape(-1).float() if "solution" in d else None
    return dict(
        path=str(path),
        coords=coords,
        edge_index=edge_index,
        bmask=bmask,
        f=f,
        g=g,
        u=u,
        mesh_key=mesh_key(coords, edge_index),
    )


# load_folder loads all files matching a pattern such as sample_*.pt.
def load_folder(folder: str, pattern: str) -> List[dict]:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern} in {folder}")
    return [load_sample(p) for p in paths]
