import torch
from .config import Config
from .normalization import Scales, node_features
from ..nn_model.model import MultiscaleGNN
from ..mesh.mesh_hierarchy import build_hierarchy
from .loading import load_sample
from ..nn_model.training import predict


def load_trained(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg, scales = Config(**ck["config"]), Scales(**ck["scales"])
    in_dim = 5 if cfg.use_pos else 3
    model = MultiscaleGNN(in_dim, cfg.hidden, cfg.n_levels, cfg.mp_per_level).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, scales


@torch.no_grad()
def solve(model, cfg: Config, scales: Scales, sample_path: str, device="cpu"):
    """Predict u for one sample file (any mesh; the hierarchy is built on the fly)."""
    s = load_sample(sample_path)
    levels = [
        l.to(device)
        for l in build_hierarchy(s["coords"], s["edge_index"], cfg.n_levels)
    ]
    x = node_features(s, scales, bool(cfg.use_pos))[None].to(device)
    return predict(
        model, x, s["g"][None].to(device), s["bmask"][None].to(device), levels, scales
    )[0].cpu()
