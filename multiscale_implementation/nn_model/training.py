from typing import List
import random
import time
import numpy as np
import torch

from ..data_loading.config import Config
from ..data_loading.normalization import Scales, compute_scales, node_features
from ..data_loading.loading import load_folder
from ..mesh.meshbank import MeshBank
from ..mesh.mesh_hierarchy import describe_hierarchy, plot_hierarchy
from .model import MultiscaleGNN, predict, relative_l2


def make_batches(samples: List[dict], batch_size: int, shuffle: bool):
    groups = {}
    for i, s in enumerate(samples):
        # groups.setdefault(s["mesh_key"], []).append(i)
        key = s.get("mesh_key", None)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    batches = []
    for idx in groups.values():
        if (
            shuffle
        ):  # the reason for this is that we want to shuffle the samples within each group, but not across groups
            random.shuffle(idx)
        batches += [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def collate(samples: List[dict], idx: List[int], device):
    def stack_fn(k):
        return torch.stack([samples[i][k] for i in idx]).to(device)

    return stack_fn("node_ft"), stack_fn("g"), stack_fn("bmask"), stack_fn("u")


@torch.no_grad()  # with torch.no_grad(), we don't compute gradients, which saves memory and computation during evaluation
def evaluate(
    model, samples, bank: MeshBank, scales: Scales, batch_size: int, device
) -> np.ndarray:
    model.eval()
    errs = []
    for idx in make_batches(samples, batch_size, shuffle=False):
        node_ft, g, bm, u = collate(samples, idx, device)
        pred = predict(model, node_ft, g, bm, bank.get(samples[idx[0]]), scales)
        errs.append(relative_l2(pred, u, (~bm).float()).cpu())
    return torch.cat(errs).numpy()


def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(
        "cuda"
        if cfg.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.device == "auto" else cfg.device)
    )

    samples = load_folder(cfg.data_dir, cfg.pattern)
    random.shuffle(samples)
    n_val = max(
        1, int(cfg.val_fraction * len(samples))
    )  # number of validation samples, at least 1
    val, tr = samples[:n_val], samples[n_val:]  # validation and training samples
    scales = compute_scales(tr)
    for s in samples:
        s["node_ft"] = node_features(s, scales, bool(cfg.use_pos))
    tests = {}
    for d in filter(None, cfg.test_dirs.split(",")):
        tests[d] = load_folder(d.strip(), cfg.pattern)
        for s in tests[d]:
            s["node_ft"] = node_features(s, scales, bool(cfg.use_pos))

    bank = MeshBank(cfg.n_levels, device)
    lv = bank.get(tr[0])
    print(
        f"{len(tr)} train / {len(val)} val samples, "
        f"{len({s['mesh_key'] for s in samples})} distinct mesh(es)"
    )
    print("hierarchy:", describe_hierarchy(lv))
    print(f"scales: f={scales.f:.3g} g={scales.g:.3g} u={scales.u:.3g}")
    if cfg.plot_hierarchy:
        plot_hierarchy([l.to("cpu") for l in lv], cfg.plot_hierarchy)

    model = MultiscaleGNN(
        in_dim_node=tr[0]["node_ft"].shape[-1],
        in_dim_edge=cfg.edge_dim,
        hidden=cfg.hidden,
        n_levels=cfg.n_levels,
        mp_per_level=cfg.mp_per_level,
        out_dim=cfg.out_dim,
    ).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps = cfg.epochs * len(make_batches(tr, cfg.batch_size, False))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.05
    )

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for idx in make_batches(tr, cfg.batch_size, shuffle=True):
            x, g, bm, u = collate(tr, idx, device)
            pred = predict(model, x, g, bm, bank.get(tr[idx[0]]), scales)
            loss = relative_l2(pred, u, (~bm).float()).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if epoch == 1 or epoch % max(1, cfg.epochs // 20) == 0 or epoch == cfg.epochs:
            v = evaluate(model, val, bank, scales, cfg.batch_size, device)
            print(
                f"epoch {epoch:4d}  train relL2 {np.mean(losses):.4f}  "
                f"val relL2 {v.mean():.4f}  ({time.time() - t0:.0f}s)"
            )

    for name, ts in tests.items():
        e = evaluate(model, ts, bank, scales, cfg.batch_size, device)
        print(f"test '{name}': relL2 mean {e.mean():.4f}, max {e.max():.4f}")

    torch.save(
        dict(model=model.state_dict(), config=vars(cfg), scales=vars(scales)), cfg.save
    )
    print("saved", cfg.save)
    return model, scales, bank
