from dataclasses import dataclass


@dataclass
class Config:  # Config is a dataclass that holds every setting
    data_dir: str = "."  # folder with your sample_XXXX.pt files
    pattern: str = "sample_*.pt"
    test_dirs: str = (
        ""  # optional extra test folders, comma separated (e.g. "test_g0,test_f0")
    )
    val_fraction: float = 0.1
    edge_dim: int = 3  # edge features: (dx, dy, |d|) / mean edge length of this level
    hidden: int = 64  # latent width; 128 is more accurate but ~3x slower
    n_levels: int = 4  # number of graph levels incl. the original mesh
    mp_per_level: int = 2  # message-passing blocks per level (down and up)
    out_dim: int = 1  # number of output features per node (1 for Poisson)
    use_pos: int = 1  # 1: give (x, y) as node input; 0: geometry only via edges
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "auto"
    save: str = "poisson_msgnn.pt"
    plot_hierarchy: str = (
        ""  # e.g. "hierarchy.png" to save a picture of the graph levels
    )
