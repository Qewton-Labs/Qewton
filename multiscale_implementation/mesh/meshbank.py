from typing import List
from .mesh_hierarchy import Level, build_hierarchy


class MeshBank:
    """Caches one hierarchy per mesh (keyed by mesh fingerprint)."""

    def __init__(self, n_levels: int, device):
        self.n_levels, self.device, self.cache = n_levels, device, {}

    def get(self, s: dict) -> List[Level]:
        k = s["mesh_key"]
        if k not in self.cache:
            lv = build_hierarchy(s["coords"], s["edge_index"], self.n_levels)
            self.cache[k] = [l.to(self.device) for l in lv]
        return self.cache[k]
