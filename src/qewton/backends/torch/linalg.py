from typing import Any
import torch

from qewton.backends.linalg import LinAlgBackend


class TorchLinAlgBackend(LinAlgBackend[torch.Tensor]):

    @staticmethod
    def svd(x: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.svd(x)
