from typing import Any
import torch

from qewton.backends.linalg import LinAlgBackend


class TorchLinAlgBackend(LinAlgBackend[torch.Tensor]):

    @staticmethod
    def svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.svd(x)

    @staticmethod
    def norm(x: torch.Tensor, ord="fro", axis=None, keepdims=False) -> torch.Tensor:
        return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)
