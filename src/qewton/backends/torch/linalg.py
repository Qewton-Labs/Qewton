import torch

from qewton.backends.linalg import LinAlgBackend


class TorchLinAlgBackend(LinAlgBackend[torch.Tensor]):

    @staticmethod
    def svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.svd(x)

    @staticmethod
    def norm(x: torch.Tensor, order="fro", axis=None, keepdims=False) -> torch.Tensor:
        return torch.norm(x, p=order, dim=axis, keepdim=keepdims)

    @staticmethod
    def det(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(x)

    @staticmethod
    def inv(x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(x)
