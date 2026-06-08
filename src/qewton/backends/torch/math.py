import torch
from qewton.backendss.base import MathBackend


class TorchMathBackend(MathBackend[torch.Tensor]):
    add = torch.add
