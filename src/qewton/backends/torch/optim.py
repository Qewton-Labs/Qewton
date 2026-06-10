import torch

from qewton.backends.optim import OptimBackend


class TorchOptimBackend(OptimBackend[torch.Tensor]):
    """Torch implementations of optimization algorithms."""
