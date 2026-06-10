import torch
from qewton.backends.nn import NNBackend


class TorchNNBackend(NNBackend[torch.Tensor]):
    """Torch implementations of neural network operations."""
