import torch
from qewton.backends.nn import NNBackend


class TorchNNBackend(NNBackend[torch.Tensor]):
    """Torch implementations of neural network operations."""

    relu = torch.nn.functional.relu
    sigmoid = torch.nn.functional.sigmoid
    tanh = torch.nn.functional.tanh
