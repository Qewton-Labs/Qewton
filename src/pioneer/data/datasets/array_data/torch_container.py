from .base import ArrayLikeDataContainer


class TorchDataContainer(ArrayLikeDataContainer):
    def __init__(self, tensor):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "TorchDenseLoader requires 'torch'. Install via pip install torch"
            ) from e

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected torch.Tensor")

        super().__init__(tensor)
        self._data = self._data

    def to(self, device):
        """Move tensor to device (cpu/cuda)."""
        self._data = self._data.to(device)
