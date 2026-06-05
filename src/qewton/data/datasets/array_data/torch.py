"""DataSet implementation for PyTorch tensors."""

from qewton.data.datasets.array_data.base import ArrayLikeDataSet


class TorchDataSet(ArrayLikeDataSet):
    """Data container specialized for PyTorch Tensors.

    Args:
        data (torch.Tensor | list[torch.Tensor]): The raw torch data.
        data_configs (DataConfiguration | list[DataConfiguration]): Axis metadata.

    Raises:
        ImportError: If torch is not installed.
        TypeError: If the data provided is not a torch.Tensor.
    """

    def __init__(self, data, data_configs):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "TorchDataSet requires 'torch'. Install via pip install torch"
            ) from e

        items = data if isinstance(data_configs, (list, tuple)) else [data]
        for item in items:
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    f"TorchDataSet only handles torch.Tensor, not {type(item)}."
                )

        super().__init__(data, data_configs)

    @classmethod
    def from_file(cls, path, data_configs, **kwargs):
        """Load a PyTorch dataset from a file using torch.load.

        Args:
            path (str): Path to the saved tensors.
            data_configs: The configuration for the data being loaded.
            **kwargs: Additional arguments passed to torch.load.

        Returns:
            TorchDataSet: Initialized dataset instance.
        """
        try:
            import torch
        except ImportError as e:
            raise ImportError("TorchDataSet requires 'torch'.") from e
        data = torch.load(path, **kwargs)
        return cls(data, data_configs)

    def to(self, device):
        """Move tensor to device (cpu/cuda)."""
        self._data = [t.to(device) for t in self._data]
        return self
