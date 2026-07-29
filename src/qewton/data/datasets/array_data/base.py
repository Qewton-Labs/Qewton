"""
Base implementation for array-like datasets.
Handles data that supports slicing and has a .shape property.
"""

from typing import Any

from qewton.backends.base import Backend
from qewton.config.axes import EllipsisAxes, EllipsisDim
from qewton.config import DataConfiguration

from qewton.data.datasets.base import DataSet


class ArrayLikeDataSet(DataSet):
    """A data container that handles "array-like" or dense "data". This
    includes for example data such as np.array, torch.tensor, or hdf5 data.
    The concrete data containers are implemented via subclasses.

    Args:
        data (Any): Data objects that allow for slicing via [:] and
            have a shape property (.shape) returning a tuple of integers.
        data_configs (DataConfiguration | list[DataConfiguration]): Configuration
            mapping the data dimensions to semantic axes.
    """

    def __init__(
        self, data: Any, data_configs: DataConfiguration | list[DataConfiguration]
    ):
        # Normalize data and configs to lists to support multi-input/output datasets
        self._data = data if isinstance(data_configs, (list, tuple)) else [data]
        self._data_configs = (
            data_configs if isinstance(data_configs, (list, tuple)) else [data_configs]
        )
        # check that variable names are unique across all configs
        all_variables = set()
        for config in self._data_configs:
            for var in config.variables:
                if var in all_variables:
                    raise ValueError(
                        f"Variable name {var} appears in multiple data configurations.\
                            Variable names must be unique across all data configs."
                    )
                all_variables.add(var)

        self.update_configs()

    def update_configs(self):
        """Validates and updates the size of dimensions in the data configurations
        based on the actual shape of the provided data objects.
        """
        assert len(self._data) == len(
            self._data_configs
        ), "A separate DataConfig is required for each data object."
        for data, config in zip(self._data, self._data_configs):
            counter = 0
            for a in config.axes:
                if isinstance(a, EllipsisAxes):
                    raise ValueError("Ellipsis not supported at this stage.")
                for d in a.shape:
                    if isinstance(d, EllipsisDim):
                        raise ValueError("Ellipsis not supported at this stage.")
                    if counter >= len(data.shape):
                        raise ValueError(
                            "Too many axes in data configuration for the given data."
                        )
                    d.update_size(data.shape[counter])
                    counter += 1
            if counter < len(data.shape):
                raise ValueError("Too few axes in data configuration for the given data.")

    def __len__(self) -> int:
        if not self._data:
            return 0
        return int(self._data[0].shape[0])

    def __getitem__(self, idx):
        return [data[idx] for data in self._data]

    @property
    def data_configs(self) -> list[DataConfiguration]:
        return self._data_configs

    def get_batch(self, idcs) -> list[Any]:
        return [data[idcs] for data in self._data]

    def get_continuous_batch(self, start_idx, end_idx) -> list[Any]:
        return [data[start_idx:end_idx] for data in self._data]

    def load_complete_data(self, variable=None, data_item=None):
        """Fully loads data into memory.

        Args:
            variable (str, optional): If provided, returns only the data for
                this specific variable.
            data_item (int, optional): If provided, returns the i-th data object.

        Returns:
            Any: The requested data.
        """
        if variable is None and data_item is None:
            return self._data
        if isinstance(data_item, int):
            return self._data[data_item]
        for i, config in enumerate(self._data_configs):
            if variable in config.variables:
                v_slice = config.get_variable_slice(variable)
                return self._data[i][v_slice]
        raise ValueError(f"No such variable {variable} in the data.")


class BackendDataSet(ArrayLikeDataSet):
    """A dataset implementation for backend-specific tensor objects, e.g. numpy.ndarray
    or torch.Tensor.
    """

    def __init__(
        self,
        data: Any,
        data_configs: DataConfiguration | list[DataConfiguration],
        backend: type[Backend],
    ):
        """Initialize the BackendDataSet.

        Args:
            data (Any): Tensors or data objects compatible with the backend.
            data_configs (DataConfiguration | list[DataConfiguration]): Configuration
                defining the dimensions and semantics of the data.
            backend (type[Backend]): The backend associated with the data.

        Raises:
            TypeError: If any data item is not compatible with the backend's default type.
        """
        self.backend = backend
        items = data if isinstance(data_configs, (list, tuple)) else [data]
        for item in items:
            if not isinstance(item, self.backend.default_dtype):
                raise TypeError(f"{self.backend.__name__} only handles \
                        {self.backend.default_dtype.__name__}, not {type(item)}.")

        super().__init__(data, data_configs)

    @classmethod
    def from_file(cls, path, data_configs, backend, **kwargs):
        """Load a Backend dataset from a file using the provided backend load.

        Args:
            path (str): Path to the saved tensors.
            data_configs: The configuration for the data being loaded.
            **kwargs: Additional arguments passed e.g. to torch.load.

        Returns:
            DataSet: Initialized dataset instance.
        """
        data = backend.load(path, **kwargs)
        return cls(data, data_configs, backend)

    def to(self, device):
        """Move tensor to device (cpu/cuda)."""
        if hasattr(self.backend, "to"):
            self._data = [self.backend.to(t, device) for t in self._data]
        else:
            raise NotImplementedError(f"{self.backend} does not provide device changes.")
        return self
