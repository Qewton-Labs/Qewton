from typing import Any

from ....config.axes import EllipsisAxes, EllipsisDim
from ....config import DataConfiguration

from ..base import DataSet


class ArrayLikeDataSet(DataSet):
    """A data container that handles "array-like" or dense "data". This
    includes for example data such as np.array, torch.tensor, or hdf5 data.
    The concrete data containers are implemented via subclasses.
    """

    def __init__(
        self, data: Any, data_configs: DataConfiguration | list[DataConfiguration]
    ):
        """
        Args:
            data (_type_): Data objects that allow for slicing via [:] and
                have a shape property (.shape) returning a tuple of integers.
        """
        self._data = data if isinstance(data_configs, (list, tuple)) else [data]
        self._data_configs = (
            data_configs if isinstance(data_configs, (list, tuple)) else [data_configs]
        )
        self.update_configs()

    def update_configs(self):
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
        return self._data[0].shape[0]

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return [data[idx] for data in self._data]

    @property
    def data_configs(self) -> list[DataConfiguration]:
        return self._data_configs

    def get_batch(self, idx_list) -> list[Any]:
        return [data[idx_list] for data in self._data]

    def get_continuous_batch(self, start_idx, end_idx) -> list[Any]:
        return [data[start_idx:end_idx] for data in self._data]

    def load_complete_data(self, variable=None, data_item=None):
        if variable is None and data_item is None:
            return self._data
        if isinstance(data_item, int):
            return self._data[data_item]
        for i, config in enumerate(self._data_configs):
            if variable in config.variables:
                v_slice = config.get_variable_slice(variable)
                return self._data[i][v_slice]
        raise ValueError(f"No such variable {variable} in the data.")

    def __getattr__(self, name):
        # Only used if attribute not found on self
        return getattr(self._data, name)
