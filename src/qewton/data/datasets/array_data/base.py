"""
Base implementation for array-like datasets.
Handles backend tensors, e.g. np.ndarray or torch.Tensor, that support slicing
and have a .shape property.
"""

from typing import Any
import warnings

from qewton.backends import resolve_backend
from qewton.backends.base import ComputingBackend
from qewton.config.axes import EllipsisAxes, EllipsisDim
from qewton.config import DataConfiguration

from qewton.config.saving.loading import Deserializer
from qewton.data.datasets.base import DataSet


class ArrayLikeDataSet(DataSet):
    """A dataset of dense, array-like data, e.g. np.ndarray or torch.Tensor.

    Holds one or more data items, e.g. inputs and targets, each described by
    its own DataConfiguration. All items belong to one backend, which is
    either given or inferred from the data. The data can also be loaded from
    files via from_file, optionally only once the dataset is set up.

    Args:
        data (Any | list[Any]): The data item(s). Several items are passed as
            a list, together with a list of data_configs. An item may be None
            if it is loaded from data_paths in setup().
        data_configs (DataConfiguration | list[DataConfiguration]): One
            configuration per data item, mapping its dimensions to semantic
            axes.
        backend (type[Backend], optional): The backend of the data. If None,
            it is inferred from the first data item. Defaults to None.
        data_paths (list[str], optional): One file per data item, from which
            setup() loads the data if it is not loaded yet. Set by from_file.
            Defaults to None.
        save_data (bool, optional): If True, the data will be saved when the
            dataset is serialized. Defaults to False.

    Raises:
        TypeError: If a data item is not a tensor of the backend.
        ValueError: If no backend is given and none can be inferred from the
            data, or if a variable name appears in several data configs.
    """

    def __init__(
        self,
        data: Any,
        data_configs: DataConfiguration | list[DataConfiguration],
        backend: type[ComputingBackend] | None = None,
        data_paths: list[str] | None = None,
        save_data: bool = False,
    ):
        # Normalize data and configs to lists to support multi-input/output datasets
        if isinstance(data_configs, (list, tuple)):
            self._data = list(data) if isinstance(data, (list, tuple)) else data
            self._data_configs = list(data_configs)
        else:
            self._data = [data]
            self._data_configs = [data_configs]
        # check that variable names are unique across all configs
        all_variables = set()
        for config in self._data_configs:
            for var in config.variables:
                if var in all_variables and not var.is_empty:
                    raise ValueError(
                        f"Variable name {var} appears in multiple data configurations. "
                        "Variable names must be unique across all data configs."
                    )
                all_variables.add(var)

        self.backend = backend if backend is not None else self._infer_backend()
        self._check_data_types()
        self.data_paths = data_paths
        self.save_data = save_data
        self.update_configs()

    @classmethod
    def from_file(
        cls,
        path: str | list[str],
        data_configs: DataConfiguration | list[DataConfiguration],
        backend: type[ComputingBackend],
        load_on_setup: bool = False,
    ):
        """Creates a dataset from files, read with backend.load_data.

        Args:
            path (str | list[str]): The file(s) to load, one per data item.
            data_configs (DataConfiguration | list[DataConfiguration]): One
                configuration per file, mapping its dimensions to semantic
                axes.
            backend (type[Backend]): The backend that reads the files and
                handles the data.
            load_on_setup (bool, optional): If True, the files are only read
                when the dataset is set up. Defaults to False.

        Returns:
            ArrayLikeDataSet: The dataset, keeping the paths so the data can
                be reloaded after deserialization.
        """
        paths = list(path) if isinstance(path, (list, tuple)) else [path]
        if not isinstance(data_configs, (list, tuple)):
            data_configs = [data_configs]
        dataset = cls([None] * len(paths), data_configs, backend, data_paths=paths)
        if not load_on_setup:
            dataset.setup()
        return dataset

    def _infer_backend(self) -> type[ComputingBackend]:
        for item in self._data:
            if item is not None:
                return resolve_backend(item)
        raise ValueError(
            "Can not infer the backend of a dataset without data. "
            "Pass the backend explicitly."
        )

    def _check_data_types(self):
        """Raises a TypeError if a data item is not a tensor of the backend."""
        for item in self._data:
            if item is not None and not isinstance(item, self.backend.default_dtype):
                raise TypeError(
                    f"{self.backend.__name__} only handles "
                    f"{self.backend.default_dtype.__name__}, not {type(item).__name__}."
                )

    def update_configs(self):
        """Validates and updates the size of dimensions in the data configurations
        based on the actual shape of the provided data objects.
        """
        assert len(self._data) == len(
            self._data_configs
        ), "A separate DataConfig is required for each data object."
        if all(data is None for data in self._data):
            return  # No data to validate against
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

    def setup(self):
        """Loads the data from data_paths, if it is not loaded yet."""
        if self.data_paths is None or self.data_set_loaded:
            return
        self._data = self._read_files()
        self._check_data_types()
        self.update_configs()

    def _read_files(self) -> list[Any]:
        """Reads all data items from data_paths. Subclasses for other file
        formats override this."""
        return [self.backend.load_data(p) for p in self.data_paths]  # type: ignore

    def to(self, device):
        """Moves the data to the given device, e.g. "cpu" or "cuda"."""
        self._data = [
            None if item is None else self.backend.to(item, device) for item in self._data
        ]
        return self

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

    @property
    def data_set_loaded(self) -> bool:
        """Check if the dataset is fully loaded into memory.

        Returns:
            bool: True if all data objects are loaded, False otherwise.
        """
        return all(data is not None for data in self._data)

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

    def _collect_serializable_attributes(self) -> tuple[list, list]:
        self_keys, self_args = super()._collect_serializable_attributes()
        if not self.save_data:
            # Save placeholders, so the loaded dataset knows how many data
            # items are missing. (The configs can not be used for this in
            # load(), since lists are only filled after load() is called.)
            data_idx = self_keys.index("_data")
            self_args[data_idx] = [None] * len(self._data)
        return self_keys, self_args

    def load(self, serializer: Deserializer, data_config: dict) -> None:
        super().load(serializer, data_config)
        if not self.save_data and self.data_paths is None:
            warnings.warn(
                "Data was not saved with this dataset. "
                "The data will be None and needs to be set manually."
            )
