from typing import Any

from qewton.backends.base import Backend
from qewton.config import DataConfiguration
from qewton.config.dtypes import Float32
from qewton.data.datasets.array_data.base import ArrayLikeDataSet


def _import_h5py():
    try:
        import h5py  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "HDF5DataSet requires 'h5py'. Install via pip install h5py"
        ) from e
    return h5py


class HDF5DataSet(ArrayLikeDataSet):
    """A dataset read from an HDF5 file into memory.

    Each key names a dataset inside the file, which becomes one data item,
    e.g. inputs and targets. The data is read completely into tensors of the
    given backend, so it behaves like any other ArrayLikeDataSet.

    Only the path and the keys are kept, never an open file. The dataset can
    therefore be saved, pickled and copied, e.g. by the Tuner. After loading
    a saved dataset, setup() reads the file again.

    Args:
        path (str): The HDF5 file.
        keys (str | list[str]): The datasets inside the file to read, one per
            data item. Datasets inside groups are given as "group/dataset".
        data_configs (DataConfiguration | list[DataConfiguration]): One
            configuration per key, mapping its dimensions to semantic axes.
        backend (type[Backend]): The backend whose tensors hold the data.
        dtype (optional): The data type the data is cast to. None keeps the
            type stored in the file. Defaults to Float32.
        load_on_setup (bool, optional): If True, the file is only read when
            the dataset is set up. Defaults to False.
        save_data (bool, optional): If True, the data will be saved when the
            dataset is serialized. Defaults to False.

    Raises:
        ImportError: If h5py is not installed.
        KeyError: If a key is not found in the file.
        TypeError: If a key names a group instead of a dataset.
    """

    def __init__(
        self,
        path: str,
        keys: str | list[str],
        data_configs: DataConfiguration | list[DataConfiguration],
        backend: type[Backend],
        dtype: Any = Float32,
        load_on_setup: bool = False,
        save_data: bool = False,
    ):
        _import_h5py()
        self.keys = list(keys) if isinstance(keys, (list, tuple)) else [keys]
        self.dtype = dtype
        if not isinstance(data_configs, (list, tuple)):
            data_configs = [data_configs]
        super().__init__(
            [None] * len(self.keys),
            data_configs,
            backend,
            data_paths=[str(path)] * len(self.keys),
            save_data=save_data,
        )
        if not load_on_setup:
            self.setup()

    @classmethod
    def from_file(  # pylint: disable=arguments-differ
        cls,
        path: str,
        keys: str | list[str],
        data_configs: DataConfiguration | list[DataConfiguration],
        backend: type[Backend],
        dtype: Any = Float32,
        load_on_setup: bool = False,
    ):
        """Same as the constructor, for symmetry with
        ArrayLikeDataSet.from_file."""
        return cls(
            path, keys, data_configs, backend, dtype=dtype, load_on_setup=load_on_setup
        )

    def _read_files(self) -> list[Any]:
        h5py = _import_h5py()
        data = []
        for path, key in zip(self.data_paths, self.keys):  # type: ignore
            with h5py.File(path, "r") as file:
                entry = self._get_entry(file, path, key, h5py)
                data.append(self.backend.build_tensor(entry[()], dtype=self.dtype))
        return data

    @staticmethod
    def _get_entry(file, path, key, h5py):
        if key not in file:
            raise KeyError(f"Dataset '{key}' not found in file '{path}'.")
        entry = file[key]
        if not isinstance(entry, h5py.Dataset):
            raise TypeError(f"'{key}' in file '{path}' is a group, not a dataset.")
        return entry

    @property
    def metadata(self) -> dict:
        """The HDF5 attributes of each key, read from the file."""
        h5py = _import_h5py()
        metadata = {}
        for path, key in zip(self.data_paths, self.keys):  # type: ignore
            with h5py.File(path, "r") as file:
                metadata[key] = dict(self._get_entry(file, path, key, h5py).attrs)
        return metadata
