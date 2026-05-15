from .base import ArrayLikeDataSet


class HDF5DataContainer(ArrayLikeDataSet):
    """Data container for HDF5 files."""

    def __init__(self, dataset, file_handle):
        """
        Args:
            dataset (h5py.Dataset): The data from the HDF5 file that should be
                contained in this class.
            file_handle (h5py.File): The file where the data originates from.

        Notes:
            Reading a file via h5py can be done via
                file_handle = h5py.File(file_path, "r")
                dataset = f[dataset_key]
        """
        self._file = file_handle  # keep handle alive
        super().__init__(dataset)

    @classmethod
    def from_file(cls, file_path: str, dataset_key: str):
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "HDF5DenseLoader requires 'h5py'. Install via pip install h5py"
            ) from e

        f = h5py.File(file_path, "r")
        if dataset_key not in f:
            raise KeyError(f"Dataset '{dataset_key}' not found in file.")

        dataset = f[dataset_key]
        return cls(dataset, f)

    def close(self):
        """Close the HDF5 file if needed."""
        if self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except (AttributeError, RuntimeError, OSError):
            pass

    @property
    def metadata(self) -> dict:
        return dict(self._data.attrs) if hasattr(self._data, "attrs") else {}
