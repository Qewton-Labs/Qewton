"""DataSet implementation for NumPy arrays."""

import numpy as np

from .base import ArrayLikeDataSet


class NumpyDataSet(ArrayLikeDataSet):
    """Data container specialized for NumPy ndarrays."""

    def __init__(self, data, data_configs):
        """
        Initializes the NumpyDataSet.

        Args:
            data (np.ndarray | list[np.ndarray]): The raw numpy data.
            data_configs (DataConfiguration | list[DataConfiguration]): Axis metadata.

        Raises:
            TypeError: If the data provided is not a numpy ndarray.
        """
        items = data if isinstance(data_configs, (list, tuple)) else [data]
        for item in items:
            if not isinstance(item, np.ndarray):
                raise TypeError(
                    f"NumpyDataContainer only handles numpy.ndarray, not {type(item)}."
                )

        super().__init__(data, data_configs)

    @classmethod
    def from_file(cls, path, data_configs, mmap_mode=None):
        """Load a NumPy dataset from a .npy or .npz file.

        Args:
            path (str): Path to the file.
            data_configs: The configuration for the data being loaded.
            mmap_mode (str, optional): Memory-mapping mode for large files.

        Returns:
            NumpyDataSet: Initialized dataset instance.
        """
        array = np.load(path, mmap_mode=mmap_mode)
        return cls(array, data_configs)
