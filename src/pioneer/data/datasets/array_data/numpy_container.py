import numpy as np

from .base import ArrayLikeDataSet


class NumpyDataContainer(ArrayLikeDataSet):

    def __init__(self, array: np.ndarray):
        assert isinstance(
            array, np.ndarray
        ), f"NumpyDataContainer only handles numpy data, not {type(array)}."
        super().__init__(array)

    @classmethod
    def from_file(cls, path, mmap_mode=None):
        array = np.load(path, mmap_mode=mmap_mode)
        return cls(array)
