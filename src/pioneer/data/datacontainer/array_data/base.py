from ..base import DataContainer


class ArrayLikeDataContainer(DataContainer):
    """A data container that handles "array-like" or dense "data". This
    includes for example data such as np.array, torch.tensor, or hdf5 data.
    The concrete data containers are implemented via subclasses.
    """

    def __init__(self, data):
        """
        Args:
            data (_type_): Data objects that allow for slicing via [:] and
                have a shape property (.shape) returning a tuple of integers.
        """
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return getattr(self._data, "dtype", None)

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]

    def __getattr__(self, name):
        # Only used if attribute not found on self
        return getattr(self._data, name)

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) + dir(self._data)))
