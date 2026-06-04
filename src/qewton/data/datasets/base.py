"""Base class definitions for all datasets."""

from typing import Any
from abc import ABC, abstractmethod


from qewton.config.data_configurations import DataConfiguration


class DataSet(ABC):
    """Abstract base class for all datasets.

    A DataSet provides a standard interface for accessing data samples,
    batches, and configuration metadata. It supports both eager and lazy loading.
    """

    @property
    @abstractmethod
    def data_configs(self) -> list[DataConfiguration]:
        """Returns the data configurations for the data stored in this dataset."""
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, idx):
        """Access data at a specific index."""
        pass

    @abstractmethod
    def get_batch(self, idcs) -> list[Any]:
        """Retrieve a batch of data given a list of indices.

        Args:
            idcs (list[int]): Indices of the samples to retrieve.

        Returns:
            list[Any]: A list where each element corresponds to a data object
                (e.g., input tensors, targets).
        """
        pass

    @abstractmethod
    def get_continuous_batch(self, start_idx, end_idx) -> list[Any]:
        """Retrieve a continuous slice of data.

        Args:
            start_idx (int): Starting index.
            end_idx (int): Ending index.

        Returns:
            list[Any]: Sliced data objects.
        """
        pass

    @abstractmethod
    def load_complete_data(self, variable=None, data_item=None):
        """Fully load the data into memory.
        Many subclasses provide *lazy* slicing, where data is only read from the
        disk when needed.

        This returns the whole dataset at once.
        """
        return self[:]

    @property
    def metadata(self) -> dict:
        """Optional metadata associated with the dataset (e.g., HDF5 attributes)."""
        return {}
