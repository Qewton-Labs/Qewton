from typing import Any
from abc import ABC, abstractmethod


from ...config.data_configurations import DataConfiguration


class DataSet(ABC):
    @property
    @abstractmethod
    def data_configs(self) -> list[DataConfiguration]:
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_batch(self, idcs) -> list[Any]:
        pass

    @abstractmethod
    def get_continuous_batch(self, start_idx, end_idx) -> list[Any]:
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
        return {}
