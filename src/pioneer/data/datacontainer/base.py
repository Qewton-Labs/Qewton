from abc import ABC, abstractmethod


class DataContainer(ABC):

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    def dtype(self):
        return None

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def load_complete_data(self):
        """Fully load the data into memory.
        Many subclasses provide *lazy* slicing, where data is only read from the
        disk when needed.

        This returns the whole dataset at once.
        """
        return self[:]

    @property
    def metadata(self) -> dict:
        return {}

    def to(self, device):
        """Move tensor to device (cpu/gpu/tpu)."""
