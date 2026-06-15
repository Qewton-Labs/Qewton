from qewton.backends.base import Backend, TensorType
from qewton.config.devices import Device, cpu


class RandomBackend(Backend[TensorType]):
    """A Backend that implements random number generation."""

    @staticmethod
    def seed(seed: int):
        raise NotImplementedError

    @staticmethod
    def normal(shape, mean=0.0, std=1.0, device: Device = cpu) -> TensorType:
        raise NotImplementedError

    @staticmethod
    def uniform(shape, low=0.0, high=1.0, device: Device = cpu) -> TensorType:
        raise NotImplementedError
