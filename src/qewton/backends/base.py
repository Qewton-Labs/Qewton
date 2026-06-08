from __future__ import annotations
from abc import abstractmethod
from typing import Annotated, Any, Generic, TypeVar

from qewton.config.dtypes import Float32

TensorType = TypeVar("TensorType")


# TODO: Wa have standard datatype gives tensors, but what about float, etc.?


class Backend(Generic[TensorType]):
    pass


class DeepLearningBackend(Backend[TensorType]):
    math: type[MathBackend]
    grad: type[GradBackend]
    optim: type[OptimBackend]

    @classmethod
    def to(cls, data, device):
        """Moves the data to the given device.

        Args:
            data (TensorType): The data to move.
            device (str): The device to move the data to.

        Returns:
            TensorType: The moved data.
        """
        raise NotImplementedError(
            "The moving to a different device is backend dependent."
        )

    @classmethod
    def from_numpy(cls, data):
        """Converts a numpy array to the standard datatype of this backend.

        Args:
            data (np.ndarray): The numpy array to convert.

        Returns:
            TensorType: The converted data.
        """
        raise NotImplementedError(
            "The from_numpy method must be implemented by subclasses of Backend."
        )


class GradBackend(Backend[TensorType]):
    @abstractmethod
    def grad(self, data, *args, **kwargs):
        return


class OptimBackend(Backend[TensorType]):
    pass


class LinalgBackend(Backend[TensorType]):
    pass
