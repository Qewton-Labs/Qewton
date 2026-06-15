from abc import abstractmethod
from typing import Any

from qewton.backends.base import Backend, TensorType


class NNBackend(Backend[TensorType]):
    """A Backend that implements all neural network related operations, such as
    activations, layers, etc.
    """

    @staticmethod
    @abstractmethod
    def relu(x: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sigmoid(x: Any, /) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tanh(x: Any, /) -> TensorType:
        pass
