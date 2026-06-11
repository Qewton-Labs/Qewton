from abc import abstractmethod
from typing import Any

from qewton.backends.base import Backend, TensorType


class NNBackend(Backend[TensorType]):
    """A Backend that implements all neural network related operations, such as
    activations, layers, etc.
    """

    @staticmethod
    @abstractmethod
    def relu(input: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def sigmoid(input: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def tanh(input: Any) -> TensorType:
        pass
