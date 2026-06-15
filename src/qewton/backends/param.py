from abc import abstractmethod

from qewton.backends.base import Backend, TensorType


class ParameterBackend(Backend[TensorType]):
    @staticmethod
    @abstractmethod
    def initialize(shape=None, tensor=None) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def to(data, device) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def requires_grad(data, requires_grad: bool) -> TensorType:
        pass
