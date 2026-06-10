from abc import abstractmethod
from typing import Any
from qewton.backends.base import Backend, TensorType


class GradBackend(Backend[TensorType]):
    """A Backend that implements all differential operators."""

    @staticmethod
    @abstractmethod
    def gradient_tracking(inp: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def gradient(u: Any, x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def laplacian(u: Any, x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def normal_derivative(u: Any, normals: Any, x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def divergence(u: Any, x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def jacobian(u: Any, x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def partial(u: Any, x: Any) -> TensorType:
        pass

    @staticmethod
    @abstractmethod
    def hessian(u: Any, x: Any) -> TensorType:
        pass
