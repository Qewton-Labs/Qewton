from abc import abstractmethod
from typing import Any
from qewton.backends.base import Backend, TensorType


class FormBackend(Backend[TensorType]):
    """A Backend that implements all operations to build a weak formulation."""

    # region Math operations
    @staticmethod
    @abstractmethod
    def add(a: Any, b: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def multiply(a: Any, b: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def subtract(a: Any, b: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def divide(a: Any, b: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def power(a: Any, b: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def negate(a: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def abs(a: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def sqrt(a: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def dot(a: Any, b: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def transpose(a: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def inner(a: Any, b: Any) -> Any:
        pass

    # endregion

    # region Assembly
    @staticmethod
    @abstractmethod
    def assemble(form: Any) -> Any:
        pass

    # endregion

    # region Derivatives
    # Derivatives dont have a dependence on x here usually
    @staticmethod
    @abstractmethod
    def gradient(u: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def rotation(u: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def divergence(u: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def derivative(u: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def time_derivative(u: Any, order: int) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def hessian(u: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def symmetric_gradient(u: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def jacobian(u: Any) -> Any:
        pass

    # endregion
    # region Integrals
    @staticmethod
    @abstractmethod
    def integrate(form: Any, measure: Any) -> Any:
        pass

    # endregion

    # region Boundary conditions
    @staticmethod
    @abstractmethod
    def build_dirichlet(boundary_value: Any, boundary: Any, function_space: Any) -> Any:
        pass

    # endregion
