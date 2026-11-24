from abc import abstractmethod
from typing import Any
from qewton.backends.base import Backend, TensorType


class FunctionSpaceBackend(Backend[TensorType]):
    """A Backend that implements all operations related to function spaces."""

    # region FunctionSpaces
    @staticmethod
    @abstractmethod
    def create_function_space(mesh: Any, function_type: Any, order: int) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def create_test_function(function_space: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def create_trial_function(function_space: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def create_function(function_space: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def create_constant(value: Any, mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def set_constant_value(constant: Any, value: Any) -> None:
        pass

    # endregion
    # region Function manipulation/evaluation

    @staticmethod
    @abstractmethod
    def set_function_dofs(function: Any, values: Any) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_function_value(function: Any, x: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_function_dofs(function: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_function_on_mesh_vertices(function: Any, mesh: Any) -> Any:
        pass

    # endregion
