from typing import Any

from dolfinx.fem import Function, functionspace
from ufl import TestFunction, TrialFunction

from qewton.backends.fem.function_space import FunctionSpaceBackend


class FenicsXFunctionSpaceBackend(FunctionSpaceBackend):
    """A Backend that implements all operations related to FEniCSx function spaces."""

    # region FunctionSpaces
    @staticmethod
    def create_function_space(mesh: Any, function_type: Any, order: int) -> Any:
        return functionspace(mesh, (function_type, order))

    @staticmethod
    def create_test_function(function_space: Any) -> Any:
        return TestFunction(function_space)

    @staticmethod
    def create_trial_function(function_space: Any) -> Any:
        return TrialFunction(function_space)

    @staticmethod
    def create_function(function_space: Any) -> Any:
        return Function(function_space)

    # endregion
    # region Function manipulation

    @staticmethod
    def set_function_dofs(function: Any, values: Any) -> None:
        function.x.array[:] = values

    @staticmethod
    def get_function_value(function: Any, x: Any) -> Any:
        # Evaluate the function at the given point x
        return function.eval(x)

    @staticmethod
    def get_function_dofs(function: Any) -> Any:
        return function.x.array[:]

    # endregion
