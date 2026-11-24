from typing import Any
import numpy as np

from dolfinx.fem import Function, functionspace, Constant
from ufl import TestFunction, TrialFunction
from dolfinx import geometry

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

    @staticmethod
    def create_constant(value: Any, mesh: Any) -> Constant:
        return Constant(mesh, value)

    @staticmethod
    def set_constant_value(constant: Constant, value: Any) -> None:
        constant.value = value

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

    @staticmethod
    def get_function_on_mesh_vertices(function: Any, mesh: Any) -> Any:
        # Mesh vertex coordinates
        vertices = mesh.geometry.x[: mesh.topology.index_map(0).size_local]

        # Find cells containing the vertices
        bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
        candidates = geometry.compute_collisions_points(bb_tree, vertices)
        collisions = geometry.compute_colliding_cells(mesh, candidates, vertices)

        # Evaluate function
        cells = np.array([collisions.links(i)[0] for i in range(len(vertices))])
        return function.eval(vertices, cells)

    # endregion
