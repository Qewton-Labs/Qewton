from __future__ import annotations

from typing import Any, Callable

import numpy as np
import ufl
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, la, mesh as dmesh

from qewton.backends.fem.form import FormBackend


class FenicsXFormBackend(FormBackend):
    """A FormBackend that implements all operations related to FEniCSx.

    Expressions and forms are plain UFL objects; assembly and boundary
    conditions use dolfinx.
    """

    # region Math operations
    @staticmethod
    def add(a: Any, b: Any) -> Any:
        return a + b

    @staticmethod
    def multiply(a: Any, b: Any) -> Any:
        return a * b

    @staticmethod
    def subtract(a: Any, b: Any) -> Any:
        return a - b

    @staticmethod
    def divide(a: Any, b: Any) -> Any:
        return a / b

    @staticmethod
    def power(a: Any, b: Any) -> Any:
        return a**b

    @staticmethod
    def negate(a: Any) -> Any:
        return -a

    @staticmethod
    def abs(a: Any) -> Any:
        return ufl.algebra.Abs(ufl.as_ufl(a))

    @staticmethod
    def sqrt(a: Any) -> Any:
        return ufl.sqrt(a)

    @staticmethod
    def dot(a: Any, b: Any) -> Any:
        return ufl.dot(a, b)

    @staticmethod
    def transpose(a: Any) -> Any:
        return ufl.transpose(a)

    @staticmethod
    def inner(a: Any, b: Any) -> Any:
        return ufl.inner(a, b)

    # endregion

    # region Assembly
    @staticmethod
    def assemble(form: Any) -> Any:
        """Assembles a form depending on its rank.

        Returns:
            rank 0 (functional): the globally reduced scalar value.
            rank 1 (linear form): a ``dolfinx.la.Vector``.
            rank 2 (bilinear form): a ``dolfinx.la.MatrixCSR``.
        """
        compiled = form if isinstance(form, fem.Form) else fem.form(form)

        if compiled.rank == 0:
            local = fem.assemble_scalar(compiled)
            return compiled.mesh.comm.allreduce(local, op=MPI.SUM)
        if compiled.rank == 1:
            vec = fem.assemble_vector(compiled)
            vec.scatter_reverse(la.InsertMode.add)
            return vec
        if compiled.rank == 2:
            mat = fem.assemble_matrix(compiled)
            mat.scatter_reverse()
            return mat
        raise ValueError(f"Cannot assemble a form of rank {compiled.rank}.")

    # endregion

    # region Derivatives
    @staticmethod
    def gradient(u: Any) -> Any:
        return ufl.grad(u)

    @staticmethod
    def rotation(u: Any) -> Any:
        """The curl. For a 2D vector field this is the scalar rotation,
        for a 2D scalar field the vector rotation."""
        return ufl.curl(u)

    @staticmethod
    def divergence(u: Any) -> Any:
        return ufl.div(u)

    @staticmethod
    def derivative(u: Any, direction: int = 0) -> Any:
        """The partial derivative of u w.r.t. the spatial coordinate `direction`."""
        return ufl.derivative(u, ufl.SpatialCoordinate(u.ufl_domain())[direction])

    @staticmethod
    def time_derivative(u: Any, order: int) -> Any:
        raise NotImplementedError("TODO!")

    @staticmethod
    def hessian(u: Any) -> Any:
        return ufl.grad(ufl.grad(u))

    @staticmethod
    def symmetric_gradient(u: Any) -> Any:
        return ufl.sym(ufl.grad(u))

    @staticmethod
    def jacobian(u: Any) -> Any:
        """The spatial Jacobian of a vector field, J_ij = du_i/dx_j."""
        return ufl.grad(u)

    # endregion

    # region Integrals
    @staticmethod
    def integrate(form: Any, measure: Any) -> Any:
        """Integrates the integrand `form` over `measure` (e.g. ``ufl.dx``, ``ufl.ds(1)``)."""
        return form * measure

    # endregion

    # region Boundary conditions
    @staticmethod
    def build_dirichlet(
        boundary_value: Any, boundary: Any, function_space: fem.FunctionSpace
    ) -> fem.DirichletBC:
        """Builds a Dirichlet boundary condition.

        Args:
            boundary_value: A ``fem.Function``, a ``fem.Constant``, a number or
                array (constant value), or a callable ``x -> values`` that is
                interpolated into the function space.
            boundary: Either a callable marker ``x -> bool array`` that selects
                boundary facets geometrically, or an array of facet indices.
            function_space: The function space the condition acts on.
        """
        V = function_space
        msh = V.mesh
        fdim = msh.topology.dim - 1

        if callable(boundary):
            facets = dmesh.locate_entities_boundary(msh, fdim, boundary)
        else:
            facets = np.asarray(boundary, dtype=np.int32)

        msh.topology.create_connectivity(fdim, msh.topology.dim)
        dofs = fem.locate_dofs_topological(V, fdim, facets)

        if isinstance(boundary_value, fem.Function):
            return fem.dirichletbc(boundary_value, dofs)
        if isinstance(boundary_value, fem.Constant):
            return fem.dirichletbc(boundary_value, dofs, V)
        if callable(boundary_value):
            u_bc = fem.Function(V)
            u_bc.interpolate(boundary_value)
            return fem.dirichletbc(u_bc, dofs)

        value = fem.Constant(msh, np.asarray(boundary_value, dtype=default_scalar_type))
        return fem.dirichletbc(value, dofs, V)

    # endregion
