from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from qewton.backends.base import Backend, TensorType


@dataclass(frozen=True)
class LinearSolverOptions:
    """Backend-neutral configuration of a linear solver.

    Each FEM backend translates these options into its own format.
    `backend_options` is passed through unchanged (e.g. raw PETSc options)
    and takes precedence over the translated options.
    """

    kind: Literal["direct", "iterative"] = "direct"
    method: str | None = None  # e.g. "lu", "cg", "gmres" (None = backend default)
    preconditioner: str | None = None  # e.g. "ilu", "amg", "jacobi", "none"
    rtol: float = 1e-8
    atol: float = 1e-12
    max_iter: int = 1000
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NonlinearSolverOptions:
    """Backend-neutral configuration of a nonlinear solver."""

    method: str = "newton"
    rtol: float = 1e-8
    atol: float = 1e-10
    max_iter: int = 50
    linear: LinearSolverOptions = field(default_factory=LinearSolverOptions)
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveResult:
    """The outcome of a solve call."""

    solution: Any  # backend function object (e.g. dolfinx.fem.Function)
    converged: bool
    iterations: int
    residual_norm: float | None = None
    info: dict[str, Any] = field(default_factory=dict)  # backend-specific extras


class SolveBackend(Backend[TensorType]):
    """A Backend that solves the discrete systems arising from weak formulations.

    The backend itself is stateless. Solver state (assembled matrices,
    factorizations, Krylov/Newton solver objects) lives in opaque handles
    created by `create_linear_solver` / `create_nonlinear_solver`, so it can
    be reused across repeated solves (e.g. time stepping or parameter sweeps).
    """

    # region One-shot API
    @classmethod
    def linear(
        cls,
        a: Any,
        L: Any,
        bcs: Sequence[Any],
        u: Any,
        options: LinearSolverOptions | None = None,
    ) -> SolveResult:
        """Solves the linear problem a(u, v) = L(v) once."""
        handle = cls.create_linear_solver(a, L, bcs, u, options or LinearSolverOptions())
        return cls.solve(handle)

    @classmethod
    def nonlinear(
        cls,
        F: Any,
        u: Any,
        bcs: Sequence[Any],
        J: Any = None,
        options: NonlinearSolverOptions | None = None,
    ) -> SolveResult:
        """Solves the nonlinear problem F(u; v) = 0 once, using `u` as initial guess."""
        handle = cls.create_nonlinear_solver(
            F, u, bcs, J, options or NonlinearSolverOptions()
        )
        return cls.solve(handle)

    # endregion

    # region Solver handles
    @staticmethod
    @abstractmethod
    def create_linear_solver(
        a: Any, L: Any, bcs: Sequence[Any], u: Any, options: LinearSolverOptions
    ) -> Any:
        """Creates a reusable handle for the linear problem a(u, v) = L(v).

        Args:
            a: The bilinear form.
            L: The linear form.
            bcs: The Dirichlet boundary conditions.
            u: The function the solution is written into.
            options: The solver configuration.
        """
        pass

    @staticmethod
    @abstractmethod
    def create_nonlinear_solver(
        F: Any, u: Any, bcs: Sequence[Any], J: Any, options: NonlinearSolverOptions
    ) -> Any:
        """Creates a reusable handle for the nonlinear problem F(u; v) = 0.

        Args:
            F: The residual form.
            u: The unknown function (also the initial guess).
            bcs: The Dirichlet boundary conditions.
            J: The Jacobian form, or None to derive it automatically.
            options: The solver configuration.
        """
        pass

    @staticmethod
    @abstractmethod
    def solve(handle: Any) -> SolveResult:
        """Runs the solver of a handle. Forms are re-assembled, so changed
        coefficients (e.g. the previous time step) are taken into account."""
        pass

    # endregion

    # region Optional
    @staticmethod
    def solve_eigen(
        a: Any, m: Any, bcs: Sequence[Any], n_eigs: int, options: dict | None = None
    ) -> Any:
        """Solves the generalized eigenvalue problem a(u, v) = lambda * m(u, v)."""
        raise NotImplementedError(
            "Eigenvalue problems are not supported by this backend."
        )

    # endregion
