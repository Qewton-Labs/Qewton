from qewton.backends.fem.solve import (
    SolveBackend,
    LinearSolverOptions,
    NonlinearSolverOptions,
    SolveResult,
)


class FenicsxSolveBackend(SolveBackend):

    # region Solver handles
    @staticmethod
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
    def solve(handle: Any) -> SolveResult:
        """Runs the solver of a handle. Forms are re-assembled, so changed
        coefficients (e.g. the previous time step) are taken into account."""
        pass

    # endregion
