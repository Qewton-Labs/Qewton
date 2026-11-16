from typing import Any, Sequence

from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

from qewton.backends.fem.solve import (
    SolveBackend,
    LinearSolverOptions,
    NonlinearSolverOptions,
    SolveResult,
)


class FenicsXSolveBackend(SolveBackend):

    # region Solver handles
    @staticmethod
    def translate_options(options: LinearSolverOptions | NonlinearSolverOptions) -> dict:
        """Translates the solver options into a format suitable for FEniCSx."""
        petsc_options = {}

        if isinstance(options, LinearSolverOptions):
            if options.kind == "direct":
                petsc_options["ksp_type"] = "preonly"
                petsc_options["pc_type"] = options.method or "lu"

            else:
                petsc_options["ksp_type"] = options.method or "gmres"
                petsc_options["pc_type"] = options.preconditioner or "ilu"

                petsc_options["ksp_rtol"] = options.rtol
                petsc_options["ksp_atol"] = options.atol
                petsc_options["ksp_max_it"] = options.max_iter

        elif isinstance(options, NonlinearSolverOptions):
            petsc_options = {
                "snes_type": {
                    "newton": "newtonls",
                    "newton_tr": "newtontr",
                }.get(options.method, options.method),
                "snes_rtol": options.rtol,
                "snes_atol": options.atol,
                "snes_max_it": options.max_iter,
            }

            # Translate the nested linear solver options
            linear_options = FenicsXSolveBackend.translate_options(options.linear)
            petsc_options.update(linear_options)

            # Backend-specific options take precedence
            petsc_options.update(options.backend_options)

        return petsc_options

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
        return LinearProblem(
            a=a,
            L=L,
            bcs=bcs,
            u=u,
            petsc_options_prefix="linear_problem_",
            petsc_options=FenicsXSolveBackend.translate_options(options),
        )

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
        return NonlinearProblem(
            F,
            u,
            bcs=bcs,
            J=J,
            petsc_options_prefix="nonlinear_problem_",
            petsc_options=FenicsXSolveBackend.translate_options(options),
        )

    @staticmethod
    def solve(handle: Any) -> SolveResult:
        """Runs the solver of a handle. Forms are re-assembled, so changed
        coefficients (e.g. the previous time step) are taken into account."""
        # Linear problem
        if isinstance(handle, LinearProblem):
            solution = handle.solve()

            return SolveResult(
                solution=solution,
                converged=True,
                iterations=1,
            )

        # Nonlinear problem
        elif isinstance(handle, NonlinearProblem):
            handle.solve()
            snes = handle.solver
            return SolveResult(
                solution=handle.u,
                converged=int(snes.getConvergedReason()) > 0,  # type: ignore
                iterations=snes.getIterationNumber(),
                residual_norm=snes.getFunctionNorm(),
                info={
                    "converged_reason": snes.getConvergedReason(),
                },
            )

        else:
            raise TypeError(f"Unsupported solver handle: {type(handle).__name__}")

    # endregion
