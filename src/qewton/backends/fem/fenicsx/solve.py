from itertools import count
from typing import Any, Sequence

from dolfinx.fem.petsc import (
    LinearProblem,
    NonlinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    assign,
    set_bc,
)
from petsc4py import PETSc

from qewton.backends.fem.solve import (
    SolveBackend,
    LinearSolverOptions,
    NonlinearSolverOptions,
    SolveResult,
)

# PETSc options prefixes should be unique within a programme, so that options and
# monitors meant for one solver do not reach another.
_handle_ids = count()


def _prefix(kind: str) -> str:
    return f"qewton_{kind}_{next(_handle_ids)}_"


class ReusableLinearProblem(LinearProblem):
    """A dolfinx `LinearProblem` that reassembles its operator only when asked to.

    `LinearProblem.solve()` zeroes and reassembles the matrix on every call, which
    also forces PETSc to set up the preconditioner (e.g. an LU factorization)
    again. Here `solve(reassemble_operator=False)` keeps the matrix, so the KSP
    reuses its factorization, and only the right-hand side is reassembled. Changed
    Dirichlet values are still applied, through the lifting of the right-hand side.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operator_assembled = False

    def solve(self, reassemble_operator: bool = True):  # type: ignore[override]
        if isinstance(self.u, Sequence):
            # Blocked and nested systems keep dolfinx's own assembly.
            self._operator_assembled = True
            return super().solve()
        if reassemble_operator or not self._operator_assembled:
            self._assemble_operator()
        self._assemble_rhs()
        self.solver.solve(self.b, self.x)
        self.x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD  # type: ignore
        )
        assign(self.x, self.u)
        return self.u

    def _assemble_operator(self) -> None:
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a, bcs=self.bcs)  # type: ignore[arg-type]
        self.A.assemble()
        if self.P_mat is not None:
            self.P_mat.zeroEntries()
            assemble_matrix(self.P_mat, self.preconditioner, bcs=self.bcs)  # type: ignore
            self.P_mat.assemble()
        self._operator_assembled = True

    def _assemble_rhs(self) -> None:
        with self.b.localForm() as local:
            local.set(0.0)
        assemble_vector(self.b, self.L)  # type: ignore[arg-type]
        bcs = self.bcs or []
        apply_lifting(self.b, [self.a], bcs=[bcs])  # type: ignore[arg-type]
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE  # type: ignore
        )
        set_bc(self.b, bcs)


class FenicsXSolveBackend(SolveBackend):

    # region Solver handles
    @staticmethod
    def translate_options(options: LinearSolverOptions | NonlinearSolverOptions) -> dict:
        """Translates the solver options into PETSc options.

        `backend_options` are passed through and take precedence over the translated
        options, for linear and nonlinear solvers alike.
        """
        petsc_options: dict[str, Any] = {}

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

            # Translate the nested linear solver options (with their own
            # backend options).
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
        return ReusableLinearProblem(
            a=a,
            L=L,
            bcs=bcs,
            u=u,
            petsc_options_prefix=_prefix("linear"),
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
            petsc_options_prefix=_prefix("nonlinear"),
            petsc_options=FenicsXSolveBackend.translate_options(options),
        )

    @staticmethod
    def solve(handle: Any, reassemble_operator: bool = True) -> SolveResult:
        """Runs the solver of a handle; see `SolveBackend.solve`."""
        if isinstance(handle, ReusableLinearProblem):
            solution = handle.solve(reassemble_operator=reassemble_operator)
            ksp = handle.solver
            return SolveResult(
                solution=solution,
                converged=int(ksp.getConvergedReason()) > 0,
                iterations=ksp.getIterationNumber(),
                residual_norm=ksp.getResidualNorm(),
                info={"converged_reason": ksp.getConvergedReason()},
            )

        if isinstance(handle, NonlinearProblem):
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

        raise TypeError(f"Unsupported solver handle: {type(handle).__name__}")

    # endregion
