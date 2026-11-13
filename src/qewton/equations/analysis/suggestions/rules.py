"""The built-in suggestion rules.

To add a method, add a `Rule` subclass here, or in any module that is imported. Most
rules belong to a family whose members differ only in data. Such a rule fills in a
small base class instead of writing `check` itself:

- `BlockTypeRule`: a method for each type of coupled block (elliptic, parabolic, ...);
- `BatchModeRule`: a method for each way a parameter enters the equations;
- `TrainingRule`: a method for each kind of data an unknown has.

The order of the classes is the order of the suggestions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

from qewton.equations.analysis.classification.equations import PdeType
from qewton.equations.analysis.classification.facts import Fact
from qewton.equations.analysis.structure.batching import BatchMode
from qewton.equations.analysis.suggestions.base import Category, Rule, Suggestion
from qewton.equations.analysis.training import TrainingMode
from qewton.equations.geometry.coordinates import root_of
from qewton.equations.model.constraints import Relation

if TYPE_CHECKING:
    from qewton.equations.analysis.context import AnalysisContext


# Discretization: one method per type of coupled block.


class BlockTypeRule(Rule, register=False):
    """Suggest `method` once if the task has a coupled block of type `kind`.

    The reason given is `why`, or "<kind> block" if it is not set.
    """

    kind: ClassVar[PdeType]
    why: ClassVar[str | None] = None

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        blocks = ctx.classification.blocks_of(self.kind)
        if blocks:
            yield self.suggestion(self.why or f"{self.kind.value} block", blocks)


class LagrangeFem(BlockTypeRule):
    kind = PdeType.ELLIPTIC
    method = "continuous Lagrange FEM; CG with multigrid, or a direct solver"


class MethodOfLines(BlockTypeRule):
    kind = PdeType.PARABOLIC
    method = "method of lines with implicit time stepping (BDF, Crank-Nicolson)"


class UpwindScheme(BlockTypeRule):
    kind = PdeType.HYPERBOLIC_TRANSPORT
    method = "FV or DG with an upwind flux; explicit Runge-Kutta under a CFL limit"


class WaveTimeStepping(BlockTypeRule):
    kind = PdeType.HYPERBOLIC_WAVE
    method = "FEM or DG in space with a symplectic or Newmark time stepper"


class InfSupStablePair(BlockTypeRule):
    kind = PdeType.SADDLE_POINT
    method = "inf-sup stable pairs (Taylor-Hood); block preconditioner"


class OdeIntegrator(BlockTypeRule):
    kind = PdeType.ODE
    method = "an ODE integrator (implicit for stiff problems)"
    why = "ODE block: no spatial derivative"


class PointwiseSolve(BlockTypeRule):
    kind = PdeType.ALGEBRAIC
    method = "a pointwise solve; boundary conditions do not apply"
    why = "algebraic block: no derivatives"


# Solvers: what the structure of the equations asks of the solver.


class Newton(Rule):
    category = Category.SOLVER
    method = "Newton (or Picard) iteration"

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        if ctx.nonlinear:
            yield self.suggestion("an unknown occurs nonlinearly in the operator")


class ActiveSet(Rule):
    """Inequalities; a complementarity calls for semismooth Newton."""

    category = Category.SOLVER
    method = "a projection or active-set method"

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        facts = ctx.classification.relations
        if not facts:
            return
        why = "a complementarity or inequality constraint"
        if any(f.value is Relation.COMPLEMENTARITY for f in facts):
            yield self.suggestion(why, facts, method="active-set or semismooth Newton")
        else:
            yield self.suggestion(why, facts)


class StationarySolves(Rule):
    category = Category.SOLVER
    method = "a sequence of stationary solves, one per time"

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        time = ctx.problem.time
        if time is not None and root_of(time) in ctx.decoupled_axes:
            yield self.suggestion(f"no derivative or shift along {time.name}")


# Batching: one method per way a parameter axis enters the problem.


class BatchModeRule(Rule, register=False):
    """Suggest `method` for every parameter axis that enters the problem as `mode`.

    The reason given is "<axis>: <reason>".
    """

    category = Category.BATCH
    mode: ClassVar[BatchMode]
    reason: ClassVar[str]

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        for entry in ctx.batching.entries:
            if entry.mode is self.mode:
                name = entry.axis.name
                fact = Fact("batch_mode", name, entry.mode, str(entry), entry.constraints)
                yield self.suggestion(f"{name}: {self.reason}", (fact,))


class SingleFactorization(BatchModeRule):
    mode = BatchMode.SOURCE_ONLY
    method = "factor the operator once, solve with many right-hand sides"
    reason = "it only enters the source part"


class ReducedBasis(BatchModeRule):
    mode = BatchMode.OPERATOR_AFFINE
    method = "reduced basis with an offline/online split"
    reason = "it enters the operator as a separable factor"


class EmpiricalInterpolation(BatchModeRule):
    mode = BatchMode.OPERATOR_NONLINEAR
    method = "empirical interpolation (EIM) or a learned surrogate"
    reason = "it enters the operator in a way that does not separate"


class ReferenceDomain(BatchModeRule):
    mode = BatchMode.GEOMETRY
    method = "pull back to a reference domain, if the topology is fixed"
    reason = "the domain changes from instance to instance"


# Learning: one method per kind of data an unknown has.


class TrainingRule(Rule, register=False):
    """Suggest `method` for every unknown whose training mode is `mode`.

    Only once samples or varying inputs make learning relevant at all. With varying
    inputs, `batched_method` is suggested instead of `method`.
    """

    category = Category.LEARNING
    mode: ClassVar[TrainingMode]
    batched_method: ClassVar[str]

    def check(self, ctx: AnalysisContext) -> Iterator[Suggestion]:
        task = ctx.task
        if not (task.observed or task.batch_axes):
            return
        method = self.batched_method if task.batch_axes else self.method
        for symbol, mode in ctx.training_modes.items():
            if mode is not self.mode:
                continue
            name = str(symbol.variable.name)
            data = task.observed.get(symbol)
            why = f"{name}: {mode.value}"
            why += ", no samples" if data is None else f" ({data})"
            if data is not None and data.noisy:
                why += "; noisy samples call for a regularized or robust misfit"
            yield self.suggestion(why, (Fact("training_mode", name, mode, why),), method)


_ONE_NETWORK = ", one network over the varying inputs"


class PhysicsInformed(TrainingRule):
    mode = TrainingMode.PHYSICS_INFORMED
    method = "physics-informed training (PINN, VPINN, Deep Ritz)"
    batched_method = method + _ONE_NETWORK


class HybridTraining(TrainingRule):
    mode = TrainingMode.HYBRID
    method = "residual loss plus a data misfit on the observed set"
    batched_method = method + _ONE_NETWORK


class SupervisedSurrogate(TrainingRule):
    mode = TrainingMode.DATA_DRIVEN
    method = "supervised regression of a surrogate; equations optional"
    batched_method = "supervised operator learning (FNO, DeepONet); equations optional"
