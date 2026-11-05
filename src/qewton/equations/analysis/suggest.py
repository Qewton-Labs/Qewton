"""suggest: solution methods for a task, each with the evidence behind it.

    for suggestion in suggest(task):
        print(suggestion)

A rule reads the `Findings` (classification, batching, linearity, decoupled axes) and
yields `Suggestion`s. Each suggestion names the method, says why, and carries the
`Fact`s it was derived from, so a report can always explain itself. Rules are plain
functions added with `@rule`; adding a method means adding one function.

The suggestions are starting points, not a ranking: several can apply at once, for
instance a method for the equation type and one for the batching.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from qewton.equations.analysis.classify.classify import Classification, classify
from qewton.equations.analysis.classify.equations import PdeType
from qewton.equations.analysis.classify.facts import Fact
from qewton.equations.analysis.structure.batching import (
    BatchEntry,
    BatchMode,
    Batching,
    batching,
)
from qewton.equations.analysis.structure.structure import decoupled_axes, split
from qewton.equations.analysis.training import TrainingMode, training_modes
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.expr.ops import Linearity
from qewton.equations.model.constraints import Relation
from qewton.equations.model.task import Task


@dataclass(frozen=True)
class Suggestion:
    """A method, the id of the rule that proposed it, and why."""

    rule: str
    method: str
    why: str
    evidence: tuple[Fact, ...] = ()

    def __str__(self) -> str:
        return f"{self.method}  <- {self.why}"


@dataclass(frozen=True)
class Findings:
    """What the rules look at."""

    task: Task
    classification: Classification
    batching: Batching
    nonlinear: bool
    decoupled: tuple[str, ...]
    training: dict[FunctionSymbol, TrainingMode]

    def blocks(self, kind: PdeType) -> tuple[Fact, ...]:
        return tuple(f for f in self.classification.blocks if f.value is kind)


Rule = Callable[[Findings], Iterator[Suggestion]]
RULES: list[Rule] = []


def rule(function: Rule) -> Rule:
    """Register a suggestion rule."""
    RULES.append(function)
    return function


def _by_block(
    kind: PdeType, rule_id: str, method: str, why: str
) -> Callable[[Findings], Iterator[Suggestion]]:
    def run(found: Findings) -> Iterator[Suggestion]:
        blocks = found.blocks(kind)
        if blocks:
            yield Suggestion(rule_id, method, why, blocks)

    return run


# Methods tied to the type of a coupled block.
for _kind, _id, _method, _why in (
    (
        PdeType.ELLIPTIC,
        "elliptic",
        "continuous Lagrange FEM; CG with multigrid, or a direct solver",
        "elliptic block",
    ),
    (
        PdeType.PARABOLIC,
        "parabolic",
        "method of lines with implicit time stepping (BDF, Crank-Nicolson)",
        "parabolic block",
    ),
    (
        PdeType.HYPERBOLIC_TRANSPORT,
        "transport",
        "FV or DG with an upwind flux; explicit Runge-Kutta under a CFL limit",
        "hyperbolic transport block",
    ),
    (
        PdeType.HYPERBOLIC_WAVE,
        "wave",
        "FEM or DG in space with a symplectic or Newmark time stepper",
        "hyperbolic wave block",
    ),
    (
        PdeType.SADDLE_POINT,
        "saddle_point",
        "inf-sup stable pairs (Taylor-Hood); block preconditioner",
        "saddle-point block",
    ),
    (
        PdeType.ODE,
        "ode",
        "an ODE integrator (implicit for stiff problems)",
        "ODE block: no spatial derivative",
    ),
):
    RULES.append(_by_block(_kind, _id, _method, _why))


@rule
def pointwise(found: Findings) -> Iterator[Suggestion]:
    blocks = found.blocks(PdeType.ALGEBRAIC)
    if blocks:
        yield Suggestion(
            "algebraic",
            "a pointwise solve; boundary conditions do not apply",
            "algebraic block: no derivatives",
            blocks,
        )


@rule
def newton(found: Findings) -> Iterator[Suggestion]:
    if found.nonlinear:
        yield Suggestion(
            "nonlinear",
            "Newton (or Picard) iteration",
            "an unknown occurs nonlinearly in the operator",
        )


@rule
def variational_inequality(found: Findings) -> Iterator[Suggestion]:
    facts = tuple(f for f in found.classification.equations if f.rule == "relation")
    relations = {f.value for f in facts}
    if relations:
        yield Suggestion(
            "inequality",
            (
                "active-set or semismooth Newton"
                if Relation.COMPLEMENTARITY in relations
                else "a projection or active-set method"
            ),
            "a complementarity or inequality constraint",
            facts,
        )


@rule
def sequence_of_stationary_solves(found: Findings) -> Iterator[Suggestion]:
    time = found.task.problem.time
    if time is not None and time.name in found.decoupled:
        yield Suggestion(
            "decoupled_time",
            "a sequence of stationary solves, one per time",
            f"no derivative or shift along {time.name}",
        )


def _batch_fact(entry: BatchEntry) -> Fact:
    return Fact("batch_mode", entry.axis.name, entry.mode, str(entry), entry.constraints)


_BATCH_METHODS = {
    BatchMode.SOURCE_ONLY: (
        "factor the operator once, solve with many right-hand sides",
        "it only enters the source part",
    ),
    BatchMode.OPERATOR_AFFINE: (
        "reduced basis with an offline/online split",
        "it enters the operator as a separable factor",
    ),
    BatchMode.OPERATOR_NONLINEAR: (
        "empirical interpolation (EIM) or a learned surrogate",
        "it enters the operator in a way that does not separate",
    ),
    BatchMode.GEOMETRY: (
        "pull back to a reference domain, if the topology is fixed",
        "the domain changes from instance to instance",
    ),
}


@rule
def batched(found: Findings) -> Iterator[Suggestion]:
    for entry in found.batching.entries:
        if entry.mode in _BATCH_METHODS:
            method, why = _BATCH_METHODS[entry.mode]
            yield Suggestion(
                f"batch_{entry.mode.name.lower()}",
                method,
                f"{entry.axis.name}: {why}",
                (_batch_fact(entry),),
            )


_TRAINING_METHODS = {
    TrainingMode.PHYSICS_INFORMED: "physics-informed training (PINN, VPINN, Deep Ritz)",
    TrainingMode.HYBRID: "residual loss plus a data misfit on the observed set",
}


def _training_method(task: Task, mode: TrainingMode) -> str:
    if mode is not TrainingMode.DATA_DRIVEN:
        method = _TRAINING_METHODS[mode]
        return (
            f"{method}, one network over the varying inputs"
            if task.batch_axes
            else method
        )
    if task.batch_axes:
        return "supervised operator learning (FNO, DeepONet); equations optional"
    return "supervised regression of a surrogate; equations optional"


@rule
def learning(found: Findings) -> Iterator[Suggestion]:
    """Learning-based methods, once data or varying inputs make them relevant."""
    task = found.task
    if not (task.observed or task.batch_axes):
        return
    for symbol, mode in sorted(
        found.training.items(), key=lambda e: str(e[0].variable.name)
    ):
        name = str(symbol.variable.name)
        data = task.observed.get(symbol)
        why = f"{name}: {mode.value}" + (f" ({data})" if data else ", no samples")
        if data is not None and data.noisy:
            why += "; noisy samples call for a regularized or robust misfit"
        yield Suggestion(
            f"learning_{mode.name.lower()}",
            _training_method(task, mode),
            why,
            (Fact("training_mode", name, mode, why),),
        )


def findings(task: Task) -> Findings:
    """Collect what the rules need; the task must have unknowns."""
    nonlinear = any(s.linearity is Linearity.NONLINEAR for s in split(task))
    decoupled = tuple(str(a.name) for a in decoupled_axes(task))
    return Findings(
        task, classify(task), batching(task), nonlinear, decoupled, training_modes(task)
    )


def suggest(task: Task) -> tuple[Suggestion, ...]:
    """The suggestions of every rule for `task`, in rule order."""
    found = findings(task)
    return tuple(s for run in RULES for s in run(found))
