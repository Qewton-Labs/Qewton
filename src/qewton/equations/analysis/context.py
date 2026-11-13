"""AnalysisContext: every analysis result of one task, each computed once.

    ctx = AnalysisContext(task)
    ctx.ok                    # no errors in the checks
    ctx.classification        # computed on first access, then kept
    print(ctx.report())       # everything as one readable text

The passes in `prepare/`, `structure/` and `classification/` are plain functions of their
inputs, such as `build_rows(problem, unknowns, shapes)`. They never call each other.
This class is the one place that wires them together: each property passes the
results of earlier ones on, so what a pass depends on can be read off here. Nothing
is computed twice, and nothing is computed before it is asked for.

All passes after `normalized` work on the normalized problem, `ctx.problem`. The
problem as written stays in `ctx.task.problem`.

The functions at the end, such as `classify(task)`, are shortcuts for one result:
`classify(task)` is `AnalysisContext(task).classification`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from qewton.config.variables import Variable
from qewton.equations.diagnostics import Diagnostic, by_severity, errors
from qewton.equations.analysis.classification.classify import (
    Classification,
    build_classification,
)
from qewton.equations.analysis.classification.rows import Row, build_rows
from qewton.equations.analysis.prepare.infer import Inference, infer_shapes
from qewton.equations.analysis.prepare.normalization import Normalized, normalize
from qewton.equations.analysis.prepare.signatures import Signatures, build_signatures
from qewton.equations.analysis.prepare.validation import validate
from qewton.equations.analysis.structure.batch_axes import build_inherited_batch_axes
from qewton.equations.analysis.structure.batching import Batching, build_batching
from qewton.equations.analysis.structure.coupling import (
    CouplingGraph,
    build_coupling_graph,
)
from qewton.equations.analysis.structure.structure import (
    BatchPlacement,
    Split,
    build_batch_placement,
    build_decoupled_axes,
    split_constraint,
)
from qewton.equations.analysis.suggestions.base import Suggestion, suggest
from qewton.equations.analysis.training import TrainingMode, build_training_modes
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.expr.ops import Linearity
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.problem import Problem
from qewton.equations.model.task import Role, Task


@dataclass(frozen=True, eq=False)
class AnalysisContext:
    """Everything the analysis finds out about one task, computed on demand."""

    task: Task

    # Prepare: check the problem and bring it into the form the passes expect.

    @cached_property
    def normalized(self) -> Normalized:
        """The problem rewritten for analysis, with the constraints it came from."""
        return normalize(self.task.problem)

    @property
    def problem(self) -> Problem:
        """The normalized problem."""
        return self.normalized.problem

    @property
    def origin(self) -> tuple[Constraint, ...]:
        """The constraint of `task.problem` each normalized constraint came from."""
        return self.normalized.origin

    @cached_property
    def inference(self) -> Inference:
        """The output shape of every function; undeclared shapes are inferred."""
        return infer_shapes(self.problem)

    @cached_property
    def signatures(self) -> Signatures:
        """Inputs, batch axes, shape and evaluation sets of every function."""
        return build_signatures(
            self.task, self.problem, self.inference, self.inherited_batch_axes
        )

    @cached_property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """What the task and the checking passes found, errors first."""
        return by_severity(
            (
                *self.task.diagnostics,
                *validate(self.task.problem),
                *self.normalized.diagnostics,
                *self.signatures.diagnostics,
            )
        )

    @property
    def ok(self) -> bool:
        """Whether no pass found an error."""
        return not errors(self.diagnostics)

    # Structure: what the equations look like and how the unknowns couple.

    @cached_property
    def coupling(self) -> CouplingGraph:
        """The scalar equation/unknown graph, its matching and its blocks."""
        return build_coupling_graph(
            self.problem, self.task.unknowns, self.inference.shapes
        )

    @cached_property
    def splits(self) -> tuple[Split, ...]:
        """The operator/source split of every normalized constraint."""
        unknowns = frozenset(self.task.unknowns)
        return tuple(split_constraint(c, unknowns) for c in self.problem.constraints)

    @property
    def nonlinear(self) -> bool:
        """Whether an unknown occurs nonlinearly in some constraint."""
        return any(s.linearity is Linearity.NONLINEAR for s in self.splits)

    @cached_property
    def decoupled_axes(self) -> tuple[Variable, ...]:
        """Coordinates along which the unknowns can be solved for pointwise."""
        return build_decoupled_axes(self.problem, self.task.unknowns)

    @cached_property
    def inherited_batch_axes(self) -> dict[FunctionSymbol, tuple[Variable, ...]]:
        """The batch axes each unknown inherits from varying data it couples to."""
        return build_inherited_batch_axes(
            self.problem, self.task.unknowns, self.task.batch_axes
        )

    @cached_property
    def batch_placement(self) -> tuple[BatchPlacement, ...]:
        """Where each varying function enters the operator and source parts."""
        return build_batch_placement(self.splits, self.task.with_role(Role.VARYING))

    @cached_property
    def batching(self) -> Batching:
        """One entry per parameter axis: where and how it enters, whom it reaches."""
        return build_batching(self.task, self.inherited_batch_axes, self.batch_placement)

    @cached_property
    def training_modes(self) -> dict[FunctionSymbol, TrainingMode]:
        """How each unknown can be learned, from the samples the task has of it."""
        return build_training_modes(self.task, self.signatures)

    # Classify and suggest.

    @cached_property
    def rows(self) -> tuple[Row, ...]:
        """One row per normalized constraint: region, unknown orders, components."""
        return build_rows(self.problem, self.task.unknowns, self.inference.shapes)

    @cached_property
    def classification(self) -> Classification:
        """Equation and block types, condition kinds and counts, function spaces."""
        return build_classification(
            self.problem, self.task.unknowns, self.rows, self.coupling
        )

    @cached_property
    def suggestions(self) -> tuple[Suggestion, ...]:
        """The suggestions of every registered rule; see `analysis/suggest`."""
        return suggest(self)

    def report(self) -> str:
        """A readable summary: the task, signatures, normalized problem, batching,
        classification, suggestions and diagnostics.

        Classification and suggestions are left out if the checks found an error or
        the task has no unknowns, since both assume a valid problem to solve.
        """
        lines = _section("Task", str(self.task).splitlines() or ["(no roles given)"])
        lines += _section("Signatures", str(self.signatures).splitlines())
        lines += _section("Normalized", str(self.problem).splitlines())
        if self.batching:
            lines += _section("Batching", str(self.batching).splitlines())
        if self.ok and self.task.unknowns:
            lines += _section("Classification", str(self.classification).splitlines())
            suggestions = [str(s) for s in self.suggestions]
            lines += _section("Suggestions", suggestions or ["none"])
        diagnostics = [str(d) for d in self.diagnostics]
        lines += _section("Diagnostics", diagnostics or ["none"])
        return "\n".join(lines)


def _section(title: str, lines: list[str]) -> list[str]:
    return [title] + [f"  {line}" for line in lines]


# Shortcuts for a single result.


def signatures(task: Task) -> Signatures:
    """The signature of every function of the task's problem."""
    return AnalysisContext(task).signatures


def coupling_graph(task: Task) -> CouplingGraph:
    """The scalar coupling graph of the task's normalized problem."""
    return AnalysisContext(task).coupling


def split(task: Task) -> tuple[Split, ...]:
    """The operator/source split of every constraint of the normalized problem."""
    return AnalysisContext(task).splits


def decoupled_axes(task: Task) -> tuple[Variable, ...]:
    """Axes along which the unknowns have no differential or nonlocal coupling."""
    return AnalysisContext(task).decoupled_axes


def inherited_batch_axes(task: Task) -> dict[FunctionSymbol, tuple[Variable, ...]]:
    """Batch axes inherited by unknowns coupled to varying functions or sets."""
    return AnalysisContext(task).inherited_batch_axes


def batch_placement(task: Task) -> tuple[BatchPlacement, ...]:
    """Where each varying function enters each normalized constraint."""
    return AnalysisContext(task).batch_placement


def batching(task: Task) -> Batching:
    """One entry per parameter axis of `task`, ordered by axis name."""
    return AnalysisContext(task).batching


def training_modes(task: Task) -> dict[FunctionSymbol, TrainingMode]:
    """The training mode of every unknown."""
    return AnalysisContext(task).training_modes


def classify(task: Task) -> Classification:
    """Classify the equations, conditions and unknowns of a task."""
    return AnalysisContext(task).classification
