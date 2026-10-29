"""The analysis pipeline: run every pass on a task and report what was found.

    ctx = run_pipeline(task)      # validate, normalize, signatures
    print(report(ctx))

Each pass reads the result of the earlier ones and adds diagnostics. No pass changes
a node: normalization builds a new problem, kept in `ctx.problem`, and the problem
you built stays in `ctx.task.problem`. Passes for structure, classification and
suggestions will join later, and each adds a field to the context.
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.equations.analysis.diagnostics import Diagnostic, Severity, errors
from qewton.equations.analysis.normalize import normalize
from qewton.equations.analysis.signatures import Signatures, signatures
from qewton.equations.analysis.validate import validate
from qewton.equations.model.problem import Problem
from qewton.equations.model.task import Task

_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass(frozen=True)
class AnalysisContext:
    """Everything the passes found out about one task.

    `problem` is the normalized problem and `origin` the constraints it came from.
    `diagnostics` holds the findings of the task and of every pass, errors first.
    """

    task: Task
    problem: Problem
    origin: tuple
    signatures: Signatures
    diagnostics: tuple[Diagnostic, ...]

    @property
    def ok(self) -> bool:
        """Whether no pass found an error."""
        return not errors(self.diagnostics)


def run_pipeline(task: Task) -> AnalysisContext:
    """Validate the task's problem, normalize it, and compute the signatures."""
    checked = validate(task.problem)
    normalized = normalize(task.problem)
    table = signatures(task)
    found = (
        *task.diagnostics,
        *checked,
        *normalized.diagnostics,
        *table.diagnostics,
    )
    ordered = tuple(sorted(found, key=lambda d: _ORDER[d.severity]))
    return AnalysisContext(task, normalized.problem, normalized.origin, table, ordered)


def _section(title: str, lines: list[str]) -> list[str]:
    return [title] + [f"  {line}" for line in lines]


def report(ctx: AnalysisContext) -> str:
    """A readable summary: the task, the signatures and the findings."""
    lines: list[str] = []
    task_lines = str(ctx.task).splitlines()
    lines += _section("Task", task_lines or ["(no roles given)"])
    lines += _section("Signatures", str(ctx.signatures).splitlines())
    lines += _section("Normalized", str(ctx.problem).splitlines())
    if ctx.diagnostics:
        lines += _section("Diagnostics", [str(d) for d in ctx.diagnostics])
    else:
        lines += _section("Diagnostics", ["none"])
    return "\n".join(lines)
