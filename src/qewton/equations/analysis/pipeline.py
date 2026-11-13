"""The analysis pipeline: run the passes on a task and report what was found.

    ctx = run_pipeline(task)      # the checks run now, everything else on demand
    print(report(ctx))

`run_pipeline` builds the task's `AnalysisContext` and runs the checking passes
(validate, normalize, signatures) right away, so `ctx.ok` and `ctx.diagnostics` are
ready. Structure, classification and suggestions are computed when first read, and
`report` reads them only if the checks found no error.
"""

from __future__ import annotations

from qewton.equations.analysis.context import AnalysisContext
from qewton.equations.model.task import Task


def run_pipeline(task: Task) -> AnalysisContext:
    """The analysis context of `task`, with the checking passes already run."""
    ctx = AnalysisContext(task)
    _ = ctx.diagnostics
    return ctx


def _section(title: str, lines: list[str]) -> list[str]:
    return [title] + [f"  {line}" for line in lines]


def report(ctx: AnalysisContext) -> str:
    """A readable summary: the task, signatures, batching, classification, suggestions
    and the findings."""
    lines: list[str] = []
    task_lines = str(ctx.task).splitlines()
    lines += _section("Task", task_lines or ["(no roles given)"])
    lines += _section("Signatures", str(ctx.signatures).splitlines())
    lines += _section("Normalized", str(ctx.problem).splitlines())
    if ctx.batching:
        lines += _section("Batching", str(ctx.batching).splitlines())
    if ctx.ok and ctx.task.unknowns:
        lines += _section("Classification", str(ctx.classification).splitlines())
        suggestions = [str(s) for s in ctx.suggestions]
        lines += _section("Suggestions", suggestions or ["none"])
    lines += _section("Diagnostics", [str(d) for d in ctx.diagnostics] or ["none"])
    return "\n".join(lines)
