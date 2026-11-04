"""Batching: what each parameter axis of a task does to the problem.

A varying function or set adds a parameter axis, one entry per instance. For each of
them `batching(task)` says where it enters the equations, how (the mode), and which
unknowns end up depending on it. Instances never couple: a parameter axis appears in
no expression, so there is no derivative, integral or shift along it, and the
instances can always be solved independently.

The mode only describes the structure. What to do with it, such as factoring the
operator once, is a suggestion and belongs to M6.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from qewton.config.variables import Variable
from qewton.equations.analysis.checks.diagnostics import Diagnostic, warning
from qewton.equations.analysis.structure.batch_axes import inherited_batch_axes
from qewton.equations.analysis.structure.structure import BatchPlacement, batch_placement
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.geometry.sets import SetNode
from qewton.equations.model.task import Task


class BatchMode(Enum):
    SOURCE_ONLY = "source only"
    OPERATOR_AFFINE = "operator, affine"
    OPERATOR_NONLINEAR = "operator, not separable"
    GEOMETRY = "geometry"
    UNUSED = "unused"


@dataclass(frozen=True)
class BatchEntry:
    """One parameter axis.

    `owner` is the varying function or set. `placements` locate a function in the
    operator and source parts of the normalized constraints (a varying set has none),
    and `reaches` lists the unknowns that depend on the axis.
    """

    axis: Variable
    owner: FunctionSymbol | SetNode
    mode: BatchMode
    placements: tuple[BatchPlacement, ...]
    reaches: tuple[FunctionSymbol, ...]

    @property
    def constraints(self) -> tuple[int, ...]:
        """The zero-based indices of the constraints the owner occurs in."""
        return tuple(sorted({p.constraint_index for p in self.placements}))

    def __str__(self) -> str:
        name = (
            self.owner.variable.name
            if isinstance(self.owner, FunctionSymbol)
            else self.owner
        )
        where = ", ".join(f"C{i + 1}" for i in self.constraints)
        reaches = ", ".join(str(s.variable.name) for s in self.reaches) or "no unknown"
        place = f" in {where}" if where else ""
        return f"{self.axis.name} ({name}): {self.mode.value}{place}; reaches {reaches}"


@dataclass(frozen=True)
class Batching:
    entries: tuple[BatchEntry, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __str__(self) -> str:
        if not self.entries:
            return "no batch axes"
        lines = [str(entry) for entry in self.entries]
        lines.append("instances are independent: parallel over the batch")
        return "\n".join(lines)


def _function_mode(placements: tuple[BatchPlacement, ...]) -> BatchMode:
    operator = [p for p in placements if p.part == "operator"]
    if not placements:
        return BatchMode.UNUSED
    if not operator:
        return BatchMode.SOURCE_ONLY
    if all(p.affine for p in operator):
        return BatchMode.OPERATOR_AFFINE
    return BatchMode.OPERATOR_NONLINEAR


def batching(task: Task) -> Batching:
    """One entry per parameter axis of `task`, ordered by axis name."""
    inherited = inherited_batch_axes(task)
    placements = batch_placement(task)
    entries, notes = [], []
    for owner, axis in sorted(task.batch_axes.items(), key=lambda it: str(it[1].name)):
        reaches = tuple(s for s in task.unknowns if axis in inherited.get(s, ()))
        if isinstance(owner, FunctionSymbol):
            own = tuple(p for p in placements if p.symbol is owner)
            mode = _function_mode(own)
        else:
            own, mode = (), BatchMode.GEOMETRY
        entries.append(BatchEntry(axis, owner, mode, own, reaches))
        if mode is BatchMode.UNUSED:
            notes.append(warning(f"{owner} varies but occurs in no constraint.", owner))
    return Batching(tuple(entries), tuple(notes))
