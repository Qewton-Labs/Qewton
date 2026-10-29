"""validate: structural checks that need the whole problem.

Construction already rejects what is wrong locally (types, shapes of known symbols,
derivatives along missing axes, constraint axes). `validate` checks what only shows
across constraints and declarations, and reports it as diagnostics instead of
raising, so several findings arrive together:

- partitions: parts belong to the whole, are distinct, and each has a constraint;
- periodic declarations: a constraint relates the two sets, with the declared offset;
- delays: a shift back in time needs history on an interval, not on a slice;
- restrictions: the expression depends on the axes of the set it is restricted to;
- shapes: conflicts are errors, and symbols whose shape had to be guessed are warned.
"""

from __future__ import annotations

from collections.abc import Iterator

from qewton.equations.analysis.axes import free_axes, set_free_axes, sliced_axes
from qewton.equations.analysis.diagnostics import (
    Diagnostic,
    error,
    warning,
)
from qewton.equations.analysis.infer import infer_shapes
from qewton.equations.expr.nodes import Const, Expr, Restrict, Shift, walk
from qewton.equations.expr.ops import ShapeError
from qewton.equations.geometry.coordinates import root_of
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.problem import Problem

Found = Iterator[Diagnostic]


def validate(problem: Problem) -> tuple[Diagnostic, ...]:
    """All findings about `problem`, errors first."""
    found = [
        *_partitions(problem),
        *_periodic(problem),
        *_delays(problem),
        *_restrictions(problem),
        *_shapes(problem),
    ]
    return tuple(sorted(found, key=lambda d: d.severity.value != "error"))


def _nodes(problem: Problem) -> Iterator[tuple[Constraint, Expr]]:
    for constraint in problem.constraints:
        for side in (constraint.lhs, constraint.rhs):
            for node in walk(side):
                yield constraint, node


def _partitions(problem: Problem) -> Found:
    held_on = {constraint.on for constraint in problem.constraints}
    for partition in problem.partitions:
        if not partition.parts:
            yield error("A partition needs at least one part.", partition)
        if len(set(partition.parts)) != len(partition.parts):
            yield error("A part is listed twice.", partition)
        for part in partition.parts:
            if part.of != partition.whole:
                yield error(
                    f"{part} is a part of {part.of}, not of {partition.whole}.", partition
                )
            elif part not in held_on:
                yield warning(f"No constraint holds on {part}.", partition)


def _periodic(problem: Problem) -> Found:
    for declaration in problem.periodic:
        axis = root_of(declaration.axis)
        sides = {declaration.first, declaration.second}
        shifts = [
            node
            for constraint, node in _nodes(problem)
            if constraint.on in sides
            and isinstance(node, Shift)
            and root_of(node.axis) == axis
        ]
        if not shifts:
            yield warning(
                "No constraint on either set relates the two through a shift "
                f"along {axis.name}.",
                declaration,
            )
        for shift in shifts:
            if isinstance(shift.offset, Const) and not _matches(
                shift.offset.value, declaration.offset
            ):
                yield error(
                    f"{shift} shifts by {shift.offset.value}, but the sets are "
                    f"declared periodic with offset {declaration.offset}.",
                    declaration,
                )


def _matches(value: object, offset: object) -> bool:
    return value == offset or value == _negated(offset)


def _negated(offset: object) -> object:
    if isinstance(offset, tuple):
        return tuple(-v for v in offset)
    return -offset  # type: ignore[operator]


def _delays(problem: Problem) -> Found:
    if problem.time is None:
        return
    time = root_of(problem.time)
    shifts = [
        (constraint, node)
        for constraint, node in _nodes(problem)
        if isinstance(node, Shift) and root_of(node.axis) == time
    ]
    for constraint, shift in shifts:
        if isinstance(shift.offset, Const) and shift.offset.value > 0:
            yield warning(
                f"{shift} looks ahead in time: it is an advance, not a delay.", constraint
            )
    delays = [s for s in shifts if not _is_advance(s[1])]
    if not delays:
        return
    held_on = {c.on for c in problem.constraints if c.on is not None}
    intervals = {s for s in held_on if time in set_free_axes(s)}
    slices = {s for s in held_on if time in sliced_axes(s)}
    if len(intervals) < 2:
        yield warning(
            "A delay equation needs history on an interval before the start, as "
            "its own constraint.",
            delays[0][0],
        )
    for s in slices:
        yield warning(
            f"A condition on the slice {s} is not enough for a delay: the history "
            "has to be given on an interval.",
            s,
        )


def _is_advance(shift: Shift) -> bool:
    return isinstance(shift.offset, Const) and shift.offset.value > 0


def _restrictions(problem: Problem) -> Found:
    for constraint, node in _nodes(problem):
        if not isinstance(node, Restrict):
            continue
        target = node.on
        missing = target.axes - free_axes(node.expr)
        if missing:
            names = ", ".join(sorted(axis.name for axis in missing))
            yield warning(
                f"{node.expr} does not depend on {names}, so restricting it to "
                f"{target} has no effect.",
                constraint,
            )


def _shapes(problem: Problem) -> Found:
    try:
        result = infer_shapes(problem)
    except ShapeError as failure:
        yield error(str(failure))
        return
    for symbol in result.defaulted:
        yield warning(
            f"The output shape of {symbol.name} is not determined; assuming scalar. "
            "Declare it with dim to silence this.",
            symbol,
        )
