"""normalize: rewrite a problem into the form the later passes expect.

Three rewrites, in this order:

1. Time lifting. A constraint written without time, such as a
Dirichlet condition `w = g on Gamma_D` for a function w(x, t), means "at every time".
Lifting makes that explicit by moving it to `Gamma_D x I`. Initial conditions are not
lifted: restricting to a time slice already removed t from their sides.
2. `lap(u, axes)` becomes `div(grad(u, axes), axes)`, so that -div(nu grad u) and
   -nu lap u are analysed the same way. Both give the same shape.
3. Residual form: `lhs REL rhs` becomes `lhs - rhs REL 0`. A right side that is already
   zero stays as it is, and a complementarity `lhs ⊥ rhs` is left alone, since its two
   sides are not a difference.

The original problem is never changed. `Normalized.origin[i]` is the constraint that
`Normalized.problem.constraints[i]` came from, for display.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from qewton.equations.expr.axes import free_axes
from qewton.equations.diagnostics import Diagnostic, info, warning
from qewton.equations.expr.nodes import Apply, Const, Expr, transform
from qewton.equations.expr.ops import OPS
from qewton.equations.geometry.coordinates import root_of
from qewton.equations.geometry.sets import (
    Domain,
    Product,
    SetNode,
    set_free_axes,
    sliced_axes,
    with_factors,
)
from qewton.equations.model.constraints import Constraint, Relation
from qewton.equations.model.problem import Problem


@dataclass(frozen=True)
class Normalized:
    """The rewritten problem, where each constraint came from, and what was done."""

    problem: Problem
    origin: tuple[Constraint, ...]
    diagnostics: tuple[Diagnostic, ...]


def _time_domains(problem: Problem) -> list[Domain]:
    """The domains that live on the time coordinate alone, such as the interval I."""
    if problem.time is None:
        return []
    time = root_of(problem.time)
    found = (
        node
        for node in with_factors(problem.sets())
        if isinstance(node, Domain) and node.axes == {time}
    )
    return sorted(found, key=lambda d: d.name)


def _needs_lift(constraint: Constraint, time) -> bool:
    held_on = frozenset()
    if constraint.on is not None:
        # A slice fixes the time, so the constraint already holds at one time only.
        held_on = set_free_axes(constraint.on) | sliced_axes(constraint.on)
    used = free_axes(constraint.lhs) | free_axes(constraint.rhs)
    return time in used and time not in held_on


def _lifted_set(on: SetNode | None, interval: Domain) -> SetNode:
    return interval if on is None else Product((on, interval))


def _lift(problem: Problem) -> Normalized:
    """Lift the constraints that hold at every time onto a set with time."""
    if problem.time is None:
        return Normalized(problem, problem.constraints, ())

    time = root_of(problem.time)
    todo = [c for c in problem.constraints if _needs_lift(c, time)]
    if not todo:
        return Normalized(problem, problem.constraints, ())

    intervals = _time_domains(problem)
    if len(intervals) != 1:
        reason = (
            "no interval on the time coordinate is declared"
            if not intervals
            else "several intervals on the time coordinate are declared"
        )
        found = [
            warning(
                f"Cannot lift the constraint to hold at all times: {reason}. "
                "Hold it on a product with the interval, e.g. Omega x I.",
                c,
            )
            for c in todo
        ]
        return Normalized(problem, problem.constraints, tuple(found))

    interval = intervals[0]
    constraints, diagnostics = [], []
    for constraint in problem.constraints:
        if constraint in todo:
            lifted = replace(constraint, on=_lifted_set(constraint.on, interval))
            diagnostics.append(
                info(f"Lifted to hold at all times: now on {lifted.on}.", constraint)
            )
            constraints.append(lifted)
        else:
            constraints.append(constraint)

    rewritten = replace(problem, constraints=tuple(constraints))
    return Normalized(rewritten, problem.constraints, tuple(diagnostics))


def _expand_lap(expr: Expr) -> Expr:
    if isinstance(expr, Apply) and expr.op.name == "lap":
        argument, axes = expr.args
        return Apply(OPS["div"], (Apply(OPS["grad"], (argument, axes)), axes))
    return expr


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Const) and expr.value == 0 and expr.shape == ()


def _residual(constraint: Constraint) -> Constraint:
    if constraint.rel is Relation.COMPLEMENTARITY or _is_zero(constraint.rhs):
        return constraint
    return replace(
        constraint,
        lhs=Apply(OPS["sub"], (constraint.lhs, constraint.rhs)),
        rhs=Const(0),
    )


def _canonical(constraint: Constraint) -> Constraint:
    expanded = replace(
        constraint,
        lhs=transform(constraint.lhs, _expand_lap),
        rhs=transform(constraint.rhs, _expand_lap),
    )
    return _residual(expanded)


def normalize(problem: Problem) -> Normalized:
    """Lift constraints in time, expand the Laplacian and write residuals."""
    lifted = _lift(problem)
    constraints = tuple(_canonical(c) for c in lifted.problem.constraints)
    if constraints == lifted.problem.constraints:
        return lifted
    rewritten = replace(problem, constraints=constraints)
    return Normalized(rewritten, problem.constraints, lifted.diagnostics)
