"""Which coordinates an expression depends on.

`free_axes(expr)` is the set of coordinates, e.g. {x, t}, that the value of `expr`
varies with. It follows the declared inputs of the functions:

- a function depends on its declared inputs, a constant on none;
- a coordinate depends on itself;
- restricting to a time slice removes that coordinate, since it is now fixed;
- integrating over a set removes the coordinates of that set;
- a shift or derivative keeps the coordinates of its operand.

Two mistakes are caught when an expression is built: differentiating along a
coordinate the operand does not depend on (the result is zero, which is almost
certainly not intended), and shifting along one.
"""

from __future__ import annotations

from functools import singledispatch

from qewton.config.variables import Variable
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Coord,
    Expr,
    FunctionSymbol,
    Integral,
    Normal,
    Restrict,
    Shift,
)
from qewton.equations.geometry.coordinates import coordinate_factors, root_of
from qewton.equations.model.constraints import Constraint
from qewton.equations.geometry.sets import (
    Boundary,
    Part,
    Product,
    SetNode,
    Slice,
)

Axes = frozenset[Variable]


class AxisError(ValueError):
    """Raised when an expression or constraint uses coordinates inconsistently."""


def _names(axes: Axes) -> str:
    return "{" + ", ".join(sorted(axis.name for axis in axes)) + "}"


def sliced_axes(s: SetNode) -> Axes:
    """The coordinates a set fixes to a single value, such as t in Omega x {t = 0}."""
    if isinstance(s, Slice):
        return frozenset({s.axis})
    if isinstance(s, Product):
        return frozenset().union(*(sliced_axes(factor) for factor in s.factors))
    if isinstance(s, (Boundary, Part)):
        return sliced_axes(s.of)
    return frozenset()


def set_free_axes(s: SetNode) -> Axes:
    """The coordinates a point of the set still varies in: its axes minus the sliced."""
    return s.axes - sliced_axes(s)


@singledispatch
def free_axes(expr: Expr) -> Axes:
    raise TypeError(f"Cannot find the axes of {type(expr).__name__}.")


@free_axes.register
def _(expr: FunctionSymbol) -> Axes:
    if expr.inputs is None:
        return frozenset()
    return frozenset(coordinate_factors(expr.inputs))


@free_axes.register
def _(expr: Coord) -> Axes:
    return frozenset({root_of(expr.variable)})


@free_axes.register
def _(_expr: Const) -> Axes:
    return frozenset()


@free_axes.register
def _(expr: Normal) -> Axes:
    return set_free_axes(expr.of)


@free_axes.register
def _(expr: Restrict) -> Axes:
    return free_axes(expr.expr) - sliced_axes(expr.on)


@free_axes.register
def _(expr: Integral) -> Axes:
    return free_axes(expr.expr) - expr.over.axes


@free_axes.register
def _(expr: Shift) -> Axes:
    operand = free_axes(expr.expr)
    axis = root_of(expr.axis)
    if axis not in operand:
        raise AxisError(
            f"Cannot shift along {axis.name!r}: {expr.expr} depends on "
            f"{_names(operand)}."
        )
    return operand | free_axes(expr.offset)


@free_axes.register
def _(expr: Apply) -> Axes:
    cached = expr.__dict__.get("_axes")
    if cached is not None:
        return cached
    operands = expr.args[: expr.op.arity]
    result: Axes = frozenset().union(*(free_axes(arg) for arg in operands))
    if expr.op.takes_axes:
        missing = frozenset(coordinate_factors(expr.args[expr.op.arity])) - result
        if missing:
            raise AxisError(
                f"Cannot differentiate along {_names(missing)} in {expr}: the operand "
                f"depends on {_names(result)}, so the derivative is zero."
            )
    object.__setattr__(expr, "_axes", result)
    return result


def check_axes(expr: Expr) -> None:
    free_axes(expr)


def check_constraint_axes(constraint: Constraint, time: Variable | None) -> None:
    """Both sides may depend only on the coordinates the constraint holds on.

    The time coordinate is the exception: a constraint on a set without time that
    uses time-dependent functions is lifted to hold at every time, so it is allowed.
    A global constraint holds on no coordinate.
    """
    on = constraint.on
    allowed = frozenset() if on is None else set_free_axes(on)
    if time is not None:
        allowed |= {root_of(time)}
    used = free_axes(constraint.lhs) | free_axes(constraint.rhs)
    extra = used - allowed
    if extra:
        where = "globally" if on is None else f"on {on}"
        hint = (
            " Integrate or restrict it first."
            if on is None
            else " Restrict it, integrate it, or hold the constraint on a larger set."
        )
        raise AxisError(
            f"The constraint {constraint} holds {where} but depends on "
            f"{_names(extra)}.{hint}"
        )


Expr.checks.append(check_axes)
