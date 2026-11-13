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
certainly not intended), and shifting along one. `check_axes` runs these checks;
`expr/__init__.py` registers it with the nodes.
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
    Index,
    Integral,
    Normal,
    Restrict,
    Shift,
)
from qewton.equations.geometry.coordinates import coordinate_factors, root_of
from qewton.equations.geometry.sets import set_free_axes, sliced_axes

Axes = frozenset[Variable]


class AxisError(ValueError):
    """Raised when an expression or constraint uses coordinates inconsistently."""


def _names(axes: Axes) -> str:
    return "{" + ", ".join(sorted(str(axis.name) for axis in axes)) + "}"


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
def _(expr: Index) -> Axes:
    return free_axes(expr.expr)


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
        expr_variables = expr.args[expr.op.arity]
        assert isinstance(
            expr_variables, Variable
        ), f"Expected a Variable, got {type(expr_variables)}"
        missing = frozenset(coordinate_factors(expr_variables)) - result
        if missing:
            raise AxisError(
                f"Cannot differentiate along {_names(missing)} in {expr}: the operand "
                f"depends on {_names(result)}, so the derivative is zero."
            )
    object.__setattr__(expr, "_axes", result)
    return result


def check_axes(expr: Expr) -> None:
    free_axes(expr)
