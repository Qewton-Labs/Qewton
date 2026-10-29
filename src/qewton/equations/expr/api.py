"""User-facing helpers that build expressions, e.g. grad(u, x) or dt(u, t).

Each helper is a thin wrapper around Apply, so the registry rules apply. Python
numbers and one-dimensional coordinates are accepted as expression arguments.
Derivative axes are passed as they are and must be coordinates or products of them.
"""

from __future__ import annotations

from qewton.config.variables import Variable
from qewton.equations.geometry.sets import SetNode
from qewton.equations.expr.nodes import Apply, Expr, Integral, Shift, as_expr
from qewton.equations.expr.ops import OPS


def _call(name: str, *args: object) -> Expr:
    return Apply(OPS[name], tuple(as_expr(arg) for arg in args))


def _call_on_axes(name: str, value: object, axes: Variable) -> Expr:
    return Apply(OPS[name], (as_expr(value), axes))


def sin(value: object) -> Expr:
    return _call("sin", value)


def cos(value: object) -> Expr:
    return _call("cos", value)


def exp(value: object) -> Expr:
    return _call("exp", value)


def dot(first: object, second: object) -> Expr:
    return _call("dot", first, second)


def inner(first: object, second: object) -> Expr:
    return _call("inner", first, second)


def d(value: object, axis: Variable) -> Expr:
    """The partial derivative along one coordinate, e.g. d(u, t)."""
    return _call_on_axes("d", value, axis)


def dt(value: object, time: Variable) -> Expr:
    """The time derivative. Same as d(value, time)."""
    return d(value, time)


def grad(value: object, axes: Variable) -> Expr:
    return _call_on_axes("grad", value, axes)


def div(value: object, axes: Variable) -> Expr:
    return _call_on_axes("div", value, axes)


def curl(value: object, axes: Variable) -> Expr:
    return _call_on_axes("curl", value, axes)


def lap(value: object, axes: Variable) -> Expr:
    return _call_on_axes("lap", value, axes)


def integral(value: object, over: SetNode) -> Expr:
    return Integral(as_expr(value), over)


def mean(value: object, over: SetNode) -> Expr:
    """The average of `value` over `over`: the integral divided by the integral of 1."""
    return integral(value, over) / integral(1, over)


def shift(value: object, axis: Variable, offset: object) -> Expr:
    """`value` evaluated at `axis + offset`, e.g. shift(u, t, -1) is u(t - 1)."""
    return Shift(as_expr(value), axis, as_expr(offset))
