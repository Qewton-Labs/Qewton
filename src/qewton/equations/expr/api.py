"""User-facing helpers that build expressions, e.g. grad(u, x) or dt(u, t).

Each helper of an operator calls it from the registry, e.g. `OPS.grad(u, x)`, so the
operator's rules apply. Python numbers and one-dimensional coordinates are accepted
as expression arguments.
Derivative axes are passed as they are and must be coordinates or products of them.
"""

from __future__ import annotations

from qewton.config.variables import Variable
from qewton.equations.geometry.sets import Interface, SetNode
from qewton.equations.expr.base import Expr
from qewton.equations.expr.leaves import as_expr
from qewton.equations.expr.wrappers import Integral, Restrict, Shift
from qewton.equations.expr.ops import OPS


def sin(value: object) -> Expr:
    return OPS.sin(value)


def cos(value: object) -> Expr:
    return OPS.cos(value)


def exp(value: object) -> Expr:
    return OPS.exp(value)


def dot(first: object, second: object) -> Expr:
    return OPS.dot(first, second)


def inner(first: object, second: object) -> Expr:
    return OPS.inner(first, second)


def d(value: object, axis: Variable) -> Expr:
    """The partial derivative along one coordinate, e.g. d(u, t)."""
    return OPS.d(value, axis)


def dt(value: object, time: Variable) -> Expr:
    """The time derivative. Same as d(value, time)."""
    return d(value, time)


def grad(value: object, axes: Variable) -> Expr:
    return OPS.grad(value, axes)


def div(value: object, axes: Variable) -> Expr:
    return OPS.div(value, axes)


def curl(value: object, axes: Variable) -> Expr:
    return OPS.curl(value, axes)


def lap(value: object, axes: Variable) -> Expr:
    return OPS.lap(value, axes)


def integral(value: object, over: SetNode) -> Expr:
    return Integral(as_expr(value), over)


def mean(value: object, over: SetNode) -> Expr:
    """The average of `value` over `over`: the integral divided by the integral of 1."""
    return integral(value, over) / integral(1, over)


def trace(value: object, interface: Interface, side: int) -> Expr:
    """The trace of `value` on an interface from one side: 0 is `first`, 1 is `second`."""
    if side not in (0, 1):
        raise ValueError(f"The side of an interface is 0 or 1, got {side!r}.")
    subdomain = (interface.first, interface.second)[side]
    return Restrict(Restrict(as_expr(value), subdomain), interface)


def jump(value: object, interface: Interface) -> Expr:
    """The jump of `value` across an interface: its trace from `second` minus `first`."""
    return trace(value, interface, 1) - trace(value, interface, 0)


def shift(value: object, axis: Variable, offset: object) -> Expr:
    """`value` evaluated at `axis + offset`, e.g. shift(u, t, -1) is u(t - 1)."""
    return Shift(as_expr(value), axis, as_expr(offset))
