"""Bottom-up shape computation for expressions.

Shapes follow the declared outputs of the function symbols: () is a scalar, (n,) a
vector, (n, m) a matrix. Inferring undeclared shapes comes in a later block.
"""

from functools import singledispatch

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
from qewton.equations.expr.ops import Shape, ShapeError


class UnknownShapeError(ShapeError):
    """Raised when a function symbol has no declared output shape."""


@singledispatch
def shape_of(expr: Expr) -> Shape:
    """The shape of `expr`, raising ShapeError if its operators do not fit."""
    raise TypeError(f"Cannot compute the shape of {type(expr).__name__}.")


@shape_of.register
def _(expr: FunctionSymbol) -> Shape:
    shape = expr.shape
    if shape is None:
        raise UnknownShapeError(f"The output shape of {expr.name!r} is not declared.")
    return shape


@shape_of.register
def _(_expr: Coord) -> Shape:
    return ()


@shape_of.register
def _(expr: Const) -> Shape:
    return expr.shape


@shape_of.register
def _(expr: Normal) -> Shape:
    axes = expr.of.axes
    if len(axes) != 1:
        raise ShapeError(f"A normal needs a single coordinate, got {len(axes)}.")
    (axis,) = axes
    return (axis.dim,)


@shape_of.register
def _(expr: Restrict) -> Shape:
    return shape_of(expr.expr)


@shape_of.register
def _(expr: Integral) -> Shape:
    return shape_of(expr.expr)


@shape_of.register
def _(expr: Shift) -> Shape:
    expected: Shape = () if expr.axis.dim == 1 else (expr.axis.dim,)
    offset = shape_of(expr.offset)
    if offset != expected:
        raise ShapeError(
            f"Shifting along {expr.axis.name!r} needs an offset of shape {expected}, "
            f"got {offset}."
        )
    return shape_of(expr.expr)


@shape_of.register
def _(expr: Apply) -> Shape:
    cached = expr.__dict__.get("_shape")
    if cached is not None:
        return cached
    op = expr.op
    shapes = tuple(shape_of(arg) for arg in expr.args[: op.arity])
    axes = expr.args[op.arity] if op.takes_axes else None
    try:
        shape = op.shape_rule(shapes, axes)
    except ShapeError as error:
        raise ShapeError(f"In {expr}: {error}") from error
    # Cache on the frozen node so building deep trees stays linear.
    object.__setattr__(expr, "_shape", shape)
    return shape


def check_shape(expr: Expr) -> Shape | None:
    """The shape of `expr`, or None while a symbol's shape is still undeclared.

    Raises ShapeError for any mismatch among the shapes that are known.
    """
    try:
        return shape_of(expr)
    except UnknownShapeError:
        return None


def check_sides(lhs: Expr, rhs: Expr) -> None:
    """Both sides of a relation must agree; a scalar constant may stand for any shape."""
    left, right = check_shape(lhs), check_shape(rhs)
    if left is None or right is None or left == right:
        return
    if (isinstance(rhs, Const) and right == ()) or (
        isinstance(lhs, Const) and left == ()
    ):
        return
    raise ShapeError(f"The sides of a constraint have shapes {left} and {right}.")


Expr.checks.append(check_shape)
