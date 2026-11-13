"""Expressions: functions, coordinates, constants and operators.

`nodes.py` has the expression tree, `ops.py` the operator registry and `api.py` the
helpers that build expressions, such as `grad(u, x)`.

Building an operator node checks it at once: its shape (`shapes.py`) and the
coordinates it depends on (`axes.py`). Both checks are written against the node
classes, so `nodes.py` cannot import them; this package registers them instead.
Any import from the package runs this file first, so the checks are always on.
"""

from qewton.equations.expr.api import (
    cos,
    curl,
    d,
    div,
    dot,
    dt,
    exp,
    grad,
    inner,
    integral,
    jump,
    lap,
    mean,
    shift,
    sin,
    trace,
)
from qewton.equations.expr.axes import AxisError, check_axes, free_axes
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
    as_expr,
    transform,
    walk,
)
from qewton.equations.expr.ops import (
    OPS,
    Linearity,
    OpLinearity,
    OpSpec,
    ShapeError,
    register,
)
from qewton.equations.expr.shapes import check_shape

Expr.checks.extend((check_shape, check_axes))

__all__ = [
    # nodes
    "Apply",
    "Const",
    "Coord",
    "Expr",
    "FunctionSymbol",
    "Index",
    "Integral",
    "Normal",
    "Restrict",
    "Shift",
    "as_expr",
    "transform",
    "walk",
    # operators
    "OPS",
    "Linearity",
    "OpLinearity",
    "OpSpec",
    "ShapeError",
    "register",
    # axes
    "AxisError",
    "free_axes",
    # building expressions
    "cos",
    "curl",
    "d",
    "div",
    "dot",
    "dt",
    "exp",
    "grad",
    "inner",
    "integral",
    "jump",
    "lap",
    "mean",
    "shift",
    "sin",
    "trace",
]
