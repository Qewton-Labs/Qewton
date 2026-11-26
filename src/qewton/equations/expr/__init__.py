"""Expressions: functions, coordinates, constants and operators.

- `base.py`: `Expr`, the base class. Each node class defines its own structure,
  shape, coordinates and linearity there; walking and rebuilding trees
  (`walk`, `transform`) are methods of it too.
- `leaves.py`: functions, coordinates, constants, normals.
- `wrappers.py`: restriction, integral, shift and component of an expression.
- `apply.py`: an operator applied to its arguments; `ops.py` has the operators,
  one class per kind with its shape rules, order and linearity, and their registry.
- `api.py`: helpers that build expressions, such as `grad(u, x)`.

Building an operator node checks its shape and coordinates at once.
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
from qewton.equations.expr.apply import Apply
from qewton.equations.expr.base import Expr
from qewton.equations.expr.leaves import Const, Coord, FunctionSymbol, Normal, as_expr
from qewton.equations.expr.ops import (
    OPS,
    OperatorRegistry,
    AxisError,
    Linearity,
    OpLinearity,
    Operator,
    ShapeError,
)
from qewton.equations.expr.wrappers import Index, Integral, Restrict, Shift

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
    # operators
    "OPS",
    "OperatorRegistry",
    "Linearity",
    "OpLinearity",
    "Operator",
    "ShapeError",
    "AxisError",
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
