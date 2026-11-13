"""Expressions: functions, coordinates, constants and operators.

Building an operator node checks it at once: its shape (`shapes.py`) and the
coordinates it depends on (`axes.py`). Both checks are written against the node
classes, so `nodes.py` cannot import them; this package registers them instead.
Any import from the package runs this file first, so the checks are always on.
"""

from qewton.equations.expr.axes import check_axes
from qewton.equations.expr.nodes import Expr
from qewton.equations.expr.shapes import check_shape

Expr.checks.extend((check_shape, check_axes))
