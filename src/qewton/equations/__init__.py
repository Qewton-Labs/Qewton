"""Symbolic PDE problems: geometry, expressions, constraints and printing.

import qewton.equations as eq

x = eq.Variable("x", dim=2)
Omega = eq.Domain("Omega", ambient=x)
"""

from qewton.config.variables import Variable
from qewton.equations.model.constraints import (
    Complementary,
    Constraint,
    Eq,
    Ge,
    Le,
    PendingConstraint,
    Relation,
)
from qewton.equations.analysis.normalize import Normalized, normalize
from qewton.equations.analysis.overview import Overview, overview
from qewton.equations.analysis.pipeline import AnalysisContext, report, run_pipeline
from qewton.equations.analysis.signatures import Signature, Signatures, signatures
from qewton.equations.analysis.validate import validate
from qewton.equations.model.task import Role, Task, TaskError
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
    lap,
    mean,
    shift,
    sin,
)
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
    as_expr,
    children,
    transform,
    walk,
)
from qewton.equations.expr.ops import OPS, OpSpec, ShapeError, register
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Part,
    Partition,
    Periodic,
    Product,
    SetNode,
    Slice,
    is_subset,
    set_from_geometry,
)
from qewton.equations.analysis.axes import AxisError, free_axes
from qewton.equations.analysis.diagnostics import Diagnostic, Severity
from qewton.equations.analysis.infer import Inference, infer_shapes
from qewton.equations.model.problem import Problem
from qewton.equations.printing.pretty import pretty

__all__ = [
    "Variable",
    "Relation",
    "Constraint",
    "PendingConstraint",
    "Eq",
    "Le",
    "Ge",
    "Complementary",
    "Expr",
    "FunctionSymbol",
    "Coord",
    "Const",
    "Normal",
    "Restrict",
    "AxisError",
    "Diagnostic",
    "Inference",
    "Integral",
    "Role",
    "Severity",
    "Shift",
    "Signature",
    "Signatures",
    "Task",
    "TaskError",
    "AnalysisContext",
    "Apply",
    "as_expr",
    "children",
    "transform",
    "walk",
    "OpSpec",
    "OPS",
    "Normalized",
    "Overview",
    "register",
    "ShapeError",
    "grad",
    "div",
    "curl",
    "lap",
    "free_axes",
    "infer_shapes",
    "integral",
    "mean",
    "normalize",
    "overview",
    "report",
    "run_pipeline",
    "shift",
    "signatures",
    "validate",
    "d",
    "dt",
    "dot",
    "inner",
    "sin",
    "cos",
    "exp",
    "SetNode",
    "Domain",
    "Boundary",
    "Part",
    "Partition",
    "Periodic",
    "Product",
    "Slice",
    "is_subset",
    "set_from_geometry",
    "Problem",
    "pretty",
]
