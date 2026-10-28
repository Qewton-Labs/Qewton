"""Symbolic PDE problems: geometry, expressions, constraints and printing.

import qewton.equations as eq

x = eq.Variable("x", dim=2)
Omega = eq.Domain("Omega", ambient=x)
"""

from qewton.config.variables import Variable
from qewton.equations.constraints import (
    Complementary,
    Constraint,
    Eq,
    Ge,
    Le,
    PendingConstraint,
    Relation,
)
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
    lap,
    sin,
)
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Coord,
    Expr,
    FunctionSymbol,
    Normal,
    Restrict,
    as_expr,
    children,
    walk,
)
from qewton.equations.expr.ops import OPS, OpSpec, ShapeError, register
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Part,
    Partition,
    Product,
    SetNode,
    Slice,
    is_subset,
    set_from_geometry,
)
from qewton.equations.model import Problem
from qewton.equations.pretty import pretty

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
    "Apply",
    "as_expr",
    "children",
    "walk",
    "OpSpec",
    "OPS",
    "register",
    "ShapeError",
    "grad",
    "div",
    "curl",
    "lap",
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
    "Product",
    "Slice",
    "is_subset",
    "set_from_geometry",
    "Problem",
    "pretty",
]
