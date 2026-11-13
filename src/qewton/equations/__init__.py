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
from qewton.equations.analysis.prepare.normalize import Normalized, normalize
from qewton.equations.analysis.classify.classify import Classification
from qewton.equations.analysis.classify.conditions import ConditionKind
from qewton.equations.analysis.classify.equations import PdeType
from qewton.equations.analysis.classify.facts import Fact
from qewton.equations.analysis.classify.spaces import FunctionSpace
from qewton.equations.analysis.overview import Overview, overview
from qewton.equations.analysis.suggest import Category, Rule, Suggestion, suggest
from qewton.equations.analysis.context import (
    AnalysisContext,
    batch_placement,
    batching,
    classify,
    coupling_graph,
    decoupled_axes,
    inherited_batch_axes,
    signatures,
    split,
    training_modes,
)
from qewton.equations.analysis.prepare.signatures import Signature, Signatures
from qewton.equations.analysis.structure.batching import (
    BatchEntry,
    BatchMode,
    Batching,
)
from qewton.equations.analysis.structure.structure import (
    BatchPlacement,
    Split,
    Term,
    constraint_orders,
    derivative_orders,
    linearity,
    unknown_orders,
)
from qewton.equations.analysis.prepare.validate import validate
from qewton.equations.analysis.training import TrainingMode
from qewton.equations.compile.forms import FormTerm, Measure, Residual, WeakProblem
from qewton.equations.compile.weak_form import weak_form
from qewton.equations.model.observation import Observed
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
    jump,
    shift,
    trace,
    sin,
)
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
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Interface,
    Part,
    Partition,
    Periodic,
    Product,
    SetNode,
    Slice,
    is_subset,
    set_from_geometry,
)
from qewton.equations.expr.axes import AxisError, free_axes
from qewton.equations.diagnostics import Diagnostic, Severity
from qewton.equations.analysis.structure.coupling import (
    CouplingBlock,
    CouplingGraph,
    EquationNode,
    UnknownNode,
)
from qewton.equations.printing.coupling_diagram import (
    CouplingDiagram,
    coupling_diagram,
    plot_coupling,
)
from qewton.equations.analysis.prepare.infer import Inference, infer_shapes
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
    "Index",
    "Integral",
    "Role",
    "Observed",
    "TrainingMode",
    "training_modes",
    "FormTerm",
    "Measure",
    "Residual",
    "WeakProblem",
    "weak_form",
    "Severity",
    "Shift",
    "Signature",
    "Signatures",
    "BatchPlacement",
    "CouplingBlock",
    "CouplingGraph",
    "EquationNode",
    "Linearity",
    "OpLinearity",
    "Split",
    "Term",
    "UnknownNode",
    "Task",
    "TaskError",
    "AnalysisContext",
    "Apply",
    "as_expr",
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
    "BatchEntry",
    "BatchMode",
    "Batching",
    "Category",
    "Rule",
    "Suggestion",
    "suggest",
    "batch_placement",
    "batching",
    "coupling_graph",
    "coupling_diagram",
    "CouplingDiagram",
    "plot_coupling",
    "Classification",
    "ConditionKind",
    "Fact",
    "FunctionSpace",
    "PdeType",
    "classify",
    "constraint_orders",
    "decoupled_axes",
    "derivative_orders",
    "linearity",
    "overview",
    "shift",
    "signatures",
    "split",
    "unknown_orders",
    "validate",
    "d",
    "dt",
    "dot",
    "inner",
    "inherited_batch_axes",
    "sin",
    "cos",
    "exp",
    "SetNode",
    "Domain",
    "Boundary",
    "Interface",
    "jump",
    "trace",
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
