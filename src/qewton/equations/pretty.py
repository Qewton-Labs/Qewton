"""Readable text for expressions, sets, constraints and problems.

    pretty(-api.div(nu * api.grad(u, x), x))  gives  "−∇_x·(nu·∇_x u)"

Operators are printed by precedence, so parentheses appear only where they are needed.
"""

from __future__ import annotations

from functools import singledispatch

from qewton.config.variables import Variable
from qewton.equations.constraints import Constraint, PendingConstraint, Relation
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Coord,
    Expr,
    FunctionSymbol,
    Normal,
    Restrict,
)
from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Part,
    Partition,
    Product,
    SetNode,
    Slice,
)
from qewton.equations.model import Problem
from qewton.equations.printable import Printable

# Precedence levels, loosest binding first.
ADD, MUL, NEG, POW, ATOM = 1, 2, 3, 4, 5

_RELATIONS = {
    Relation.EQ: "=",
    Relation.LE: "≤",
    Relation.GE: "≥",
    Relation.COMPLEMENTARITY: "⊥",
}
# Derivative operators written before their operand, as (symbol, separator).
_DERIVATIVES = {
    "grad": ("∇", " "),
    "div": ("∇", "·"),
    "curl": ("∇", "×"),
    "lap": ("Δ", " "),
}
_FUNCTIONS = {"sin", "cos", "exp", "dot", "inner"}
_PRECEDENCE = {
    "add": ADD,
    "sub": ADD,
    "mul": MUL,
    "pow": POW,
    "neg": NEG,
    **dict.fromkeys(_DERIVATIVES, NEG),
}
_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _number(value: float) -> str:
    return f"{value:g}".replace("-", "−")


def _const(value: float | tuple) -> str:
    if not isinstance(value, tuple):
        return _number(value)
    if value and isinstance(value[0], tuple):
        return "[" + ", ".join(_const(item) for item in value) + "]"
    return "(" + ", ".join(_const(item) for item in value) + ")"


def _is_negative(value: object) -> bool:
    return isinstance(value, (int, float)) and value < 0


def _prec(expr: Expr) -> int:
    if isinstance(expr, Const) and _is_negative(expr.value):
        return NEG
    if not isinstance(expr, Apply):
        return ATOM
    return _PRECEDENCE.get(expr.op.name, ATOM)


def _operand(expr: Expr, minimum: int) -> str:
    """Print `expr`, in parentheses if it binds looser than `minimum`."""
    text = _expr(expr)
    return f"({text})" if _prec(expr) < minimum else text


def _component(coord: Coord) -> str:
    var = coord.variable
    if var.dim == 1:
        return var.name
    return var.name + str(coord.component + 1).translate(_SUBSCRIPT)


def _sum(args: tuple) -> str:
    return f"{_operand(args[0], ADD)} + {_operand(args[1], ADD + 1)}"


def _difference(args: tuple) -> str:
    return f"{_operand(args[0], ADD)} − {_operand(args[1], ADD + 1)}"


def _negation(args: tuple) -> str:
    return f"−{_operand(args[0], NEG)}"


def _is_reciprocal(expr: Expr) -> bool:
    return isinstance(expr, Apply) and expr.op.name == "pow" and expr.args[1] == Const(-1)


def _product(args: tuple) -> str:
    first, second = args
    if _is_reciprocal(second):
        return f"{_operand(first, MUL)} / {_operand(second.args[0], MUL + 1)}"
    return f"{_operand(first, MUL)}·{_operand(second, MUL + 1)}"


def _power(args: tuple) -> str:
    base, exponent = args
    base_text = _operand(base, ATOM)
    if isinstance(exponent, Const) and isinstance(exponent.value, int):
        return base_text + str(exponent.value).translate(_SUPERSCRIPT)
    return f"{base_text}^{_operand(exponent, ATOM)}"


def _partial(args: tuple) -> str:
    operand, axis = args
    return f"∂{_operand(operand, ATOM)}/∂{axis.name}"


_OPERATORS = {
    "add": _sum,
    "sub": _difference,
    "neg": _negation,
    "mul": _product,
    "pow": _power,
    "d": _partial,
}


def _derivative(name: str, args: tuple) -> str:
    symbol, separator = _DERIVATIVES[name]
    operand, axes = args
    return f"{symbol}_{axes.name}{separator}{_operand(operand, ATOM)}"


def _apply(expr: Apply) -> str:
    name, args = expr.op.name, expr.args
    if name in _OPERATORS:
        return _OPERATORS[name](args)
    if name in _DERIVATIVES:
        return _derivative(name, args)
    return f"{name}({', '.join(_expr(arg) for arg in args)})"


def _leaf(expr: Expr) -> str:
    if isinstance(expr, FunctionSymbol):
        return expr.name
    if isinstance(expr, Coord):
        return _component(expr)
    if isinstance(expr, Const):
        return _const(expr.value)
    if isinstance(expr, Normal):
        return f"n_{{{_set(expr.of)}}}"
    raise TypeError(f"Nothing to print for {type(expr).__name__}.")


def _expr(expr: Expr) -> str:
    if isinstance(expr, Apply):
        return _apply(expr)
    if isinstance(expr, Restrict):
        return f"{_operand(expr.expr, ATOM)}|_{{{_set(expr.on)}}}"
    return _leaf(expr)


def _factor(factor: SetNode) -> str:
    return f"({_set(factor)})" if isinstance(factor, Product) else _set(factor)


def _set(s: SetNode) -> str:
    if isinstance(s, (Domain, Part)):
        return s.name
    if isinstance(s, Boundary):
        inner = _set(s.of)
        return f"∂{inner}" if isinstance(s.of, (Domain, Part)) else f"∂({inner})"
    if isinstance(s, Product):
        return " × ".join(_factor(factor) for factor in s.factors)
    if isinstance(s, Slice):
        return f"{{{s.axis.name} = {_number(s.value)}}}"
    raise TypeError(f"Nothing to print for {type(s).__name__}.")


def _constraint(constraint: Constraint) -> str:
    label = f"{constraint.name}: " if constraint.name else ""
    return (
        f"{label}{_expr(constraint.lhs)} {_RELATIONS[constraint.rel]} "
        f"{_expr(constraint.rhs)} on {_set(constraint.on)}"
    )


def _pending(pending: PendingConstraint) -> str:
    return (
        f"{_expr(pending.lhs)} {_RELATIONS[pending.rel]} {_expr(pending.rhs)}"
        "  (no set yet)"
    )


def _partition(partition: Partition) -> str:
    parts = " ⊔ ".join(_set(part) for part in partition.parts)
    return f"{_set(partition.whole)} = {parts}"


def _problem(problem: Problem) -> str:
    lines = ["constraints:"]
    lines += [
        f"  {number}. {_constraint(constraint)}"
        for number, constraint in enumerate(problem.constraints, start=1)
    ]
    if problem.partitions:
        lines.append("partitions:")
        lines += [f"  {_partition(partition)}" for partition in problem.partitions]
    if problem.time is not None:
        lines.append(f"time: {problem.time.name}")
    return "\n".join(lines)


def _variable(var: Variable) -> str:
    return var.name


@singledispatch
def pretty(obj: object) -> str:
    """Readable text for an expression, set, constraint, partition or problem."""
    raise TypeError(f"Nothing to print for {type(obj).__name__}.")


pretty.register(Expr, _expr)
pretty.register(SetNode, _set)
pretty.register(Constraint, _constraint)
pretty.register(PendingConstraint, _pending)
pretty.register(Partition, _partition)
pretty.register(Problem, _problem)
pretty.register(Variable, _variable)
Printable.printer = staticmethod(pretty)
