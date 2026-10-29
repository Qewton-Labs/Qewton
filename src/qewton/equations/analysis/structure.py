"""structure: what the constraints look like, read off the expression trees.

Block 1: derivative orders. For a constraint and a function, the highest derivative
order of the function along each coordinate, e.g. {x: 2, t: 1} for the heat equation.
Block 2: the operator/source split and linearity. Each constraint is cut into signed
terms at its top-level sums. A term that contains an unknown belongs to the operator
part, the others to the source part. Every term is labelled by how it depends on the
unknowns. Later blocks add batch placement and the coupling graph.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from qewton.config.variables import Variable
from qewton.equations.analysis.normalize import normalize
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Expr,
    FunctionSymbol,
    Integral,
    Restrict,
    Shift,
    children,
    walk,
)
from qewton.equations.geometry.coordinates import coordinate_factors
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.task import Task

Orders = dict[Variable, int]


def derivative_orders(expr: Expr) -> dict[FunctionSymbol, Orders]:
    """The highest derivative order of every function in `expr`, per coordinate.

    Orders add up along the path from the root to the function: lap(u, x) is order 2
    in x, and d(d(u, x), t) is order 1 in x and 1 in t. A function without derivatives
    maps to {}. Different occurrences keep their highest order, so d(u, x) + u * u is
    order 1 in x. A shift (a delay) is not a derivative and adds nothing.
    """
    found: dict[FunctionSymbol, Orders] = {}
    _collect(expr, {}, found)
    return found


def _collect(expr: Expr, above: Orders, found: dict[FunctionSymbol, Orders]) -> None:
    if isinstance(expr, FunctionSymbol):
        seen = found.setdefault(expr, {})
        for axis, order in above.items():
            seen[axis] = max(seen.get(axis, 0), order)
        return
    inner = above
    if isinstance(expr, Apply) and expr.op.takes_axes:
        inner = dict(above)
        for axis in coordinate_factors(expr.args[expr.op.arity]):
            inner[axis] = inner.get(axis, 0) + expr.op.order
    for child in children(expr):
        _collect(child, inner, found)


def constraint_orders(constraint: Constraint) -> dict[FunctionSymbol, Orders]:
    """`derivative_orders` over both sides of a constraint."""
    found = derivative_orders(constraint.lhs)
    for symbol, orders in derivative_orders(constraint.rhs).items():
        seen = found.setdefault(symbol, {})
        for axis, order in orders.items():
            seen[axis] = max(seen.get(axis, 0), order)
    return found


def unknown_orders(task: Task) -> tuple[dict[FunctionSymbol, Orders], ...]:
    """For each constraint of the task's problem, the orders of the unknowns in it.

    Constraints without an unknown give {}. Functions that are not unknowns are left
    out.
    """
    unknowns = set(task.unknowns)
    return tuple(
        {s: o for s, o in constraint_orders(c).items() if s in unknowns}
        for c in task.problem.constraints
    )


Linearity = Literal["none", "linear", "affine", "nonlinear"]


def _combine(first: Linearity, second: Linearity) -> Linearity:
    """The dependence of a sum of two expressions."""
    if "nonlinear" in (first, second):
        return "nonlinear"
    if first == "none":
        return second
    if second == "none":
        return first
    return "linear" if first == second == "linear" else "affine"


def linearity(expr: Expr, unknowns: frozenset[FunctionSymbol]) -> Linearity:
    """How `expr` depends on the unknowns.

    - "none": no unknown occurs.
    - "linear": a linear function of the unknowns, e.g. nu * grad(u).
    - "affine": linear plus something that does not depend on them, e.g. u + 1.
    - "nonlinear": anything else, e.g. u * u or sin(u).

    This follows `OpSpec.linearity`: a multilinear operator stays linear when only one
    argument contains unknowns, and is nonlinear when both do. Restrictions, integrals
    and shifts of an expression keep its dependence, but a shift offset that contains
    an unknown makes the term nonlinear.
    """
    if isinstance(expr, FunctionSymbol):
        return "linear" if expr in unknowns else "none"
    if isinstance(expr, (Shift, Restrict, Integral)):
        if isinstance(expr, Shift) and linearity(expr.offset, unknowns) != "none":
            return "nonlinear"
        return linearity(expr.expr, unknowns)
    if isinstance(expr, Apply):
        return _apply_linearity(expr, unknowns)
    return "none"


def _apply_linearity(expr: Apply, unknowns: frozenset[FunctionSymbol]) -> Linearity:
    parts = [linearity(arg, unknowns) for arg in children(expr)]
    involved = [part for part in parts if part != "none"]
    kind = expr.op.linearity
    if not involved:
        return "none"
    if kind == "nonlinear" or (kind == "multilinear" and len(involved) > 1):
        return "nonlinear"
    if kind == "multilinear":
        return involved[0]
    result: Linearity = "none"
    for part in parts:
        result = _combine(result, part)
    # A term without unknowns next to one with is a constant offset.
    return "affine" if "none" in parts and result == "linear" else result


@dataclass(frozen=True)
class Term:
    """One signed term of a constraint, with how it depends on the unknowns."""

    expr: Expr
    sign: int
    linearity: Linearity

    def __str__(self) -> str:
        return f"{'+' if self.sign > 0 else '-'} {self.expr}  [{self.linearity}]"


@dataclass(frozen=True)
class Split:
    """A constraint cut into an operator part and a source part.

    `operator` holds the terms that contain an unknown, `source` the others.
    `linearity` is "none" without unknowns, "nonlinear" if any term is, and "linear"
    otherwise (in the sense of a linear PDE: a linear operator plus a source).
    """

    operator: tuple[Term, ...]
    source: tuple[Term, ...]

    @property
    def linearity(self) -> Linearity:
        if not self.operator:
            return "none"
        if any(term.linearity == "nonlinear" for term in self.operator):
            return "nonlinear"
        return "linear"


def _terms(expr: Expr, sign: int = 1) -> Iterator[tuple[Expr, int]]:
    """Cut `expr` at its top-level sums, keeping track of the sign."""
    if isinstance(expr, Apply) and expr.op.name == "neg":
        yield from _terms(expr.args[0], -sign)
    elif isinstance(expr, Apply) and expr.op.name in ("add", "sub"):
        yield from _terms(expr.args[0], sign)
        second = sign if expr.op.name == "add" else -sign
        yield from _terms(expr.args[1], second)
    else:
        yield expr, sign


def split_constraint(
    constraint: Constraint, unknowns: frozenset[FunctionSymbol]
) -> Split:
    """The terms of `lhs` and (with opposite sign) `rhs`, split by the unknowns.

    A term that is a zero constant is dropped. The sign of a right-hand term is
    flipped, so the terms read as `sum of terms rel 0`.
    """
    pieces = list(_terms(constraint.lhs)) + list(_terms(constraint.rhs, -1))
    operator, source = [], []
    for expr, sign in pieces:
        if isinstance(expr, Const) and expr.value == 0:
            continue
        term = Term(expr, sign, linearity(expr, unknowns))
        has_unknown = any(node in unknowns for node in walk(expr))
        (operator if has_unknown else source).append(term)
    return Split(tuple(operator), tuple(source))


def split(task: Task) -> tuple[Split, ...]:
    """The operator/source split of every constraint of the normalized problem."""
    unknowns = frozenset(task.unknowns)
    return tuple(
        split_constraint(c, unknowns) for c in normalize(task.problem).problem.constraints
    )
