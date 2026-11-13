"""structure: what the constraints look like, read off the expression trees.

Block 1: derivative orders. For a constraint and a function, the highest derivative
order of the function along each coordinate, e.g. {x: 2, t: 1} for the heat equation.
Block 2: the operator/source split and linearity. Each constraint is cut into signed
terms at its top-level sums. A term that contains an unknown belongs to the operator
part, the others to the source part. Every term is labelled by how it depends on the
unknowns. Batch placement and decoupled coordinate axes are included here; the scalar
coupling graph is in `structure/coupling.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import singledispatch
from typing import Literal

from qewton.config.variables import Variable
from qewton.equations.expr.axes import free_axes
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Expr,
    FunctionSymbol,
    Index,
    Integral,
    Restrict,
    Shift,
    contains,
    symbols_in,
    walk,
)
from qewton.equations.geometry.coordinates import coordinate_factors, root_of
from qewton.equations.expr.ops import Linearity
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.problem import Problem
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
        for axis in coordinate_factors(expr.args[expr.op.arity]):  # type: ignore
            inner[axis] = inner.get(axis, 0) + expr.op.order
    for child in expr.children():
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


def build_decoupled_axes(
    problem: Problem, unknowns: Iterable[FunctionSymbol]
) -> tuple[Variable, ...]:
    """Axes along which unknowns have no differential or nonlocal coupling.

    These are candidate independent directions: the equations of the (normalized)
    `problem` can be solved pointwise along such coordinates. An integral over an
    axis or a shift along it couples different points even without a derivative, so
    those axes are excluded.
    """
    unknowns = frozenset(unknowns)
    if not unknowns:
        return ()
    axes = {
        root_of(axis)
        for symbol in unknowns
        if symbol.inputs is not None
        for axis in coordinate_factors(symbol.inputs)
    }
    differentiated: set[Variable] = set()
    nonlocal_axes: set[Variable] = set()
    for constraint in problem.constraints:
        for side in constraint.sides:
            differentiated.update(_differentiated_axes(side, unknowns))
            nonlocal_axes.update(_nonlocal_axes(side, unknowns))
    return tuple(sorted(axes - differentiated - nonlocal_axes, key=lambda a: str(a.name)))


def _differentiated_axes(
    expr: Expr, unknowns: frozenset[FunctionSymbol]
) -> set[Variable]:
    return {
        root_of(axis)
        for node in walk(expr)
        if isinstance(node, Apply)
        and node.op.takes_axes
        and contains(node.args[0], unknowns)  # type: ignore
        for axis in coordinate_factors(node.args[node.op.arity])  # type: ignore
    }


def _nonlocal_axes(expr: Expr, unknowns: frozenset[FunctionSymbol]) -> set[Variable]:
    coupled = set()
    for node in walk(expr):
        if isinstance(node, Integral):
            symbols = {s for s in symbols_in(node.expr) if s in unknowns}
            for symbol in symbols:
                coupled.update(
                    root_of(axis) for axis in free_axes(symbol) & node.over.axes
                )
        elif isinstance(node, Shift) and contains(node.expr, unknowns):
            coupled.add(root_of(node.axis))
    return coupled


@singledispatch
def linearity(_expr: Expr, _unknowns: frozenset[FunctionSymbol]) -> Linearity:
    """How `expr` depends on the unknowns.

    This follows `OpSpec.linearity`: a multilinear operator stays linear when only one
    argument contains unknowns, and is nonlinear when both do. Restrictions, integrals
    and shifts of an expression keep its dependence, but a shift offset that contains
    an unknown makes the term nonlinear.
    """
    return Linearity.NONE


@linearity.register
def _(expr: FunctionSymbol, unknowns: frozenset[FunctionSymbol]) -> Linearity:
    return Linearity.LINEAR if expr in unknowns else Linearity.NONE


@linearity.register(Restrict)
@linearity.register(Integral)
@linearity.register(Index)
def _(
    expr: Restrict | Integral | Index, unknowns: frozenset[FunctionSymbol]
) -> Linearity:
    return linearity(expr.expr, unknowns)


@linearity.register
def _(expr: Shift, unknowns: frozenset[FunctionSymbol]) -> Linearity:
    if linearity(expr.offset, unknowns) is not Linearity.NONE:
        return Linearity.NONLINEAR
    return linearity(expr.expr, unknowns)


@linearity.register
def _(expr: Apply, unknowns: frozenset[FunctionSymbol]) -> Linearity:
    parts = [linearity(arg, unknowns) for arg in expr.children()]
    return expr.op.linearity.of_result(parts)


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
    `linearity` is NONE without unknowns, NONLINEAR if any term is, and LINEAR
    otherwise (in the sense of a linear PDE: a linear operator plus a source).
    """

    operator: tuple[Term, ...]
    source: tuple[Term, ...]

    @property
    def linearity(self) -> Linearity:
        if not self.operator:
            return Linearity.NONE
        if any(term.linearity is Linearity.NONLINEAR for term in self.operator):
            return Linearity.NONLINEAR
        return Linearity.LINEAR


def signed_terms(
    expr: Expr | Variable, sign: int = 1
) -> Iterator[tuple[Expr | Variable, int]]:
    """Cut `expr` at its top-level sums, keeping track of the sign."""
    if isinstance(expr, Apply) and expr.op.name == "neg":
        yield from signed_terms(expr.args[0], -sign)
    elif isinstance(expr, Apply) and expr.op.name in ("add", "sub"):
        yield from signed_terms(expr.args[0], sign)
        second = sign if expr.op.name == "add" else -sign
        yield from signed_terms(expr.args[1], second)
    else:
        yield expr, sign


def split_constraint(
    constraint: Constraint, unknowns: frozenset[FunctionSymbol]
) -> Split:
    """The terms of `lhs` and (with opposite sign) `rhs`, split by the unknowns.

    A term that is a zero constant is dropped. The sign of a right-hand term is
    flipped, so the terms read as `sum of terms rel 0`.
    """
    pieces = list(signed_terms(constraint.lhs)) + list(signed_terms(constraint.rhs, -1))
    operator, source = [], []
    for expr, sign in pieces:
        if isinstance(expr, Const) and expr.value == 0:
            continue
        term = Term(expr, sign, linearity(expr, unknowns))  # type: ignore
        (operator if contains(expr, unknowns) else source).append(term)  # type: ignore
    return Split(tuple(operator), tuple(source))


@dataclass(frozen=True)
class BatchPlacement:
    """Where one varying function enters one normalized constraint."""

    constraint_index: int
    symbol: FunctionSymbol
    part: Literal["operator", "source"]
    affine: bool


def _differentiates(expr: Expr, symbol: FunctionSymbol) -> bool:
    """Whether a derivative acts on a varying symbol along one of its input axes."""
    if symbol.inputs is None:
        return False
    symbol_axes = set(coordinate_factors(symbol.inputs))
    for node in walk(expr):
        if not isinstance(node, Apply) or not node.op.takes_axes:
            continue
        argument = node.args[0]
        axes = node.args[node.op.arity]
        if contains(argument, (symbol,)) and symbol_axes.intersection(  # type: ignore
            coordinate_factors(axes)  # type: ignore
        ):
            return True
    return False


def _affine_in(expr: Expr, symbol: FunctionSymbol) -> bool:
    """Whether `expr` separates as the varying symbol times a term independent of it."""
    return linearity(
        expr, frozenset({symbol})
    ) is not Linearity.NONLINEAR and not _differentiates(expr, symbol)


def build_batch_placement(
    splits: Sequence[Split], varying: Sequence[FunctionSymbol]
) -> tuple[BatchPlacement, ...]:
    """Locate the `varying` functions in the operator/source split of each constraint.

    `splits` holds one split per normalized constraint. One result is returned per
    (constraint, varying function, part). A symbol can occur in both parts or in
    several terms of one part. In that case `affine` is true only if all its terms in
    that part separate from the rest of the expression.

    Varying sets are not listed: they change the constraint's geometry rather than
    appearing as expression symbols. A varying function differentiated along one of
    its declared input axes is not separable. Thus `div(nu * grad(u, x), x)` is affine
    in `nu` when `nu` is spatially constant, but not when `nu` depends on `x`.
    """
    results = []
    for index, constraint_split in enumerate(splits):
        for symbol in varying:
            for name, terms in (
                ("operator", constraint_split.operator),
                ("source", constraint_split.source),
            ):
                uses = [term.expr for term in terms if contains(term.expr, (symbol,))]
                if uses:
                    results.append(
                        BatchPlacement(
                            index,
                            symbol,
                            name,  # type: ignore
                            all(_affine_in(expr, symbol) for expr in uses),
                        )
                    )
    return tuple(results)
