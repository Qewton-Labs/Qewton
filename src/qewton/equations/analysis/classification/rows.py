"""The constraints of a task, one row each, with the data every classifier needs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from math import prod

from qewton.config.variables import Variable
from qewton.equations.analysis.prepare.infer import constraint_shape
from qewton.equations.analysis.structure.structure import Orders, constraint_orders
from qewton.equations.expr.nodes import Expr, FunctionSymbol, Restrict
from qewton.equations.geometry.sets import (
    Interface,
    SetNode,
    Slice,
    factors,
    lies_on_boundary,
)
from qewton.equations.expr.ops import Shape
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.problem import Problem


class Region(Enum):
    """Where in the problem a constraint acts."""

    BULK = "bulk"
    BOUNDARY = "boundary"
    INTERFACE = "interface"
    INITIAL = "initial"
    GLOBAL = "global"


def _only_restricted(expr: Expr, on: SetNode, unknowns: frozenset) -> bool:
    """Whether every unknown in `expr` sits inside a restriction to `on`."""
    if isinstance(expr, Restrict) and expr.on == on:
        return True
    if isinstance(expr, FunctionSymbol):
        return expr not in unknowns
    return all(_only_restricted(c, on, unknowns) for c in expr.children())


def _set_region(parts: tuple[SetNode, ...]) -> Region | None:
    """The region a set shows by its factors alone, in the order they take priority."""
    tests = (
        (Region.INITIAL, lambda p: isinstance(p, Slice)),
        (Region.INTERFACE, lambda p: isinstance(p, Interface)),
        (Region.BOUNDARY, lies_on_boundary),
    )
    return next((r for r, test in tests if any(test(p) for p in parts)), None)


def _region(constraint: Constraint, time: Variable | None, unknowns: frozenset) -> Region:
    on = constraint.on
    if on is None:
        return Region.GLOBAL
    shown = _set_region(factors(on))
    if shown is not None:
        return shown
    # The history of a delay equation: a time interval where u itself is prescribed.
    history = time is not None and on.axes == frozenset({time})
    if history and all(_only_restricted(s, on, unknowns) for s in constraint.sides):
        return Region.INITIAL
    return Region.BULK


@dataclass(frozen=True)
class Row:
    """A normalized constraint with its region, unknown orders and component count."""

    index: int
    constraint: Constraint
    region: Region
    orders: dict[FunctionSymbol, Orders]
    components: int

    def space_order(self, symbol: FunctionSymbol, time: Variable | None) -> int:
        """The highest derivative order of `symbol` along the non-time axes."""
        found = self.orders.get(symbol, {})
        return max((o for axis, o in found.items() if axis != time), default=0)

    def time_order(self, symbol: FunctionSymbol, time: Variable | None) -> int:
        return self.orders.get(symbol, {}).get(time, 0) if time is not None else 0


def build_rows(
    problem: Problem,
    unknowns: Iterable[FunctionSymbol],
    shapes: Mapping[FunctionSymbol, Shape],
) -> tuple[Row, ...]:
    """One row per constraint of the normalized `problem`.

    `shapes` holds the output shape of every function; the component count of a row
    is the number of scalar equations its constraint stands for.
    """
    unknowns = frozenset(unknowns)
    result = []
    for index, constraint in enumerate(problem.constraints):
        orders = {s: o for s, o in constraint_orders(constraint).items() if s in unknowns}
        count = prod(constraint_shape(constraint, shapes))
        region = _region(constraint, problem.time, unknowns)
        result.append(Row(index, constraint, region, orders, count))
    return tuple(result)
