"""A first guess of the function space of each unknown.

Take a bulk equation whose highest spatial order over its unknowns is m. Its weak form
moves r = m // 2 derivatives onto the test function, so an unknown differentiated k
times needs max(0, k - r) derivatives in the space:

    0: L2.
    1: H(div) or H(curl) if every spatial operator applied to it is a div or a curl
       (then the full gradient is not needed), otherwise H1.
    2 or more: H2.

The strongest space over all equations wins, in the order L2 < H(div), H(curl) < H1 <
H2. Time derivatives are ignored. This is a guess from the strong form: the weak form
(M8) decides what is really used. Stokes gives u in H1 and p in L2, and Poisson, heat
and wave give H1.
"""

from __future__ import annotations

from enum import Enum

from qewton.config.variables import Variable
from qewton.equations.analysis.classification.facts import Fact
from qewton.equations.analysis.classification.rows import Region, Row
from qewton.equations.expr.nodes import Apply, Expr, FunctionSymbol
from qewton.equations.geometry.coordinates import coordinate_factors


class FunctionSpace(Enum):
    L2 = "L2"
    H_DIV = "H(div)"
    H_CURL = "H(curl)"
    H1 = "H1"
    H2 = "H2"


_RANK = {
    FunctionSpace.L2: 0,
    FunctionSpace.H_DIV: 1,
    FunctionSpace.H_CURL: 1,
    FunctionSpace.H1: 2,
    FunctionSpace.H2: 3,
}

# The highest spatial derivative order of a function in each space.
_HIGHEST_ORDER = {
    FunctionSpace.L2: 0,
    FunctionSpace.H_DIV: 1,
    FunctionSpace.H_CURL: 1,
    FunctionSpace.H1: 1,
    FunctionSpace.H2: 2,
}

_BY_OPERATOR = {"div": FunctionSpace.H_DIV, "curl": FunctionSpace.H_CURL}


def highest_order(space: FunctionSpace) -> int:
    """The highest spatial derivative order a function in `space` has: 1 for H1."""
    return _HIGHEST_ORDER[space]


def _operators(expr: Expr, symbol: FunctionSymbol, time: Variable | None) -> set[str]:
    """The spatial derivative operators applied directly to `symbol` in `expr`."""
    found: set[str] = set()
    for child in expr.children():
        if isinstance(expr, Apply) and expr.op.takes_axes and child == symbol:
            axes = coordinate_factors(expr.args[expr.op.arity])  # type: ignore
            if any(axis != time for axis in axes):
                found.add(expr.op.name)
        found |= _operators(child, symbol, time)
    return found


def _stronger(first: FunctionSpace, second: FunctionSpace) -> FunctionSpace:
    if {first, second} == {FunctionSpace.H_DIV, FunctionSpace.H_CURL}:
        return FunctionSpace.H1
    return first if _RANK[first] >= _RANK[second] else second


def _needed(
    row: Row, symbol: FunctionSymbol, time: Variable | None, r: int
) -> FunctionSpace:
    order = max(0, row.space_order(symbol, time) - r)
    if order == 0:
        return FunctionSpace.L2
    if order >= 2:
        return FunctionSpace.H2
    names = set()
    for side in row.constraint.sides:
        names |= _operators(side, symbol, time)
    if len(names) == 1:
        return _BY_OPERATOR.get(names.pop(), FunctionSpace.H1)
    return FunctionSpace.H1


def function_spaces(rows: tuple[Row, ...], time: Variable | None) -> tuple[Fact, ...]:
    """One fact per unknown that occurs in a bulk equation."""
    spaces: dict[FunctionSymbol, FunctionSpace] = {}
    where: dict[FunctionSymbol, list[int]] = {}
    for row in rows:
        if row.region is not Region.BULK:
            continue
        top = max((row.space_order(s, time) for s in row.orders), default=0)
        for symbol in row.orders:
            need = _needed(row, symbol, time, top // 2)
            spaces[symbol] = _stronger(spaces.get(symbol, FunctionSpace.L2), need)
            where.setdefault(symbol, []).append(row.index)
    return tuple(
        Fact(
            "function_space",
            symbol,
            space,
            "derivatives left after moving half the equation order to the test function",
            tuple(where[symbol]),
        )
        for symbol, space in spaces.items()
    )
