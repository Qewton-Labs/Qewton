"""PDE type of every bulk equation, and of every coupled block.

The type is read from the highest derivative orders of the unknowns in an equation:
t for the time axis and s for all other axes. The sign of the leading terms is not
checked, so -u_t - lap u = 0 is called parabolic although it is ill-posed forward in
time. The rules are:

    t = 0:  s = 0 algebraic, s = 1 first-order stationary, s = 2 elliptic
    t > 0:  s = 0 ODE, (t, s) = (1, 2) parabolic, (2, 2) wave, (1, 1) transport

Anything else is UNCLASSIFIED. Without a time variable every axis counts as space.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from qewton.config.variables import Variable
from qewton.equations.analysis.classify.facts import Fact
from qewton.equations.analysis.classify.rows import Region, Row
from qewton.equations.analysis.structure.coupling import CouplingBlock, CouplingGraph
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.model.constraints import Relation


class PdeType(Enum):
    ALGEBRAIC = "algebraic"
    ODE = "ode"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC_WAVE = "hyperbolic (wave)"
    HYPERBOLIC_TRANSPORT = "hyperbolic (transport)"
    FIRST_ORDER_STATIONARY = "first-order stationary"
    SADDLE_POINT = "saddle point"
    UNCLASSIFIED = "unclassified"


_BY_ORDERS = {
    (0, 0): PdeType.ALGEBRAIC,
    (0, 1): PdeType.FIRST_ORDER_STATIONARY,
    (0, 2): PdeType.ELLIPTIC,
    (1, 2): PdeType.PARABOLIC,
    (2, 2): PdeType.HYPERBOLIC_WAVE,
    (1, 1): PdeType.HYPERBOLIC_TRANSPORT,
}

# The type of a block of several equations is the first of these that occurs in it.
_PRIORITY = (
    PdeType.HYPERBOLIC_WAVE,
    PdeType.HYPERBOLIC_TRANSPORT,
    PdeType.PARABOLIC,
    PdeType.ELLIPTIC,
    PdeType.ODE,
    PdeType.FIRST_ORDER_STATIONARY,
    PdeType.ALGEBRAIC,
    PdeType.UNCLASSIFIED,
)


def pde_type(t: int, s: int) -> PdeType:
    """The type for time order `t` and space order `s`."""
    if t > 0 and s == 0:
        return PdeType.ODE
    return _BY_ORDERS.get((t, s), PdeType.UNCLASSIFIED)


def equation_types(rows: tuple[Row, ...], time: Variable | None) -> tuple[Fact, ...]:
    """One fact per bulk constraint: its PDE type."""
    facts = []
    for row in rows:
        if row.region is not Region.BULK:
            continue
        symbols = row.orders
        t = max((row.time_order(s, time) for s in symbols), default=0)
        s = max((row.space_order(s, time) for s in symbols), default=0)
        why = f"highest time order {t}, highest space order {s}"
        facts.append(Fact("pde_type", row.constraint, pde_type(t, s), why, (row.index,)))
    return tuple(facts)


def relations(rows: tuple[Row, ...]) -> tuple[Fact, ...]:
    """One fact per bulk constraint that is an inequality or a complementarity."""
    facts = []
    for row in rows:
        relation = row.constraint.rel
        if row.region is Region.BULK and relation is not Relation.EQ:
            note = f"the relation is '{relation.value}': a variational inequality"
            facts.append(Fact("relation", row.constraint, relation, note, (row.index,)))
    return tuple(facts)


def _symbols_in(row: Row) -> set[FunctionSymbol]:
    """The unknowns of the row's constraint."""
    return {s for s in row.constraint.symbols() if s in row.orders}


def _saddle_multiplier(
    block: CouplingBlock, by_index: dict[int, Row]
) -> FunctionSymbol | None:
    """An unknown missing from some equations, fewer than the others have components.

    Such an unknown acts as a Lagrange multiplier: the equations without it are
    constraints on the other unknowns, e.g. div u = 0 in the Stokes system.
    """
    symbols = {u.symbol for u in block.unknowns}
    if len(symbols) < 2:
        return None
    for q in symbols:
        free = sum(
            1 for e in block.equations if q not in _symbols_in(by_index[e.constraint])
        )
        others = sum(1 for u in block.unknowns if u.symbol != q)
        if 0 < free < others:
            return q
    return None


def block_types(
    rows: tuple[Row, ...], graph: CouplingGraph, kinds: Mapping[int, PdeType]
) -> tuple[Fact, ...]:
    """One fact per coupled block: SADDLE_POINT, else the dominant equation type.

    `kinds` maps the index of each bulk equation to its type.
    """
    by_index = {row.index: row for row in rows}
    result = []
    for block in graph.blocks:
        indices = tuple(sorted({e.constraint for e in block.equations}))
        bulk = [kinds[i] for i in indices if i in kinds]
        if not bulk:
            continue
        names = ", ".join(sorted({str(u.symbol.variable.name) for u in block.unknowns}))
        multiplier = _saddle_multiplier(block, by_index)
        if multiplier is not None:
            why = f"{multiplier.variable.name} only appears in some of the equations"
            result.append(Fact("block_type", names, PdeType.SADDLE_POINT, why, indices))
            continue
        kind = next(k for k in _PRIORITY if k in bulk)
        why = f"dominant type of equations {', '.join(str(i + 1) for i in indices)}"
        result.append(Fact("block_type", names, kind, why, indices))
    return tuple(result)
