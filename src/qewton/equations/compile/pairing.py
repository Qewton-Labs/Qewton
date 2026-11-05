"""Pairing each bulk equation with the unknown whose test function it is tested with.

An equation with k scalar components is tested with the test function of an unknown
with k components. The choice is a modelling decision and is made by a heuristic:
an unknown that appears in the equation is preferred, the one with the highest spatial
derivative order first. An equation that does not contain a matching unknown at all
may still test a free one: the continuity equation `div u = 0` tests with the
pressure, which does not appear in it. A bipartite matching makes the choices
consistent, so that Stokes pairs momentum with `v_u` and continuity with `v_p`.

Equations on different subdomains of one domain that contain the same unknowns are one
equation with a piecewise definition, and share their partner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qewton.config.variables import Variable
from qewton.equations.analysis.classify.rows import Row, base_domain, spatial_domain
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.geometry.sets import Part


def _candidates(
    row: Row,
    unknowns: Sequence[FunctionSymbol],
    sizes: Mapping[FunctionSymbol, int],
    time: Variable | None,
) -> list[FunctionSymbol]:
    fitting = [u for u in unknowns if sizes[u] == row.components]
    present = [u for u in fitting if u in row.orders]
    present.sort(key=lambda u: -row.space_order(u, time))
    return present + [u for u in fitting if u not in row.orders]


def _augment(
    row: int,
    seen: set[FunctionSymbol],
    owner: dict[FunctionSymbol, int],
    candidates: Mapping[int, list[FunctionSymbol]],
) -> bool:
    """Give `row` an unknown, moving other rows to their next choice if needed."""
    for unknown in candidates[row]:
        if unknown in seen:
            continue
        seen.add(unknown)
        if unknown not in owner or _augment(owner[unknown], seen, owner, candidates):
            owner[unknown] = row
            return True
    return False


def _group_key(row: Row, time: Variable | None) -> object:
    domain = spatial_domain(row.constraint.on, time)
    if isinstance(domain, Part):
        return (base_domain(domain), frozenset(row.orders), row.components)
    return row.index


def pair(
    rows: Sequence[Row],
    unknowns: Sequence[FunctionSymbol],
    sizes: Mapping[FunctionSymbol, int],
    time: Variable | None,
) -> dict[int, FunctionSymbol]:
    """The unknown of each row, by row index. Rows without a partner are left out."""
    leaders: dict[object, Row] = {}
    for row in rows:
        leaders.setdefault(_group_key(row, time), row)
    candidates = {
        r.index: _candidates(r, unknowns, sizes, time) for r in leaders.values()
    }
    owner: dict[FunctionSymbol, int] = {}
    for leader in leaders.values():
        _augment(leader.index, set(), owner, candidates)
    partner = {index: unknown for unknown, index in owner.items()}
    return {
        row.index: partner[leaders[_group_key(row, time)].index]
        for row in rows
        if leaders[_group_key(row, time)].index in partner
    }
