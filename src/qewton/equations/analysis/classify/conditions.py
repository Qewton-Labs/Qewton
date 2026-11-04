"""Boundary, initial and global conditions: their kinds, and whether there are enough.

Kinds. A boundary condition is DIRICHLET if its unknown terms only contain values,
NEUMANN if they only contain spatial derivatives, and ROBIN if both occur. A condition
on the second set of a declared `Periodic` is PERIODIC. Conditions on a time slice, or
the history of a delay equation, are INITIAL; conditions without a set are GLOBAL.

Counts. An unknown with spatial order 2 needs as many boundary conditions as it has
components, on every part of the boundary of its domain. A condition on the whole
boundary counts for each part, and a periodic condition counts for both of its parts.
An unknown with time order k needs initial data for the orders 0 .. k - 1. Missing or
surplus conditions are reported as warnings, not errors: a problem may leave them out
on purpose, as the Stokes fixture does.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum

from qewton.config.variables import Variable
from qewton.equations.analysis.checks.diagnostics import Diagnostic, info, warning
from qewton.equations.analysis.classify.equations import PdeType
from qewton.equations.analysis.classify.facts import Fact
from qewton.equations.analysis.classify.rows import (
    Region,
    Row,
    base_domain,
    boundary_of,
    factors,
    interface_of,
    spatial_domain,
)
from qewton.equations.analysis.structure.coupling import CouplingGraph
from qewton.equations.analysis.structure.structure import (
    derivative_orders,
    split_constraint,
)
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.geometry.sets import Boundary, Interface, Part, SetNode
from qewton.equations.model.problem import Problem


class ConditionKind(Enum):
    DIRICHLET = "Dirichlet"
    NEUMANN = "Neumann"
    ROBIN = "Robin"
    PERIODIC = "periodic"
    INTERFACE_VALUE = "interface value"
    INTERFACE_FLUX = "interface flux"
    INITIAL = "initial"
    GLOBAL = "global"


def _is_periodic(row: Row, problem: Problem) -> bool:
    sets = factors(row.constraint.on)
    return any(p.second in sets for p in problem.periodic)


def _boundary_kind(row: Row, unknowns: frozenset, time: Variable | None) -> ConditionKind:
    values = derivatives = False
    for term in split_constraint(row.constraint, unknowns).operator:
        found = derivative_orders(term.expr)  # type: ignore
        order = max(
            (
                o
                for s, orders in found.items()
                if s in unknowns
                for axis, o in orders.items()
                if axis != time
            ),
            default=0,
        )
        derivatives, values = derivatives or order > 0, values or order == 0
    if values and derivatives:
        return ConditionKind.ROBIN
    return ConditionKind.NEUMANN if derivatives else ConditionKind.DIRICHLET


def _interface_kind(
    row: Row, unknowns: frozenset, time: Variable | None
) -> ConditionKind:
    """INTERFACE_VALUE if the unknown terms hold values only, else INTERFACE_FLUX."""
    if _boundary_kind(row, unknowns, time) is ConditionKind.DIRICHLET:
        return ConditionKind.INTERFACE_VALUE
    return ConditionKind.INTERFACE_FLUX


def condition_kinds(
    rows: tuple[Row, ...], problem: Problem, unknowns: frozenset
) -> tuple[Fact, ...]:
    """One fact per constraint that is not a bulk equation."""
    facts = []
    for row in rows:
        if row.region is Region.BULK:
            continue
        if row.region is Region.GLOBAL:
            kind, why = ConditionKind.GLOBAL, "it holds without a set"
        elif row.region is Region.INITIAL:
            kind, why = ConditionKind.INITIAL, "it fixes the unknown at the start time"
        elif row.region is Region.INTERFACE:
            kind = _interface_kind(row, unknowns, problem.time)
            what = "values" if kind is ConditionKind.INTERFACE_VALUE else "derivatives"
            why = f"it relates the {what} of the unknown across the interface"
        elif _is_periodic(row, problem):
            kind, why = ConditionKind.PERIODIC, "its set is declared periodic"
        else:
            kind = _boundary_kind(row, unknowns, problem.time)
            why = f"the unknown terms contain {kind.value} data"
        facts.append(Fact("condition", row.constraint, kind, why, (row.index,)))
    return tuple(facts)


def _unknown_components(graph: CouplingGraph) -> Counter:
    return Counter(u.symbol for u in graph.unknowns)


def _needed(rows: tuple[Row, ...], problem: Problem) -> dict:
    """The unknowns with second spatial order per domain of the bulk equations."""
    needed: dict[SetNode, set[FunctionSymbol]] = {}
    for row in rows:
        if row.region is not Region.BULK:
            continue
        domain = base_domain(spatial_domain(row.constraint.on, problem.time))
        for symbol in row.orders:
            if row.space_order(symbol, problem.time) >= 2:
                needed.setdefault(domain, set()).add(symbol)  # type: ignore
    return needed


def _covers(part: Part | None, on: SetNode | None, problem: Problem) -> bool:
    """Whether a condition on `on` constrains the boundary part `part`."""
    sets = factors(on)
    if part is None or part in sets or any(isinstance(s, Boundary) for s in sets):
        return True
    return any(p.first == part and p.second in sets for p in problem.periodic)


def _count(
    rule: str, subject: object, used: list[Row], need: int
) -> tuple[Fact, Diagnostic | None]:
    """A count fact for the conditions `used`, and a warning if there are not `need`."""
    have = sum(r.components for r in used)
    why = f"{have} given, {need} needed"
    fact = Fact(rule, subject, (have, need), why, tuple(r.index for r in used))
    return fact, (None if have == need else warning(f"On {subject}: {why}.", subject))


def boundary_counts(
    rows: tuple[Row, ...], problem: Problem, graph: CouplingGraph
) -> tuple[tuple[Fact, ...], tuple[Diagnostic, ...]]:
    """Compare the boundary conditions given with the ones the equations need."""
    components = _unknown_components(graph)
    results: list[tuple[Fact, Diagnostic | None]] = []
    notes: list[Diagnostic] = []
    given = [r for r in rows if r.region is Region.BOUNDARY]
    for domain, symbols in _needed(rows, problem).items():
        boundary = Boundary(domain)
        on_boundary = [r for r in given if boundary_of(r.constraint.on) == boundary]
        if not on_boundary:
            notes.append(warning(f"No boundary conditions on {boundary}.", boundary))
            continue
        need = sum(components[s] for s in symbols)
        wholes = [p.parts for p in problem.partitions if p.whole == boundary]
        for part in wholes[0] if wholes else (None,):
            used = [r for r in on_boundary if _covers(part, r.constraint.on, problem)]
            results.append(_count("boundary_count", part or boundary, used, need))
    notes.extend(n for _, n in results if n is not None)
    return tuple(f for f, _ in results), tuple(notes)


def _interface_need(rows: tuple[Row, ...], interface: Interface, problem: Problem):
    """The unknowns with second spatial order in the equations on either subdomain."""
    sides = {interface.first, interface.second}
    return {
        symbol
        for row in rows
        if row.region is Region.BULK
        and spatial_domain(row.constraint.on, problem.time) in sides
        for symbol in row.orders
        if row.space_order(symbol, problem.time) >= 2
    }


def interface_counts(
    rows: tuple[Row, ...],
    problem: Problem,
    graph: CouplingGraph,
    kinds: tuple[Fact, ...],
) -> tuple[tuple[Fact, ...], tuple[Diagnostic, ...]]:
    """Each declared interface needs as many value as flux conditions as there are
    components of its second-order unknowns: the unknown and its flux are continuous."""
    components = _unknown_components(graph)
    by_row = {f.constraints[0]: f.value for f in kinds}
    results = []
    for interface in problem.interfaces:
        need = sum(components[s] for s in _interface_need(rows, interface, problem))
        if not need:
            continue
        on_it = [r for r in rows if interface_of(r.constraint.on) == interface]
        for kind in (ConditionKind.INTERFACE_VALUE, ConditionKind.INTERFACE_FLUX):
            used = [r for r in on_it if by_row.get(r.index) is kind]
            subject = f"{interface}, {kind.value}"
            results.append(_count("interface_count", subject, used, need))
    notes = tuple(n for _, n in results if n is not None)
    return tuple(f for f, _ in results), notes


def _required_time_orders(rows: tuple[Row, ...], time: Variable | None) -> dict:
    """The highest time order of each unknown in the bulk equations."""
    orders: dict[FunctionSymbol, int] = {}
    for row in rows:
        if row.region is Region.BULK:
            for symbol in row.orders:
                orders[symbol] = max(orders.get(symbol, 0), row.time_order(symbol, time))
    return orders


def initial_counts(
    rows: tuple[Row, ...], problem: Problem, graph: CouplingGraph
) -> tuple[tuple[Fact, ...], tuple[Diagnostic, ...]]:
    """Compare the initial data given per time order with the data the equations need."""
    components = _unknown_components(graph)
    results = []
    for symbol, top in _required_time_orders(rows, problem.time).items():
        for order in range(top):
            used = [
                r
                for r in rows
                if r.region is Region.INITIAL
                and symbol in r.orders
                and r.time_order(symbol, problem.time) == order
            ]
            subject = f"{symbol.variable.name}, order {order}"
            results.append(_count("initial_count", subject, used, components[symbol]))
    notes = tuple(n for _, n in results if n is not None)
    return tuple(f for f, _ in results), notes


def transport_notes(
    equations: tuple[Fact, ...], rows: tuple[Row, ...], problem: Problem
) -> tuple[Diagnostic, ...]:
    """Remind that a transport equation needs inflow data, and that it is not checked."""
    if not any(f.value is PdeType.HYPERBOLIC_TRANSPORT for f in equations):
        return ()
    if problem.periodic or any(r.region is Region.BOUNDARY for r in rows):
        return (
            info(
                "Transport equation: boundary data is only needed where the flow enters; "
                "the flow direction is not checked."
            ),
        )
    return (warning("Transport equation without inflow boundary data or periodicity."),)
