"""From a task to its weak form: `weak_form(task)`.

Steps: build one test function per unknown, pair every bulk equation with a test
function, integrate each equation tested with it by parts until every unknown is within
the derivative order of its space, treat the boundary terms with the boundary
conditions, and keep what is not part of the form (essential, initial, global
conditions) next to it. What cannot be compiled yet is reported in `diagnostics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce

from qewton.config.variables import Variable
from qewton.equations.analysis.checks.diagnostics import Diagnostic, error, warning
from qewton.equations.analysis.classify.conditions import ConditionKind, condition_kinds
from qewton.equations.analysis.classify.rows import (
    Region,
    Row,
    rows,
    spatial_domain,
)
from qewton.equations.analysis.classify.spaces import (
    FunctionSpace,
    function_spaces,
    _ALLOWED,
)
from qewton.equations.analysis.prepare.infer import infer_shapes
from qewton.equations.analysis.prepare.normalize import normalize
from qewton.equations.analysis.structure.structure import derivative_orders
from qewton.equations.compile.boundary import BoundaryContext, Condition, boundary_terms
from qewton.equations.compile.by_parts import Cutter, Pieces
from qewton.equations.compile.forms import FormTerm, Measure, Residual, WeakProblem
from qewton.equations.compile.pairing import pair
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.expr.ops import Shape
from qewton.equations.geometry.coordinates import coordinate_factors
from qewton.equations.geometry.sets import Boundary, Domain, Part, SetNode
from qewton.equations.model.constraints import Constraint, Relation
from qewton.equations.model.problem import Problem
from qewton.equations.model.task import Task


def _dim(shape: Shape) -> int | tuple[int, ...]:
    if len(shape) == 1:
        return shape[0]
    return shape if shape else 1


def _test_function(
    unknown: FunctionSymbol, shape: Shape, time: Variable | None
) -> FunctionSymbol:
    """`v_<name>`, with the shape of the unknown and its spatial inputs only."""
    axes = () if unknown.inputs is None else coordinate_factors(unknown.inputs)
    space = [a for a in axes if a != time]
    inputs = reduce(lambda a, b: a * b, space) if space else None
    return FunctionSymbol(Variable(f"v_{unknown.variable.name}", dim=_dim(shape)), inputs)


@dataclass(frozen=True)
class _Setup:
    problem: Problem
    unknowns: tuple[FunctionSymbol, ...]
    tests: dict[FunctionSymbol, FunctionSymbol]
    cutter: Cutter
    conditions: tuple[Condition, ...]


def _conditions(
    table: tuple[Row, ...], kinds: dict[int, ConditionKind], time: Variable | None
) -> tuple[Condition, ...]:
    found = []
    for row in table:
        part = spatial_domain(row.constraint.on, time)
        if row.region is Region.BOUNDARY and part is not None and row.index in kinds:
            found.append(Condition(row, kinds[row.index], part))
    return tuple(found)


def _regions(boundary: Boundary, problem: Problem) -> tuple[SetNode, ...]:
    """The parts of the boundary, or the boundary itself if nothing is declared.

    Without a partition, periodic parts of the boundary are the only parts known.
    """
    for partition in problem.partitions:
        if partition.whole == boundary:
            return tuple(partition.parts)
    periodic = tuple(
        s
        for p in problem.periodic
        for s in (p.first, p.second)
        if isinstance(s, Part) and s.of == boundary
    )
    return periodic or (boundary,)


def _equation(
    row: Row, owner: FunctionSymbol, setup: _Setup
) -> tuple[Residual | None, list[Diagnostic]]:
    domain = spatial_domain(row.constraint.on, setup.problem.time)
    if domain is None:
        return None, [
            error("An equation in time only is not compiled yet.", row.constraint)
        ]
    test = setup.tests[owner]
    pieces = Pieces()
    setup.cutter.cut(row.constraint.lhs, test, 1, pieces)
    measure = Measure(domain)
    terms = [FormTerm(sign, expr, measure) for sign, expr in pieces.interior]
    notes = [warning(text, row.constraint) for text in pieces.notes]
    if pieces.boundary and not isinstance(domain, Domain):
        notes.append(
            error(
                "Integration by parts on a subdomain needs its interface conditions, "
                "which are not compiled yet.",
                row.constraint,
            )
        )
    elif pieces.boundary:
        boundary = Boundary(domain)
        periodic = frozenset(
            s for p in setup.problem.periodic for s in (p.first, p.second)
        )
        regions = _regions(boundary, setup.problem)
        if boundary not in regions and not setup.problem.partitions:
            notes.append(
                warning(
                    "The boundary has no partition: outside the periodic parts its "
                    "boundary terms are dropped, which assumes a natural condition.",
                    boundary,
                )
            )
        context = BoundaryContext(
            boundary,
            regions,
            owner,
            setup.conditions,
            frozenset(setup.unknowns),
            periodic,
        )
        extra, more = boundary_terms(pieces.boundary, context)
        terms += extra
        notes += more
    return Residual(owner, test, row.index, tuple(terms)), notes


def _final_spaces(
    residuals: tuple[Residual, ...],
    first: dict[str, FunctionSpace],
    setup: _Setup,
) -> dict[FunctionSymbol, FunctionSpace]:
    """The space of each unknown that the derivatives left in the form call for."""
    time = setup.problem.time
    orders: dict[FunctionSymbol, int] = {}
    for residual in residuals:
        for term in residual.terms:
            for symbol, found in derivative_orders(term.integrand).items():
                spatial = max((o for a, o in found.items() if a != time), default=0)
                if symbol in setup.tests:
                    orders[symbol] = max(orders.get(symbol, 0), spatial)
    spaces = {}
    for symbol, order in orders.items():
        guess = first.get(str(symbol.variable.name), FunctionSpace.H1)
        if order >= 2:
            spaces[symbol] = FunctionSpace.H2
        elif order == 1:
            keep = guess in (FunctionSpace.H_DIV, FunctionSpace.H_CURL)
            spaces[symbol] = guess if keep else FunctionSpace.H1
        else:
            spaces[symbol] = FunctionSpace.L2
    return spaces


def _others(
    table: tuple[Row, ...], kinds: dict[int, ConditionKind]
) -> tuple[list[Constraint], list[Constraint], list[Diagnostic]]:
    """The essential conditions, the conditions kept as they are, and what is skipped."""
    essential: list[Constraint] = []
    kept: list[Constraint] = []
    notes: list[Diagnostic] = []
    for row in table:
        constraint = row.constraint
        if constraint.rel is not Relation.EQ:
            notes.append(error("Inequalities are not compiled yet.", constraint))
        elif row.region is Region.INTERFACE:
            notes.append(error("Interface conditions are not compiled yet.", constraint))
        elif row.region in (Region.INITIAL, Region.GLOBAL):
            kept.append(constraint)
        elif (
            row.region is Region.BOUNDARY and kinds[row.index] is ConditionKind.DIRICHLET
        ):
            essential.append(constraint)
    return essential, kept, notes


def _setup(task: Task, table: tuple[Row, ...]) -> tuple[_Setup, dict, dict]:
    """The setup, the first-guess spaces by name, and the number of components."""
    problem = normalize(task.problem).problem
    time = problem.time
    unknowns = task.unknowns
    shapes = dict(infer_shapes(problem).shapes)
    tests = {u: _test_function(u, shapes[u], time) for u in unknowns}
    shapes.update({tests[u]: shapes[u] for u in unknowns})
    kinds = {
        f.constraints[0]: f.value
        for f in condition_kinds(table, problem, frozenset(unknowns))
    }
    first = {f.subject: f.value for f in function_spaces(table, time)}
    allowed = {
        u: _ALLOWED[first.get(u.variable.name, FunctionSpace.H1)] for u in unknowns
    }
    cutter = Cutter(allowed, shapes, time)
    setup = _Setup(problem, unknowns, tests, cutter, _conditions(table, kinds, time))
    return setup, first, kinds


def _residuals(
    table: tuple[Row, ...], setup: _Setup
) -> tuple[list[Residual], list[Diagnostic]]:
    equations = [
        r for r in table if r.region is Region.BULK and r.constraint.rel is Relation.EQ
    ]
    shapes = setup.cutter.shapes
    sizes = {u: math.prod(shapes[u]) for u in setup.unknowns}
    partner = pair(equations, setup.unknowns, sizes, setup.problem.time)
    residuals: list[Residual] = []
    notes: list[Diagnostic] = []
    for row in equations:
        if row.index not in partner:
            notes.append(
                error(
                    "No unknown with its number of components to test with.",
                    row.constraint,
                )
            )
            continue
        residual, found = _equation(row, partner[row.index], setup)
        notes += found
        if residual is not None:
            residuals.append(residual)
    return residuals, notes


def weak_form(task: Task) -> WeakProblem:
    """The weak form of the normalized problem of `task`, in space (method of lines)."""
    table = rows(task)
    setup, first, kinds = _setup(task, table)
    residuals, notes = _residuals(table, setup)
    essential, kept, skipped = _others(table, kinds)
    done = tuple(residuals)
    return WeakProblem(
        setup.unknowns,
        setup.tests,
        _final_spaces(done, first, setup),
        done,
        tuple(essential),
        setup.problem.periodic,
        tuple(kept),
        setup.problem.time,
        tuple(notes + skipped),
    )
