"""Batch-axis propagation from varying data and geometry to unknowns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qewton.config.variables import Variable
from qewton.equations.expr.nodes import FunctionSymbol, walk
from qewton.equations.geometry.sets import Product, SetNode
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.problem import Problem

BatchAxes = Mapping[FunctionSymbol | SetNode, Variable]


def _set_depends_on(node: SetNode, source: SetNode, problem: Problem) -> bool:
    if node is source:
        return True
    if isinstance(node, Product):
        if any(_set_depends_on(factor, source, problem) for factor in node.factors):
            return True
    elif hasattr(node, "of"):
        if _set_depends_on(node.of, source, problem):  # type: ignore
            return True
    return any(
        node in partition.parts and source in partition.parts
        for partition in problem.partitions
    )


def _function_symbols(constraint: Constraint) -> set[FunctionSymbol]:
    return {
        node
        for side in (constraint.lhs, constraint.rhs)
        for node in walk(side)
        if isinstance(node, FunctionSymbol)
    }


def _evaluation_sets(constraint: Constraint) -> tuple[SetNode, ...]:
    found = [constraint.on] if constraint.on is not None else []
    for side in (constraint.lhs, constraint.rhs):
        for node in walk(side):
            found.extend(node.sets())
    return tuple(found)


def _constraint_seeds(
    problem: Problem,
    constraint: Constraint,
    unknowns: frozenset[FunctionSymbol],
    varying_functions: dict[FunctionSymbol, Variable],
    varying_sets: dict[SetNode, Variable],
) -> tuple[set[FunctionSymbol], dict[Variable, set[FunctionSymbol]]]:
    symbols = _function_symbols(constraint)
    involved = symbols & unknowns
    seeds: dict[Variable, set[FunctionSymbol]] = {}
    for symbol in symbols & varying_functions.keys():
        seeds.setdefault(varying_functions[symbol], set()).update(involved)
    for evaluated_on in _evaluation_sets(constraint):
        for varying_set, axis in varying_sets.items():
            if _set_depends_on(evaluated_on, varying_set, problem):
                seeds.setdefault(axis, set()).update(involved)
    return involved, seeds


def _connected(
    starts: set[FunctionSymbol],
    adjacency: dict[FunctionSymbol, set[FunctionSymbol]],
) -> set[FunctionSymbol]:
    visited = set(starts)
    pending = list(starts)
    while pending:
        symbol = pending.pop()
        for neighbor in adjacency[symbol] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    return visited


def _influence(
    problem: Problem,
    unknowns: frozenset[FunctionSymbol],
    batch_axes: BatchAxes,
) -> tuple[
    dict[FunctionSymbol, set[FunctionSymbol]], dict[Variable, set[FunctionSymbol]]
]:
    varying_functions = {
        symbol: axis
        for symbol, axis in batch_axes.items()
        if isinstance(symbol, FunctionSymbol)
    }
    varying_sets = {
        node: axis for node, axis in batch_axes.items() if isinstance(node, SetNode)
    }
    adjacency = {symbol: set() for symbol in unknowns}
    seeds: dict[Variable, set[FunctionSymbol]] = {}
    for constraint in problem.constraints:
        involved, constraint_axes = _constraint_seeds(
            problem, constraint, unknowns, varying_functions, varying_sets
        )
        for symbol in involved:
            adjacency[symbol].update(involved - {symbol})
        for axis, sources in constraint_axes.items():
            seeds.setdefault(axis, set()).update(sources)
    return adjacency, seeds


def build_inherited_batch_axes(
    problem: Problem, unknowns: Sequence[FunctionSymbol], batch_axes: BatchAxes
) -> dict[FunctionSymbol, tuple[Variable, ...]]:
    """Axes inherited by unknowns coupled to varying functions or sets.

    `problem` is the normalized problem and `batch_axes` the parameter axis of each
    varying function or set (`Task.batch_axes`).

    Unknowns share a batch axis if they occur in the same constraint. Varying
    functions seed the constraints in which they occur. A varying set seeds
    constraints whose evaluation set is derived from it, including sibling parts of
    a declared partition. Seeds then propagate through the unknown/constraint graph.

    Returns only inherited axes; a varying function keeps its own axis in
    ``task.batch_axes``. Axes are ordered by name for reproducible reports.
    """
    if not unknowns:
        return {}
    unknown_set = frozenset(unknowns)
    adjacency, seeds = _influence(problem, unknown_set, batch_axes)
    inherited: dict[FunctionSymbol, set[Variable]] = {
        symbol: set() for symbol in unknowns
    }
    for axis, starts in seeds.items():
        for symbol in _connected(starts, adjacency):
            inherited[symbol].add(axis)
    return {
        symbol: tuple(sorted(axes, key=lambda axis: str(axis.name)))
        for symbol, axes in inherited.items()
        if axes
    }
