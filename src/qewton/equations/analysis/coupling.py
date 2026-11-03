"""Structural equation/unknown coupling, matching and block ordering."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from qewton.equations.analysis.infer import infer_shapes
from qewton.equations.analysis.normalize import normalize
from qewton.equations.analysis.shapes import shape_of
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Expr,
    FunctionSymbol,
    Integral,
    Restrict,
    Shift,
    walk,
)
from qewton.equations.expr.ops import ShapeError
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.task import Task

Shape = tuple[int, ...]


@dataclass(frozen=True)
class EquationNode:
    """One scalar component of a constraint, indexed from zero."""

    constraint: int
    component: int

    def __str__(self) -> str:
        return f"c{self.constraint + 1}[{self.component}]"


@dataclass(frozen=True)
class UnknownNode:
    """One scalar component of an unknown function, indexed from zero."""

    symbol: FunctionSymbol
    component: int

    def __str__(self) -> str:
        return f"{self.symbol.variable.name}[{self.component}]"


@dataclass(frozen=True)
class CouplingBlock:
    """A strongly connected block after pairing matched equations and unknowns."""

    equations: tuple[EquationNode, ...]
    unknowns: tuple[UnknownNode, ...]


@dataclass(frozen=True)
class CouplingGraph:
    """Scalar equation/unknown incidence graph and structural analysis.

    Edges mean that an equation structurally contains an unknown. Without component
    indexing in the expression language, a vector equation is conservatively connected
    to every component of each unknown that occurs in it.
    """

    equations: tuple[EquationNode, ...]
    unknowns: tuple[UnknownNode, ...]
    edges: tuple[tuple[EquationNode, UnknownNode], ...]
    matching: tuple[tuple[EquationNode, UnknownNode], ...]
    blocks: tuple[CouplingBlock, ...]
    unmatched_equations: tuple[EquationNode, ...]
    unmatched_unknowns: tuple[UnknownNode, ...]

    @property
    def has_full_matching(self) -> bool:
        """Whether every equation and unknown component can be paired."""
        return len(self.matching) == len(self.equations) == len(self.unknowns)


def _shape(expr: Expr, shapes: dict[FunctionSymbol, Shape]) -> Shape:
    if isinstance(expr, FunctionSymbol):
        return shapes[expr]
    if isinstance(expr, Apply):
        args = tuple(_shape(arg, shapes) for arg in expr.args[: expr.op.arity])
        axes = expr.args[expr.op.arity] if expr.op.takes_axes else None
        return expr.op.shape_rule(args, axes)
    if isinstance(expr, (Restrict, Integral, Shift)):
        if isinstance(expr, Shift):
            _shape(expr.offset, shapes)
        return _shape(expr.expr, shapes)
    return shape_of(expr)


def _constraint_shape(
    constraint: Constraint, shapes: dict[FunctionSymbol, Shape]
) -> Shape:
    left = _shape(constraint.lhs, shapes)
    right = _shape(constraint.rhs, shapes)
    if left == right:
        return left
    if left == () and isinstance(constraint.lhs, Const):
        return right
    if right == () and isinstance(constraint.rhs, Const):
        return left
    raise ShapeError(f"Cannot determine scalar equation components for {constraint}.")


def _component_count(shape: Shape) -> int:
    return prod(shape) if shape else 1


def _symbols(
    constraint: Constraint, unknowns: frozenset[FunctionSymbol]
) -> set[FunctionSymbol]:
    return {
        node
        for side in (constraint.lhs, constraint.rhs)
        for node in walk(side)
        if isinstance(node, FunctionSymbol) and node in unknowns
    }


def _strong_components(
    nodes: tuple[UnknownNode, ...],
    adjacency: dict[UnknownNode, set[UnknownNode]],
    equation_for: dict[UnknownNode, EquationNode],
) -> tuple[CouplingBlock, ...]:
    """Tarjan SCCs on the graph of unknowns induced by matched equation rows."""
    components = _tarjan_components(nodes, adjacency)
    ordered = _topological_components(components, adjacency, nodes)
    order = {node: position for position, node in enumerate(nodes)}
    return tuple(
        CouplingBlock(
            tuple(
                sorted(
                    (equation_for[node] for node in group),
                    key=lambda equation: (
                        equation.constraint,
                        equation.component,
                    ),
                )
            ),
            tuple(sorted(group, key=order.__getitem__)),
        )
        for group in ordered
    )


def _tarjan_components(
    nodes: tuple[UnknownNode, ...],
    adjacency: dict[UnknownNode, set[UnknownNode]],
) -> list[tuple[UnknownNode, ...]]:
    # Tarjan's traversal needs separate index, low-link, stack, active and result state.
    # pylint: disable=too-many-locals
    index = 0
    indices: dict[UnknownNode, int] = {}
    low: dict[UnknownNode, int] = {}
    stack: list[UnknownNode] = []
    active: set[UnknownNode] = set()
    components: list[tuple[UnknownNode, ...]] = []

    def visit(node: UnknownNode) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for neighbor in adjacency[node]:
            if neighbor not in indices:
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
            elif neighbor in active:
                low[node] = min(low[node], indices[neighbor])
        if low[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(component))

    for node in nodes:
        if node not in indices:
            visit(node)
    return components


def _topological_components(
    components: list[tuple[UnknownNode, ...]],
    adjacency: dict[UnknownNode, set[UnknownNode]],
    nodes: tuple[UnknownNode, ...],
) -> list[tuple[UnknownNode, ...]]:
    order = {node: position for position, node in enumerate(nodes)}
    component_edges, indegree = _condensation(components, adjacency)
    ready = sorted(
        (i for i, count in indegree.items() if count == 0),
        key=lambda i: min(order[node] for node in components[i]),
    )
    ordered_components = []
    while ready:
        component_index = ready.pop(0)
        ordered_components.append(components[component_index])
        for neighbor in sorted(
            component_edges[component_index],
            key=lambda i: min(order[node] for node in components[i]),
        ):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                ready.append(neighbor)
                ready.sort(key=lambda i: min(order[node] for node in components[i]))
    return ordered_components


def _condensation(
    components: list[tuple[UnknownNode, ...]],
    adjacency: dict[UnknownNode, set[UnknownNode]],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    component_of = {
        node: group_index
        for group_index, group in enumerate(components)
        for node in group
    }
    component_edges = {i: set() for i in range(len(components))}
    indegree = {i: 0 for i in range(len(components))}
    for source, targets in adjacency.items():
        for target in targets:
            left, right = component_of[source], component_of[target]
            if left != right and right not in component_edges[left]:
                component_edges[left].add(right)
                indegree[right] += 1
    return component_edges, indegree


def _nodes(
    constraints: tuple[Constraint, ...],
    task: Task,
    shapes: dict[FunctionSymbol, Shape],
) -> tuple[tuple[EquationNode, ...], tuple[UnknownNode, ...]]:
    equations = tuple(
        EquationNode(index, component)
        for index, constraint in enumerate(constraints)
        for component in range(_component_count(_constraint_shape(constraint, shapes)))
    )
    unknowns = tuple(
        UnknownNode(symbol, component)
        for symbol in task.unknowns
        for component in range(
            _component_count(symbol.shape if symbol.shape is not None else shapes[symbol])
        )
    )
    return equations, unknowns


def _incidence(
    constraints: tuple[Constraint, ...],
    equations: tuple[EquationNode, ...],
    unknowns: tuple[UnknownNode, ...],
    unknown_set: frozenset[FunctionSymbol],
) -> tuple[
    dict[EquationNode, tuple[UnknownNode, ...]],
    tuple[tuple[EquationNode, UnknownNode], ...],
]:
    incidence: dict[EquationNode, tuple[UnknownNode, ...]] = {}
    edges = []
    for constraint_index, constraint in enumerate(constraints):
        symbols = _symbols(constraint, unknown_set)
        components = tuple(node for node in unknowns if node.symbol in symbols)
        for equation in (
            node for node in equations if node.constraint == constraint_index
        ):
            incidence[equation] = components
            edges.extend((equation, node) for node in components)
    return incidence, tuple(edges)


def _maximum_matching(
    equations: tuple[EquationNode, ...],
    incidence: dict[EquationNode, tuple[UnknownNode, ...]],
) -> tuple[tuple[tuple[EquationNode, UnknownNode], ...], set[UnknownNode]]:
    matched: dict[UnknownNode, EquationNode] = {}

    def augment(equation: EquationNode, seen: set[UnknownNode]) -> bool:
        for unknown in incidence[equation]:
            if unknown in seen:
                continue
            seen.add(unknown)
            previous = matched.get(unknown)
            if previous is None or augment(previous, seen):
                matched[unknown] = equation
                return True
        return False

    for equation in equations:
        augment(equation, set())
    return (
        tuple(
            (equation, unknown)
            for unknown, equation in sorted(
                matched.items(),
                key=lambda item: (item[1].constraint, item[1].component),
            )
        ),
        set(matched),
    )


def _matched_adjacency(
    matching: tuple[tuple[EquationNode, UnknownNode], ...],
    incidence: dict[EquationNode, tuple[UnknownNode, ...]],
    matched_unknowns: set[UnknownNode],
) -> dict[UnknownNode, set[UnknownNode]]:
    adjacency = {unknown: set() for unknown in matched_unknowns}
    for equation, paired in matching:
        for unknown in incidence[equation]:
            if unknown in matched_unknowns:
                adjacency[unknown].add(paired)
    return adjacency


def _blocks(
    unknowns: tuple[UnknownNode, ...],
    matching: tuple[tuple[EquationNode, UnknownNode], ...],
    incidence: dict[EquationNode, tuple[UnknownNode, ...]],
    matched_unknowns: set[UnknownNode],
) -> tuple[CouplingBlock, ...]:
    equation_for = {unknown: equation for equation, unknown in matching}
    adjacency = _matched_adjacency(matching, incidence, matched_unknowns)
    ordered = tuple(node for node in unknowns if node in matched_unknowns)
    return _strong_components(ordered, adjacency, equation_for)


def coupling_graph(task: Task) -> CouplingGraph:
    """Build the scalar bipartite graph, a maximum matching and matched SCC blocks.

    Matching uses a deterministic augmenting-path algorithm. For the block ordering,
    each matched equation is paired with its matched unknown; an incidence to another
    matched unknown becomes a directed edge between the two unknowns. Unmatched rows
    and columns are reported separately and are excluded from blocks.
    """
    problem = normalize(task.problem).problem
    inference = infer_shapes(problem)
    shapes = dict(inference.shapes)
    equations, unknowns = _nodes(problem.constraints, task, shapes)
    incidence, edges = _incidence(
        problem.constraints, equations, unknowns, frozenset(task.unknowns)
    )
    matching, matched_unknowns = _maximum_matching(equations, incidence)
    matched_equations = {equation for equation, _ in matching}
    unmatched_equations = tuple(
        node for node in equations if node not in matched_equations
    )
    unmatched_unknowns = tuple(node for node in unknowns if node not in matched_unknowns)
    return CouplingGraph(
        equations,
        unknowns,
        edges,
        matching,
        _blocks(unknowns, matching, incidence, matched_unknowns),
        unmatched_equations,
        unmatched_unknowns,
    )
