"""classify: what kind of problem is this?

    result = classify(task)       # or AnalysisContext(task).classification
    result.pde_types              # e.g. {PdeType.PARABOLIC}
    result.blocks_of(PdeType.PARABOLIC)
    print(result)

The result lists facts for the type of each bulk equation, the relation of each
inequality, the type of each coupled block, the kind of each boundary, initial and
global condition, the condition counts, and a first guess of the function space of
each unknown. Each fact says which rule produced it. Missing or surplus conditions
are warnings in `diagnostics`.

Code that uses the result reads its fields and query methods, never the `rule` ids
of the facts: those are for display.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from qewton.equations.analysis.checks.diagnostics import Diagnostic
from qewton.equations.analysis.classify.conditions import (
    ConditionKind,
    boundary_counts,
    condition_kinds,
    initial_counts,
    interface_counts,
    transport_notes,
)
from qewton.equations.analysis.classify.equations import (
    PdeType,
    block_types,
    equation_types,
    relations,
)
from qewton.equations.analysis.classify.facts import Fact, by_constraint
from qewton.equations.analysis.classify.rows import Row
from qewton.equations.analysis.classify.spaces import FunctionSpace, function_spaces
from qewton.equations.analysis.structure.coupling import CouplingGraph
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.model.problem import Problem


@dataclass(frozen=True)
class Classification:
    """The classification of a task, as facts grouped by what they are about.

    `equations` holds the PDE type of each bulk equation and `relations` the relation
    of each bulk inequality or complementarity. `blocks` holds the type of each
    coupled block, `conditions` the kind of each condition that is not a bulk
    equation, `counts` the given and needed condition counts, and `spaces` a first
    guess of the function space of each unknown.
    """

    equations: tuple[Fact, ...]
    relations: tuple[Fact, ...]
    blocks: tuple[Fact, ...]
    conditions: tuple[Fact, ...]
    counts: tuple[Fact, ...]
    spaces: tuple[Fact, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def pde_types(self) -> frozenset[PdeType]:
        """The types of all coupled blocks."""
        return frozenset(f.value for f in self.blocks)  # type: ignore

    def blocks_of(self, kind: PdeType) -> tuple[Fact, ...]:
        """The facts of the coupled blocks of type `kind`."""
        return tuple(f for f in self.blocks if f.value is kind)

    @property
    def condition_kind_of(self) -> dict[int, ConditionKind]:
        """The kind of each condition, by the index of its normalized constraint."""
        return by_constraint(self.conditions)

    @property
    def space_of(self) -> dict[FunctionSymbol, FunctionSpace]:
        """The first-guess function space of each unknown in a bulk equation."""
        return {f.subject: f.value for f in self.spaces}  # type: ignore

    def __str__(self) -> str:
        sections: Iterable[tuple[str, tuple[Fact, ...]]] = (
            ("equations", self.equations + self.relations),
            ("blocks", self.blocks),
            ("conditions", self.conditions),
            ("counts", self.counts),
            ("function spaces", self.spaces),
        )
        lines = []
        for title, facts in sections:
            lines.append(f"{title}:")
            lines.extend(f"  {fact}" for fact in facts)
        lines.extend(str(d) for d in self.diagnostics)
        return "\n".join(lines)


def build_classification(
    problem: Problem,
    unknowns: Iterable[FunctionSymbol],
    rows: tuple[Row, ...],
    graph: CouplingGraph,
) -> Classification:
    """Classify the equations, conditions and unknowns of a normalized problem.

    `rows` are the rows of `problem` (see `build_rows`) and `graph` its coupling graph.
    """
    equations = equation_types(rows, problem.time)
    boundary_facts, boundary_notes = boundary_counts(rows, problem, graph)
    initial_facts, initial_notes = initial_counts(rows, problem, graph)
    conditions = condition_kinds(rows, problem, frozenset(unknowns))
    interface_facts, interface_notes = interface_counts(
        rows, problem, graph, by_constraint(conditions)
    )
    return Classification(
        equations=equations,
        relations=relations(rows),
        blocks=block_types(rows, graph, by_constraint(equations)),
        conditions=conditions,
        counts=boundary_facts + interface_facts + initial_facts,
        spaces=function_spaces(rows, problem.time),
        diagnostics=boundary_notes
        + interface_notes
        + initial_notes
        + transport_notes(equations, rows, problem),
    )
