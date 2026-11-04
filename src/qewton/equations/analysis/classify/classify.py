"""classify: what kind of problem is this?

    result = classify(task)
    result.pde_types      # e.g. {PdeType.PARABOLIC}
    print(result)

The result lists facts for the type of each bulk equation, the type of each coupled
block, the kind of each boundary, initial and global condition, the condition counts,
and a first guess of the function space of each unknown. Each fact says which rule
produced it. Missing or surplus conditions are warnings in `diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.equations.analysis.checks.diagnostics import Diagnostic
from qewton.equations.analysis.classify.conditions import (
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
)
from qewton.equations.analysis.classify.facts import Fact
from qewton.equations.analysis.classify.rows import rows
from qewton.equations.analysis.classify.spaces import function_spaces
from qewton.equations.analysis.prepare.normalize import normalize
from qewton.equations.analysis.structure.coupling import coupling_graph
from qewton.equations.model.task import Task


@dataclass(frozen=True)
class Classification:
    equations: tuple[Fact, ...]
    blocks: tuple[Fact, ...]
    conditions: tuple[Fact, ...]
    counts: tuple[Fact, ...]
    spaces: tuple[Fact, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def pde_types(self) -> frozenset[PdeType]:
        """The types of all coupled blocks."""
        return frozenset(f.value for f in self.blocks)  # type: ignore

    def __str__(self) -> str:
        sections = (
            ("equations", self.equations),
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


def classify(task: Task) -> Classification:
    """Classify the equations, conditions and unknowns of a task."""
    problem = normalize(task.problem).problem
    graph = coupling_graph(task)
    table = rows(task)
    equations = equation_types(table, problem.time)
    boundary_facts, boundary_notes = boundary_counts(table, problem, graph)
    initial_facts, initial_notes = initial_counts(table, problem, graph)
    conditions = condition_kinds(table, problem, frozenset(task.unknowns))
    interface_facts, interface_notes = interface_counts(table, problem, graph, conditions)
    return Classification(
        equations=equations,
        blocks=block_types(table, graph, equations),
        conditions=conditions,
        counts=boundary_facts + interface_facts + initial_facts,
        spaces=function_spaces(table, problem.time),
        diagnostics=boundary_notes
        + interface_notes
        + initial_notes
        + transport_notes(equations, table, problem),
    )
