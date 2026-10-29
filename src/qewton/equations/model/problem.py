"""The PDE problem: constraints, boundary partitions and the time variable.

A Problem is mathematics only. It assigns no roles (unknown, fixed, varying); that
is the Task's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.equations.analysis.axes import check_constraint_axes, free_axes
from qewton.equations.model.constraints import Constraint
from qewton.equations.expr.nodes import (
    Apply,
    Coord,
    Expr,
    FunctionSymbol,
    Integral,
    Normal,
    Restrict,
    Shift,
    walk,
)
from qewton.equations.geometry.coordinates import (
    check_coordinate,
    coordinate_factors,
    root_of,
)
from qewton.equations.geometry.sets import Partition, Periodic, SetNode
from qewton.equations.model.task import Task, build_task
from qewton.equations.printing.printable import Printable


@dataclass(frozen=True)
class Problem(Printable):
    constraints: tuple[Constraint, ...]
    partitions: tuple[Partition, ...] = ()
    periodic: tuple[Periodic, ...] = ()
    time: Variable | None = None

    def __post_init__(self):
        # Frozen dataclass: store the sequences as tuples, so the problem hashes.
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "partitions", tuple(self.partitions))
        object.__setattr__(self, "periodic", tuple(self.periodic))
        if not self.constraints:
            raise ValueError("A problem needs at least one constraint.")
        for constraint in self.constraints:
            if not isinstance(constraint, Constraint):
                raise TypeError(
                    f"Expected a Constraint, got {type(constraint).__name__}. "
                    "Attach the set with .on(...), e.g. Eq(lhs, rhs).on(Omega)."
                )
        for partition in self.partitions:
            if not isinstance(partition, Partition):
                raise TypeError(f"Expected a Partition, got {type(partition).__name__}.")
        for declaration in self.periodic:
            if not isinstance(declaration, Periodic):
                raise TypeError(f"Expected a Periodic, got {type(declaration).__name__}.")
        if self.time is not None:
            check_coordinate(self.time)
            if self.time.dim != 1:
                raise ValueError(
                    f"The time variable must be one-dimensional, got dim {self.time.dim}."
                )
        for constraint in self.constraints:
            check_constraint_axes(constraint, self.time)

    def task(self, unknown=(), fixed=None, varying=None) -> Task:
        """Assign roles to the functions of this problem; see `build_task`.

        problem.task(unknown=[u], fixed={f: 1.0, Omega: mesh}, varying={nu: samples})
        """
        return build_task(self, unknown, fixed, varying)

    def overview(self):
        """The coordinates, sets and functions of this problem; print it to read it."""
        # Imported here: the analysis package imports the model, not the other way.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from qewton.equations.analysis.overview import overview

        return overview(self)

    def _sides(self) -> list[Expr]:
        return [
            side
            for constraint in self.constraints
            for side in (constraint.lhs, constraint.rhs)
        ]

    def symbols(self) -> frozenset[FunctionSymbol]:
        """Every function that appears in a constraint."""
        return frozenset(
            node
            for side in self._sides()
            for node in walk(side)
            if isinstance(node, FunctionSymbol)
        )

    def sets(self) -> frozenset[SetNode]:
        """Every set the problem refers to: where constraints hold, where expressions are
        restricted or normals taken, and the boundary partition."""
        found = {c.on for c in self.constraints if c.on is not None}
        for side in self._sides():
            for node in walk(side):
                if isinstance(node, Restrict):
                    found.add(node.on)
                elif isinstance(node, Integral):
                    found.add(node.over)
                elif isinstance(node, Normal):
                    found.add(node.of)
        for partition in self.partitions:
            found.add(partition.whole)
            found.update(partition.parts)
        for declaration in self.periodic:
            found.update((declaration.first, declaration.second))
        return frozenset(found)

    def coordinates(self) -> frozenset[Variable]:
        """Every coordinate the problem uses: those of its sets, the inputs of its
        functions, coordinate components, derivative and shift axes, and time."""
        found: set[Variable] = {axis for s in self.sets() for axis in s.axes}
        for side in self._sides():
            for node in walk(side):
                if isinstance(node, FunctionSymbol):
                    found |= free_axes(node)
                elif isinstance(node, Coord):
                    found.add(root_of(node.variable))
                elif isinstance(node, Shift):
                    found.add(root_of(node.axis))
                elif isinstance(node, Apply) and node.op.takes_axes:
                    found.update(coordinate_factors(node.args[node.op.arity]))
        if self.time is not None:
            found.add(root_of(self.time))
        return frozenset(found)
