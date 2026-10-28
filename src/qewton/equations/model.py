"""The PDE problem: constraints, boundary partitions and the time variable.

A Problem is mathematics only. It assigns no roles (unknown, fixed, batched); that
is the Task's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.equations.constraints import Constraint
from qewton.equations.expr.nodes import Expr, FunctionSymbol, Normal, Restrict, walk
from qewton.equations.geometry.coordinates import check_coordinate
from qewton.equations.geometry.sets import Partition, SetNode
from qewton.equations.printable import Printable


@dataclass(frozen=True)
class Problem(Printable):
    constraints: tuple[Constraint, ...]
    partitions: tuple[Partition, ...] = ()
    time: Variable | None = None

    def __post_init__(self):
        # Frozen dataclass: store the sequences as tuples, so the problem hashes.
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "partitions", tuple(self.partitions))
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
        if self.time is not None:
            check_coordinate(self.time)
            if self.time.dim != 1:
                raise ValueError(
                    f"The time variable must be one-dimensional, got dim {self.time.dim}."
                )

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
        found = {constraint.on for constraint in self.constraints}
        for side in self._sides():
            for node in walk(side):
                if isinstance(node, Restrict):
                    found.add(node.on)
                elif isinstance(node, Normal):
                    found.add(node.of)
        for partition in self.partitions:
            found.add(partition.whole)
            found.update(partition.parts)
        return frozenset(found)
