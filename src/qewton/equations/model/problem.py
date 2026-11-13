"""The underlying mathematical problem: constraints,
boundary partitions and the variables.

A Problem is mathematics only. It assigns no roles (unknown, fixed, varying);
that is the Task's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.equations.expr.axes import AxisError, free_axes
from qewton.equations.expr.nodes import FunctionSymbol
from qewton.equations.geometry.coordinates import check_coordinate, root_of
from qewton.equations.geometry.sets import (
    Interface,
    Partition,
    Periodic,
    SetNode,
    set_free_axes,
)
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.task import Task, build_task
from qewton.equations.printable import Printable


@dataclass(frozen=True)
class Problem(Printable):
    constraints: tuple[Constraint, ...]
    partitions: tuple[Partition, ...] = ()
    periodic: tuple[Periodic, ...] = ()
    time: Variable | None = None
    interfaces: tuple[Interface, ...] = ()

    def __post_init__(self):
        # Frozen dataclass: store the sequences as tuples, so the problem hashes.
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "partitions", tuple(self.partitions))
        object.__setattr__(self, "periodic", tuple(self.periodic))
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
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
        for interface in self.interfaces:
            if not isinstance(interface, Interface):
                raise TypeError(f"Expected an Interface, got {type(interface).__name__}.")
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
            _check_axes(constraint, self.time)

    def task(self, unknown=(), fixed=None, varying=None, observed=None) -> Task:
        """Assign roles to the functions of this problem; see `build_task`.

        problem.task(unknown=[u], fixed={f: 1.0, Omega: mesh}, varying={nu: samples})

        Samples of an unknown go in `observed={u: Observed(on=..., samples=...)}`.
        """
        return build_task(self, unknown, fixed, varying, observed)

    def overview(self):
        """The coordinates, sets and functions of this problem; print it to read it."""
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from qewton.equations.analysis.problem_overview import overview

        return overview(self)

    def symbols(self) -> frozenset[FunctionSymbol]:
        """Every function that appears in a constraint."""
        return frozenset(s for c in self.constraints for s in c.symbols())

    def sets(self) -> frozenset[SetNode]:
        """Every set the problem refers to: where constraints hold, where expressions are
        restricted, integrated or normals taken, and the partitions, periodic pairs
        and interfaces.
        """
        found = {s for c in self.constraints for s in c.sets()}
        for partition in self.partitions:
            found.add(partition.whole)
            found.update(partition.parts)
        for declaration in self.periodic:
            found.update((declaration.first, declaration.second))
        found.update(self.interfaces)
        return frozenset(found)

    def coordinates(self) -> frozenset[Variable]:
        """Every coordinate the problem uses: those of its sets, the inputs of its
        functions, coordinate components, derivative and shift axes, and time."""
        found: set[Variable] = {axis for s in self.sets() for axis in s.axes}
        for constraint in self.constraints:
            for node in constraint.nodes():
                found.update(node.coordinates())
        if self.time is not None:
            found.add(root_of(self.time))
        return frozenset(found)


def _check_axes(constraint: Constraint, time: Variable | None) -> None:
    """Both sides may depend only on the coordinates the constraint holds on.

    The time coordinate is the exception: a constraint on a set without time that
    uses time-dependent functions is lifted to hold at every time, so it is allowed.
    A global constraint holds on no coordinate.
    """
    on = constraint.on
    allowed = frozenset() if on is None else set_free_axes(on)
    if time is not None:
        allowed |= {root_of(time)}
    used = free_axes(constraint.lhs) | free_axes(constraint.rhs)
    extra = used - allowed
    if extra:
        names = "{" + ", ".join(sorted(str(axis.name) for axis in extra)) + "}"
        where = "globally" if on is None else f"on {on}"
        hint = (
            " Integrate or restrict it first."
            if on is None
            else " Restrict it, integrate it, or hold the constraint on a larger set."
        )
        raise AxisError(
            f"The constraint {constraint} holds {where} but depends on {names}.{hint}"
        )
