"""Constraints: a relation between two expressions, holding on a set.

Build one with Eq, Le, Ge or Complementary, then attach the set with .on(...):

    Eq(-div(nu * grad(u, x), x), f).on(Omega)

Until .on(...) is called the result is a PendingConstraint, which Problem refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from qewton.equations.expr.nodes import Expr, as_expr
from qewton.equations.geometry.sets import SetNode
from qewton.equations.analysis.shapes import check_sides
from qewton.equations.printing.printable import Printable


class Relation(Enum):
    EQ = "="
    LE = "<="
    GE = ">="
    COMPLEMENTARITY = "⊥"  # lhs ⊥ rhs: both are non-negative and their product is zero


@dataclass(frozen=True)
class Constraint(Printable):
    """`lhs rel rhs`, required at every point of `on`.

    `on=None` is a global constraint, e.g. a zero mean of the pressure: it holds once,
    and its sides must not depend on any coordinate.
    """

    lhs: Expr
    rel: Relation
    rhs: Expr
    on: SetNode | None
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.lhs, Expr) or not isinstance(self.rhs, Expr):
            raise TypeError("Both sides of a constraint must be expressions.")
        if not isinstance(self.rel, Relation):
            raise TypeError(f"Expected a Relation, got {self.rel!r}.")
        if self.on is not None and not isinstance(self.on, SetNode):
            raise TypeError(
                f"A constraint holds on a set or None, got {type(self.on).__name__}."
            )
        check_sides(self.lhs, self.rhs)


@dataclass(frozen=True)
class PendingConstraint(Printable):
    """Both sides and the relation, waiting for the set they hold on."""

    lhs: Expr
    rel: Relation
    rhs: Expr

    def on(self, on: SetNode | None, name: str | None = None) -> Constraint:
        return Constraint(self.lhs, self.rel, self.rhs, on, name)


def _pending(lhs: object, rel: Relation, rhs: object) -> PendingConstraint:
    return PendingConstraint(as_expr(lhs), rel, as_expr(rhs))


# Named after the math they build, which is why the names are capitalized.
# pylint: disable=invalid-name
def Eq(lhs: object, rhs: object) -> PendingConstraint:
    return _pending(lhs, Relation.EQ, rhs)


def Le(lhs: object, rhs: object) -> PendingConstraint:
    return _pending(lhs, Relation.LE, rhs)


def Ge(lhs: object, rhs: object) -> PendingConstraint:
    return _pending(lhs, Relation.GE, rhs)


def Complementary(lhs: object, rhs: object) -> PendingConstraint:
    return _pending(lhs, Relation.COMPLEMENTARITY, rhs)
