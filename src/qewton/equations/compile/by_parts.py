"""Integration by parts: moving derivatives from the unknown onto the test function.

For a term `T` of an equation tested with `w`, the integral of `inner(T, w)` is cut
into interior terms and boundary pieces. A term is cut while one of its unknowns is
differentiated more often than its space allows. Each operator that can be cut has a
rule in `RULES`:

    ∫ div(σ)·w  = -∫ σ : ∇w  + ∮ (σ·n)·w
    ∫ ∇p · w    = -∫ p div w + ∮ (p n)·w     (p scalar)

The interior term is cut again if it still has too many derivatives. A scalar factor
without unknowns is moved onto the test function first, so that `ν ∇u` stays together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from qewton.config.variables import Variable
from qewton.equations.analysis.structure.structure import derivative_orders, signed_terms
from qewton.equations.expr.api import div, dot, grad, inner
from qewton.equations.expr.nodes import (
    Apply,
    Coord,
    Expr,
    FunctionSymbol,
    contains,
    walk,
)
from qewton.equations.expr.ops import ShapeError, Shape
from qewton.equations.analysis.prepare.infer import shape_with


@dataclass(frozen=True)
class Rule:
    """How one operator moves onto the test function.

    `moved` builds the test factor of the interior term from the current one and the
    derivative axes. `flux` builds the boundary integrand's flux from the operand and
    the normal. `applies` tells from the operand's shape whether the rule is valid.
    """

    moved: Callable[[Expr, Variable], Expr]
    flux: Callable[[Expr, Expr], Expr]
    applies: Callable[[Shape], bool] = lambda _shape: True


RULES: dict[str, Rule] = {
    "div": Rule(moved=grad, flux=dot),
    "grad": Rule(
        moved=div,
        flux=lambda operand, normal: operand * normal,
        applies=lambda shape: shape == (),
    ),
}


@dataclass(frozen=True)
class BoundaryPiece:
    """`sign * ∮ inner(flux(operand, n), test)`, on any part of the boundary."""

    sign: int
    operand: Expr
    flux: Callable[[Expr, Expr], Expr]
    test: Expr


@dataclass
class Pieces:
    """What integration by parts produced: interior terms, boundary pieces, notes."""

    interior: list[tuple[int, Expr]] = field(default_factory=list)
    boundary: list[BoundaryPiece] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _constant(expr: Expr) -> bool:
    """Whether `expr` does not depend on any coordinate."""
    return not any(
        isinstance(node, Coord)
        or (isinstance(node, FunctionSymbol) and node.coordinates())
        for node in walk(expr)
    )


def _moved(rule: Rule, test: Expr, axes: Variable) -> Expr:
    """The test factor after the operator moved onto it; a constant factor stays out."""
    if isinstance(test, Apply) and test.op.name == "mul":
        factor, base = test.args
        if _constant(factor):  # type: ignore
            return factor * rule.moved(base, axes)  # type: ignore
    return rule.moved(test, axes)


@dataclass(frozen=True)
class Cutter:
    """Cuts terms until every unknown is within the derivative order of its space.

    `allowed` is the highest spatial derivative order of each unknown, `shapes` the
    shape of every function including the test functions.
    """

    allowed: Mapping[FunctionSymbol, int]
    shapes: Mapping[FunctionSymbol, Shape]
    time: Variable | None

    def excess(self, term: Expr) -> int:
        """How far the spatial derivatives in `term` exceed what the spaces allow."""
        found = derivative_orders(term)
        return max(
            (
                max((o for a, o in orders.items() if a != self.time), default=0)
                - self.allowed[symbol]
                for symbol, orders in found.items()
                if symbol in self.allowed
            ),
            default=0,
        )

    def cut(self, expr: Expr, test: Expr, sign: int, out: Pieces) -> None:
        """Add the pieces of `sign * ∫ inner(expr, test)` to `out`."""
        for term, term_sign in signed_terms(expr, sign):
            self._cut_term(term, test, term_sign, out)  # type: ignore

    def _cut_term(self, term: Expr, test: Expr, sign: int, out: Pieces) -> None:
        if self.excess(term) <= 0:
            out.interior.append((sign, inner(term, test)))
        elif self._by_parts(term, test, sign, out):
            return
        elif not self._scaled(term, test, sign, out):
            out.interior.append((sign, inner(term, test)))
            out.notes.append(
                f"{term} has more derivatives than its space allows and cannot be "
                "integrated by parts; it is kept as it is."
            )

    def _shape(self, expr: Expr) -> Shape | None:
        try:
            return shape_with(expr, self.shapes)
        except ShapeError:
            return None

    def _by_parts(self, term: Expr, test: Expr, sign: int, out: Pieces) -> bool:
        if not isinstance(term, Apply) or term.op.name not in RULES:
            return False
        rule = RULES[term.op.name]
        operand, axes = term.args[0], term.args[1]
        shape = self._shape(operand)  # type: ignore
        if shape is None or not rule.applies(shape):
            return False
        out.boundary.append(BoundaryPiece(sign, operand, rule.flux, test))  # type: ignore
        self.cut(operand, _moved(rule, test, axes), -sign, out)  # type: ignore
        return True

    def _scaled(self, term: Expr, test: Expr, sign: int, out: Pieces) -> bool:
        """Move a scalar factor without unknowns from `term = c * B` onto the test."""
        if not isinstance(term, Apply) or term.op.name != "mul":
            return False
        unknowns = frozenset(self.allowed)
        first, second = term.args
        for factor, rest in ((first, second), (second, first)):
            scalar = self._shape(factor) == ()  # type: ignore
            if scalar and not contains(factor, unknowns):  # type: ignore
                self.cut(rest, factor * test, sign, out)  # type: ignore
                return True
        return False
