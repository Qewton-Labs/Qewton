"""The weak formulation: integrals over measures, one residual per test function.

The form is written in space only (method of lines): a time-dependent problem keeps
its time derivatives inside the integrands, and the test functions depend on space.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.equations.diagnostics import Diagnostic
from qewton.equations.analysis.classify.spaces import FunctionSpace
from qewton.equations.analysis.structure.structure import linearity
from qewton.equations.expr.nodes import Expr, FunctionSymbol
from qewton.equations.expr.ops import Linearity
from qewton.equations.geometry.sets import Part, Periodic, SetNode, lies_on_boundary
from qewton.equations.model.constraints import Constraint


@dataclass(frozen=True)
class Measure:
    """Where an integral is taken: `dx` over a domain, `ds` over a boundary or a part."""

    on: SetNode

    @property
    def is_boundary(self) -> bool:
        return lies_on_boundary(self.on)

    def __str__(self) -> str:
        symbol = "ds" if self.is_boundary else "dx"
        if isinstance(self.on, Part):
            return f"{symbol}({self.on.name})"
        return symbol


@dataclass(frozen=True)
class FormTerm:
    """`sign * integral of integrand over measure`."""

    sign: int
    integrand: Expr
    measure: Measure

    def __str__(self) -> str:
        return f"{'+' if self.sign > 0 else '-'} ∫ {self.integrand} {self.measure}"


@dataclass(frozen=True)
class Residual:
    """`F(u; v) = sum of terms`, required to vanish for every test function `v`.

    `row` is the index of the normalized constraint that was tested.
    """

    unknown: FunctionSymbol
    test: FunctionSymbol
    row: int
    terms: tuple[FormTerm, ...]

    def __str__(self) -> str:
        body = " ".join(str(term) for term in self.terms) or "0"
        return f"F[{self.test.variable.name}] = {body} = 0"


@dataclass(frozen=True)
class WeakProblem:
    """A problem in weak form.

    `tests` gives the test function of every unknown and `spaces` the space it needs
    after integration by parts. `essential` are the Dirichlet conditions, imposed on
    the space and not in the form; `periodic` are the declared periodic pairs, which
    belong to the space too. `constraints` are the conditions that stay as they are
    (initial and global ones). `diagnostics` hold what could not be compiled.
    """

    unknowns: tuple[FunctionSymbol, ...]
    tests: Mapping[FunctionSymbol, FunctionSymbol]
    spaces: Mapping[FunctionSymbol, FunctionSpace]
    residuals: tuple[Residual, ...]
    essential: tuple[Constraint, ...]
    periodic: tuple[Periodic, ...]
    constraints: tuple[Constraint, ...]
    time: Variable | None
    diagnostics: tuple[Diagnostic, ...]

    @property
    def linearity(self) -> Linearity:
        """LINEAR if every term is linear in the unknowns, NONLINEAR if one is not."""
        unknowns = frozenset(self.unknowns)
        found = {
            linearity(term.integrand, unknowns)
            for residual in self.residuals
            for term in residual.terms
        }
        if Linearity.NONLINEAR in found:
            return Linearity.NONLINEAR
        return Linearity.LINEAR if Linearity.LINEAR in found else Linearity.NONE

    def operator_terms(self, residual: Residual) -> tuple[FormTerm, ...]:
        """The terms of `residual` that contain an unknown: the form a(u, v)."""
        return tuple(t for t in residual.terms if self._has_unknown(t))

    def source_terms(self, residual: Residual) -> tuple[FormTerm, ...]:
        """The terms without an unknown; the load is ℓ(v) = -sum of them."""
        return tuple(t for t in residual.terms if not self._has_unknown(t))

    def _has_unknown(self, term: FormTerm) -> bool:
        unknowns = frozenset(self.unknowns)
        return linearity(term.integrand, unknowns) is not Linearity.NONE

    def __str__(self) -> str:
        lines = [f"find {', '.join(str(u.variable.name) for u in self.unknowns)}:"]
        lines += [f"  {residual}" for residual in self.residuals]
        for constraint in self.essential:
            lines.append(f"  essential: {constraint}")
        for constraint in self.constraints:
            lines.append(f"  also: {constraint}")
        for unknown, space in self.spaces.items():
            lines.append(f"  {unknown.variable.name} in {space.value}")
        lines += [f"  {diagnostic}" for diagnostic in self.diagnostics]
        return "\n".join(lines)
