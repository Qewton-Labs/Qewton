"""Boundary terms of a weak form: what becomes of the pieces integration by parts leaves.

On each region of the boundary of a domain (a part of its partition, or the whole
boundary if it is not partitioned) the pieces `∮ flux·w` are treated as follows:

- a periodic part: dropped, the terms of the two sides cancel;
- a Dirichlet condition on the unknown of `w`: dropped, since `w` vanishes there;
- a Neumann or Robin condition that states the flux: the flux is replaced by the
  other terms of the condition;
- otherwise the raw integral is kept, with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.equations.analysis.checks.diagnostics import Diagnostic, info, warning
from qewton.equations.analysis.classify.conditions import ConditionKind
from qewton.equations.analysis.classify.rows import Row
from qewton.equations.analysis.structure.structure import split_constraint
from qewton.equations.compile.by_parts import BoundaryPiece
from qewton.equations.compile.forms import FormTerm, Measure
from qewton.equations.expr.api import inner
from qewton.equations.expr.nodes import FunctionSymbol, Normal, Restrict
from qewton.equations.geometry.sets import Boundary, SetNode


@dataclass(frozen=True)
class Condition:
    """A boundary condition: its row, its kind and the part of the boundary it is on."""

    row: Row
    kind: ConditionKind
    on: SetNode


@dataclass(frozen=True)
class BoundaryContext:
    """What the treatment of the boundary of one equation needs to know."""

    boundary: Boundary
    regions: tuple[SetNode, ...]
    owner: FunctionSymbol
    conditions: tuple[Condition, ...]
    unknowns: frozenset[FunctionSymbol]
    periodic: frozenset[SetNode]


def _name(region: SetNode) -> str:
    return str(getattr(region, "name", region))


def _groups(pieces: list[BoundaryPiece]) -> list[list[BoundaryPiece]]:
    """Pieces that are tested with the same factor; they share one condition."""
    groups: list[list[BoundaryPiece]] = []
    for piece in pieces:
        for group in groups:
            if group[0].test == piece.test:
                group.append(piece)
                break
        else:
            groups.append([piece])
    return groups


def _substitute(
    group: list[BoundaryPiece],
    region: SetNode,
    condition: Condition,
    context: BoundaryContext,
) -> list[FormTerm] | None:
    """Replace the fluxes of `group` using `condition`, or None if it does not state them.

    With the condition's terms t_i * T_i summing to zero, and the fluxes s_j * F_j, each
    flux has to be one of the T_i, with the same ratio k = t_i * s_j every time. Then the
    sum of the fluxes is -k times the other terms of the condition.
    """
    split = split_constraint(condition.row.constraint, context.unknowns)
    terms = [(t.expr, t.sign) for t in (*split.operator, *split.source)]
    normal = Normal(context.boundary)
    used: set[int] = set()
    ratios: set[int] = set()
    for piece in group:
        target = piece.flux(Restrict(piece.operand, region), normal)
        index = next(
            (i for i, (e, _) in enumerate(terms) if i not in used and e == target), None
        )
        if index is None:
            return None
        used.add(index)
        ratios.add(terms[index][1] * piece.sign)
    if len(ratios) != 1:
        return None
    k = ratios.pop()
    measure = Measure(region)
    return [
        FormTerm(-k * sign, inner(expr, group[0].test), measure)  # type: ignore
        for i, (expr, sign) in enumerate(terms)
        if i not in used
    ]


def _raw(
    group: list[BoundaryPiece], region: SetNode, context: BoundaryContext
) -> list[FormTerm]:
    normal = Normal(context.boundary)
    measure = Measure(region)
    return [
        FormTerm(
            p.sign,
            inner(p.flux(Restrict(p.operand, region), normal), p.test),
            measure,
        )
        for p in group
    ]


def _natural(
    pieces: list[BoundaryPiece], region: SetNode, context: BoundaryContext
) -> list[FormTerm] | None:
    """The terms from a condition that states the flux of every group, else None."""
    named = [
        c
        for c in context.conditions
        if c.on in (region, context.boundary)
        and c.kind in (ConditionKind.NEUMANN, ConditionKind.ROBIN)
    ]
    found: list[FormTerm] = []
    for group in _groups(pieces):
        for condition in named:
            terms = _substitute(group, region, condition, context)
            if terms is not None:
                found += terms
                break
        else:
            return None
    return found


def _dirichlet(region: SetNode, context: BoundaryContext) -> bool:
    return any(
        c.kind is ConditionKind.DIRICHLET
        and c.on in (region, context.boundary)
        and context.owner in c.row.orders
        for c in context.conditions
    )


def boundary_terms(
    pieces: list[BoundaryPiece], context: BoundaryContext
) -> tuple[list[FormTerm], list[Diagnostic]]:
    """The boundary part of a weak form, and notes about what had to be kept raw."""
    terms: list[FormTerm] = []
    notes: list[Diagnostic] = []
    if not pieces:
        return terms, notes
    for region in context.regions:
        if region in context.periodic:
            notes.append(info(f"The boundary terms on {_name(region)} cancel.", region))
        elif _dirichlet(region, context):
            continue
        else:
            natural = _natural(pieces, region, context)
            if natural is not None:
                terms += natural
                continue
            notes.append(_unstated(region, context))
            for group in _groups(pieces):
                terms += _raw(group, region, context)
    return terms, notes


def _unstated(region: SetNode, context: BoundaryContext) -> Diagnostic:
    here = [c for c in context.conditions if c.on in (region, context.boundary)]
    if here:
        why = "no condition on it states the boundary flux"
    else:
        why = "there is no condition on it"
    return warning(
        f"The boundary integral on {_name(region)} is kept as it is: {why}.", region
    )
