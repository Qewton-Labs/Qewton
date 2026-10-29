"""signatures: what each function is, as a type checker would show it.

For every function of a problem: the coordinates it depends on (its declared inputs),
its batch axis if it varies, its output shape, and the sets it is evaluated on.

    u : x → ℝ        on Omega, Gamma_D, Gamma_N
    nu : P → ℝ       varying, constant in x

Inputs are declared, never inferred. The analysis adds only what it can know: the
shape (see `infer_shapes`), the sets, and a warning for an input that no constraint
makes use of.

A function that is coupled to a varying one (u depends on a varying nu, so u has a
value per instance too) is not marked here. That is a statement about the coupling of
the equations, which comes with the structure analysis.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.equations.analysis.diagnostics import Diagnostic, warning
from qewton.equations.analysis.infer import infer_shapes
from qewton.equations.analysis.normalize import normalize
from qewton.equations.expr.nodes import (
    Apply,
    Expr,
    FunctionSymbol,
    Integral,
    Restrict,
    Shift,
)
from qewton.equations.geometry.coordinates import coordinate_factors, root_of
from qewton.equations.geometry.sets import Product, SetNode, is_subset
from qewton.equations.model.task import Role, Task

Shape = tuple[int, ...]


@dataclass(frozen=True)
class Signature:
    """One function's type.

    `axes` are the declared input coordinates in order, `batch` is the parameter axis
    of a varying function (or None), `shape` is the output shape, `evaluated_on` the
    sets the function is evaluated on, none of them inside another, and `role` comes
    from the task. `shape_assumed` is True when the shape was not declared and had to
    be taken as scalar.
    """

    symbol: FunctionSymbol
    axes: tuple[Variable, ...]
    batch: Variable | None
    shape: Shape
    evaluated_on: tuple[SetNode, ...]
    role: Role
    shape_assumed: bool = False

    def __str__(self) -> str:
        inputs = [a.name for a in self.axes]
        if self.batch is not None:
            inputs.append(self.batch.name)
        domain = " × ".join(inputs) if inputs else "·"
        shape = f"ℝ^{self.shape}" if self.shape else "ℝ"
        assumed = " (assumed)" if self.shape_assumed else ""
        sets = ", ".join(str(s) for s in self.evaluated_on)
        where = f"  on {sets}" if sets else ""
        return (
            f"{self.symbol.variable.name} : {domain} → {shape}{assumed}"
            f"  [{self.role.value}]{where}"
        )


@dataclass(frozen=True)
class Signatures:
    """The signature of every function, and what the analysis noticed on the way."""

    table: dict[FunctionSymbol, Signature]
    diagnostics: tuple[Diagnostic, ...]

    def __getitem__(self, symbol: FunctionSymbol) -> Signature:
        return self.table[symbol]

    def __iter__(self) -> Iterator[Signature]:
        return iter(self.table.values())

    def __str__(self) -> str:
        return "\n".join(str(signature) for signature in self)


def _narrowed(where: SetNode | None, on: SetNode) -> SetNode:
    """The set a restriction evaluates on, inside a constraint holding on `where`.

    A restriction to a boundary inside a constraint lifted to boundary x I still
    evaluates on boundary x I: the factors it does not mention are kept.
    """
    if not isinstance(where, Product) or not on.axes < where.axes:
        return on
    rest = tuple(f for f in where.factors if not f.axes & on.axes)
    return Product((on,) + rest) if rest else on


def _occurrences(expr: Expr, where: SetNode | None):
    """Yield (function, set it is evaluated on) for every function in `expr`.

    A restriction or an integral changes the set; everything else keeps it.
    """
    if isinstance(expr, FunctionSymbol):
        yield expr, where
    elif isinstance(expr, Restrict):
        yield from _occurrences(expr.expr, _narrowed(where, expr.on))
    elif isinstance(expr, Integral):
        yield from _occurrences(expr.expr, expr.over)
    elif isinstance(expr, Shift):
        yield from _occurrences(expr.expr, where)
        yield from _occurrences(expr.offset, where)
    elif isinstance(expr, Apply):
        for arg in expr.args:
            if isinstance(arg, Expr):
                yield from _occurrences(arg, where)


def _outermost(sets: set[SetNode]) -> tuple[SetNode, ...]:
    kept = [s for s in sets if not any(o != s and is_subset(s, o) for o in sets)]
    return tuple(sorted(kept, key=lambda s: (s.codim, str(s))))


def signatures(task: Task) -> Signatures:
    """The signature of every function of the task's problem.

    Runs on the normalized problem, so time lifting is already applied: a condition
    on a boundary for a time-dependent function counts as evaluated on boundary × I.
    """
    problem = normalize(task.problem).problem
    inference = infer_shapes(problem)

    evaluated: dict[FunctionSymbol, set[SetNode]] = {}
    seen_with: dict[FunctionSymbol, set[Variable]] = {}
    for constraint in problem.constraints:
        for side in (constraint.lhs, constraint.rhs):
            for symbol, where in _occurrences(side, constraint.on):
                if where is None:
                    continue
                evaluated.setdefault(symbol, set()).add(where)
                seen_with.setdefault(symbol, set()).update(where.axes)

    table: dict[FunctionSymbol, Signature] = {}
    found: list[Diagnostic] = []
    for symbol in sorted(inference.shapes, key=lambda s: s.variable.name):
        axes = () if symbol.inputs is None else coordinate_factors(symbol.inputs)
        table[symbol] = Signature(
            symbol,
            axes,
            task.batch_axes.get(symbol),
            inference.shapes[symbol],
            _outermost(evaluated.get(symbol, set())),
            task.roles[symbol],
            symbol in inference.defaulted,
        )
        if symbol in evaluated:
            unused = [a for a in axes if root_of(a) not in seen_with.get(symbol, set())]
            if unused:
                names = ", ".join(a.name for a in unused)
                found.append(
                    warning(
                        f"{symbol.variable.name} is declared to depend on {names}, "
                        "but no constraint it appears in holds on a set with it. "
                        "Drop the input, or hold the constraint on a larger set.",
                        symbol,
                    )
                )
    return Signatures(table, tuple(found))
