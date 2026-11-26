"""Shape inference for function symbols whose output shape is not declared.

A symbol made with `Variable("u")` has no `dim`, so its shape is unknown. It gets one
data configuration of unknown shape, shared by all its occurrences. Building the
configurations of both sides of a constraint (`Expr.config`) relates them to their
operands, and merging the two sides makes them one shape, so what is known at one
place is known at every other: a scalar right side of `-div(q, x) = f` makes `q` a
vector of length dim(x), and `sin(u)` makes `u` a scalar.

Some relations cannot decide while their operands are open, e.g. which side of a sum
is the scalar; such a constraint waits. When no waiting constraint can go on, the
first symbol (in order of appearance) of which nothing is known is taken to be a
scalar and recorded in `Inference.defaulted`. A symbol known only in part, such as
`q` in `q[0]`, is never guessed: its shape has to be declared. Finally every
constraint is checked with the inferred shapes, so a conflict raises a ShapeError.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from qewton.config.data_configurations.base import DataConfiguration
from qewton.config.errors import StructureUnknown
from qewton.equations.expr.base import Expr, SymbolConfigs
from qewton.equations.expr.leaves import Const, FunctionSymbol
from qewton.equations.expr.ops import (
    Shape,
    ShapeError,
    describe,
    fixed,
    known_rank,
    known_shape,
    unify,
    unknown,
)
from qewton.equations.model.constraints import Constraint
from qewton.equations.model.problem import Problem


@dataclass(frozen=True)
class Inference:
    """The output shape of every function in the problem.

    `defaulted` lists the symbols that nothing determined and that were taken to be
    scalars, in the order they were decided. Each deserves a warning.
    """

    shapes: Mapping[FunctionSymbol, Shape]
    defaulted: tuple[FunctionSymbol, ...]


def infer_shapes(problem: Problem) -> Inference:
    order = list(dict.fromkeys(s for c in problem.constraints for s in c.symbols()))
    env: SymbolConfigs = {}
    defaulted: list[FunctionSymbol] = []
    waiting = list(problem.constraints)
    while waiting:
        still = [c for c in waiting if not _relate(c, env)]
        if len(still) == len(waiting):
            symbol = _first_unknown(order, env)
            if symbol is None:
                raise ShapeError(
                    f"The shapes in {still[0]} cannot be inferred; declare the output "
                    "shapes of its functions, e.g. Variable('u', dim=2)."
                )
            _default(symbol, env, defaulted)
        waiting = still
    # Symbols that no constraint says anything about are scalars too.
    while (symbol := _first_unknown(order, env)) is not None:
        _default(symbol, env, defaulted)

    shapes = {s: s.shape if s.shape is not None else _determined(s, env) for s in order}
    for constraint in problem.constraints:
        _check(constraint, shapes)
    return Inference(shapes, tuple(defaulted))


def _relate(constraint: Constraint, env: SymbolConfigs) -> bool:
    """Relate both sides of `constraint`; False while it has to wait."""
    try:
        left, right = constraint.lhs.config(env), constraint.rhs.config(env)
    except StructureUnknown:
        return False
    # A scalar constant stands for any shape, e.g. 0 for a zero vector.
    if not (_scalar_const(constraint.lhs, left) or _scalar_const(constraint.rhs, right)):
        sides = f"{describe(left)} and {describe(right)}"
        unify(left, right, f"In {constraint}: the sides have shapes {sides}.")
    return True


def _scalar_const(expr: Expr, config: DataConfiguration) -> bool:
    return isinstance(expr, Const) and known_rank(config) == 0


def _first_unknown(
    order: list[FunctionSymbol], env: SymbolConfigs
) -> FunctionSymbol | None:
    """The first undeclared symbol of which nothing is known yet."""
    for symbol in order:
        if symbol.shape is None and env.setdefault(symbol, unknown()).resolve().shape == (
            ...,
        ):
            return symbol
    return None


def _default(symbol: FunctionSymbol, env: SymbolConfigs, defaulted: list) -> None:
    unify(env[symbol], fixed(()), f"{symbol.name!r} cannot be a scalar.")
    defaulted.append(symbol)


def _determined(symbol: FunctionSymbol, env: SymbolConfigs) -> Shape:
    shape = known_shape(env[symbol])
    if shape is None:
        raise ShapeError(
            f"The output shape of {symbol.name!r} is only known in part, "
            f"{describe(env[symbol])}; declare it, e.g. Variable({symbol.name!r}, dim=2)."
        )
    return shape


def constraint_shape(
    constraint: Constraint, shapes: Mapping[FunctionSymbol, Shape]
) -> Shape:
    """The shape of both sides of `constraint`, when the shape of every function is
    given; a scalar constant side fits any shape."""
    left, right = (_shape_of(side, shapes) for side in constraint.sides)
    if left == right:
        return left
    if left == () and isinstance(constraint.lhs, Const):
        return right
    if right == () and isinstance(constraint.rhs, Const):
        return left
    raise ShapeError(f"Cannot determine scalar equation components for {constraint}.")


def _shape_of(expr: Expr, shapes: Mapping[FunctionSymbol, Shape]) -> Shape:
    shape = expr.shape_with(shapes)
    if shape is None:
        raise ShapeError(f"The shape of {expr} is not determined by the given shapes.")
    return shape


def _check(constraint: Constraint, shapes: Mapping[FunctionSymbol, Shape]) -> None:
    left, right = (side.shape_with(shapes) for side in constraint.sides)
    if left == right or (left == () and isinstance(constraint.lhs, Const)):
        return
    if right == () and isinstance(constraint.rhs, Const):
        return
    raise ShapeError(f"In {constraint}: the sides have shapes {left} and {right}.")
