"""Shape inference for function symbols whose output shape is not declared.

A symbol made with `Variable("u")` has no `dim`, so its shape is unknown. Three steps
fill it in, in this order:

1. Known shapes flow backwards through operators that can be inverted. If the right
   side of an equation is a scalar, then `-div(q, x)` is a scalar, so `q` is a vector
   of length `dim(x)`, and so on down to the symbols.
2. Scalar-only operators (sin, cos, exp, pow) force their arguments to be scalar.
3. If nothing more follows, the first symbol that is still unknown (in order of
   appearance) is taken to be a scalar and recorded in `Inference.defaulted`. Then
   steps 1 and 2 continue, which usually resolves the rest.

Finally every constraint is checked with the inferred shapes, so a conflict raises
a ShapeError.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import singledispatch

from qewton.equations.analysis.checks.shapes import index_shape, shape_of
from qewton.equations.model.constraints import Constraint
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Expr,
    FunctionSymbol,
    Index,
    Integral,
    Restrict,
    Shift,
    walk,
)
from qewton.equations.expr.ops import Shape, ShapeError
from qewton.equations.model.problem import Problem

Env = dict[FunctionSymbol, Shape]


@dataclass(frozen=True)
class Inference:
    """The output shape of every function in the problem.

    `defaulted` lists the symbols that nothing determined and that were taken to be
    scalars, in the order they were decided. Each deserves a warning.
    """

    shapes: Mapping[FunctionSymbol, Shape]
    defaulted: tuple[FunctionSymbol, ...]


def infer_shapes(problem: Problem) -> Inference:
    env: Env = {}
    defaulted: list[FunctionSymbol] = []
    while True:
        _propagate(problem, env)
        pending = [
            s for s in _symbols_in_order(problem) if s.shape is None and s not in env
        ]
        if not pending:
            break
        _reject_indexed(problem, pending[0])
        env[pending[0]] = ()
        defaulted.append(pending[0])
    for constraint in problem.constraints:
        _check(constraint, env)
    shapes = {
        symbol: symbol.shape if symbol.shape is not None else env[symbol]
        for symbol in _symbols_in_order(problem)
    }
    return Inference(shapes, tuple(defaulted))


def _reject_indexed(problem: Problem, symbol: FunctionSymbol) -> None:
    """A component does not say how many components there are, so it cannot default."""
    for constraint in problem.constraints:
        for side in (constraint.lhs, constraint.rhs):
            for node in walk(side):
                if isinstance(node, Index) and node.expr == symbol:
                    raise ShapeError(
                        f"The output shape of {symbol.name!r} cannot be inferred from "
                        f"{node}; declare it, e.g. Variable({symbol.name!r}, dim=2)."
                    )


def shape_with(expr: Expr, shapes: Mapping[FunctionSymbol, Shape]) -> Shape:
    """The shape of `expr` when the output shape of every function is given."""
    shape = _partial(expr, dict(shapes))
    if shape is None:
        raise ShapeError(f"The shape of {expr} is not determined by the given shapes.")
    return shape


def constraint_shape(
    constraint: Constraint, shapes: Mapping[FunctionSymbol, Shape]
) -> Shape:
    """The shape of both sides of `constraint`; a scalar constant side fits any shape."""
    left = shape_with(constraint.lhs, shapes)
    right = shape_with(constraint.rhs, shapes)
    if left == right:
        return left
    if left == () and isinstance(constraint.lhs, Const):
        return right
    if right == () and isinstance(constraint.rhs, Const):
        return left
    raise ShapeError(f"Cannot determine scalar equation components for {constraint}.")


def _symbols_in_order(problem: Problem) -> list[FunctionSymbol]:
    found: dict[FunctionSymbol, None] = {}
    for constraint in problem.constraints:
        for side in (constraint.lhs, constraint.rhs):
            for node in walk(side):
                if isinstance(node, FunctionSymbol):
                    found[node] = None
    return list(found)


def _propagate(problem: Problem, env: Env) -> None:
    """Apply steps 1 and 2 until the environment stops growing."""
    while True:
        known = len(env)
        for constraint in problem.constraints:
            _constrain(constraint.lhs, constraint.rhs, env)
            for side in (constraint.lhs, constraint.rhs):
                for node in walk(side):
                    _demand(node, env)
        if len(env) == known:
            return


def _constrain(lhs: Expr, rhs: Expr, env: Env) -> None:
    """Equal sides have equal shapes, except that a scalar constant fits any shape."""
    left, right = _partial(lhs, env), _partial(rhs, env)
    if left is None and right is not None and not _scalar_const(rhs, right):
        _back(lhs, right, env)
    elif right is None and left is not None and not _scalar_const(lhs, left):
        _back(rhs, left, env)


def _scalar_const(expr: Expr, shape: Shape) -> bool:
    return isinstance(expr, Const) and shape == ()


@singledispatch
def _demand(_node: Expr, _env: Env) -> None:
    """Shapes that a node needs whatever its result is."""


@_demand.register
def _(node: Apply, env: Env) -> None:
    if node.op.scalar_args:
        for arg in node.args[: node.op.arity]:
            _back(arg, (), env)


@_demand.register
def _(node: Shift, env: Env) -> None:
    _back(node.offset, _offset_shape(node), env)


def _offset_shape(shift: Shift) -> Shape:
    if isinstance(shift.axis.dim, tuple):
        return shift.axis.dim
    return () if shift.axis.dim == 1 or not shift.axis.dim else (shift.axis.dim,)


@singledispatch
def _partial(expr: Expr, _env: Env) -> Shape | None:
    """The shape of `expr` from declared and inferred shapes, or None if unknown."""
    return shape_of(expr)


@_partial.register
def _(expr: FunctionSymbol, env: Env) -> Shape | None:
    return expr.shape if expr.shape is not None else env.get(expr)


@_partial.register(Restrict)
@_partial.register(Integral)
def _(expr: Restrict | Integral, env: Env) -> Shape | None:
    return _partial(expr.expr, env)


@_partial.register
def _(expr: Index, env: Env) -> Shape | None:
    operand = _partial(expr.expr, env)
    if operand is None:
        return None
    return index_shape(expr, operand)


@_partial.register
def _(expr: Shift, env: Env) -> Shape | None:
    offset = _partial(expr.offset, env)
    expected = _offset_shape(expr)
    if offset is not None and offset != expected:
        raise ShapeError(f"In {expr}: the offset needs shape {expected}, got {offset}.")
    return _partial(expr.expr, env)


@_partial.register
def _(expr: Apply, env: Env) -> Shape | None:
    shapes = tuple(_partial(arg, env) for arg in expr.args[: expr.op.arity])
    if any(shape is None for shape in shapes):
        return None
    axes = expr.args[expr.op.arity] if expr.op.takes_axes else None
    try:
        return expr.op.shape_rule(shapes, axes)  # type: ignore
    except ShapeError as error:
        raise ShapeError(f"In {expr}: {error}") from error


def _assign(symbol: FunctionSymbol, shape: Shape, env: Env) -> None:
    if symbol.shape is not None:
        return
    if env.setdefault(symbol, shape) != shape:
        raise ShapeError(
            f"{symbol.name!r} is used with shapes {env[symbol]} and {shape}."
        )


@singledispatch
def _back(_expr: Expr, _target: Shape, _env: Env) -> None:
    """Record what `expr` having shape `target` says about the symbols inside it."""


@_back.register
def _(expr: FunctionSymbol, target: Shape, env: Env) -> None:
    _assign(expr, target, env)


@_back.register(Restrict)
@_back.register(Integral)
@_back.register(Shift)
def _(expr: Restrict | Integral | Shift, target: Shape, env: Env) -> None:
    _back(expr.expr, target, env)


@_back.register
def _(expr: Apply, target: Shape, env: Env) -> None:
    rule = expr.op.back_rule
    if rule is None:
        return
    args = expr.args[: expr.op.arity]
    axes = expr.args[expr.op.arity] if expr.op.takes_axes else None
    try:
        rule(args, axes, target, _Solver(env))  # type: ignore
    except ShapeError as error:
        raise ShapeError(f"In {expr}: {error}") from error


@dataclass(frozen=True)
class _Solver:
    """The view of shape inference that operator back rules get."""

    env: Env

    def known(self, expr: Expr) -> Shape | None:
        return _partial(expr, self.env)

    def back(self, expr: Expr, target: Shape) -> None:
        _back(expr, target, self.env)


def _check(constraint: Constraint, env: Env) -> None:
    left, right = _partial(constraint.lhs, env), _partial(constraint.rhs, env)
    if (
        left == right
        or _scalar_const(constraint.lhs, left)  # type: ignore
        or _scalar_const(constraint.rhs, right)  # type: ignore
    ):
        return
    raise ShapeError(f"In {constraint}: the sides have shapes {left} and {right}.")
