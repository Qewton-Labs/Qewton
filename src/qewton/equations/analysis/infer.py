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

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from qewton.equations.analysis.shapes import shape_of
from qewton.equations.model.constraints import Constraint
from qewton.equations.expr.nodes import (
    Apply,
    Const,
    Expr,
    FunctionSymbol,
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
        env[pending[0]] = ()
        defaulted.append(pending[0])
    for constraint in problem.constraints:
        _check(constraint, env)
    shapes = {
        symbol: symbol.shape if symbol.shape is not None else env[symbol]
        for symbol in _symbols_in_order(problem)
    }
    return Inference(shapes, tuple(defaulted))


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


def _demand(node: Expr, env: Env) -> None:
    """Shapes that an operator needs whatever its result is."""
    if isinstance(node, Apply) and node.op.name in _SCALAR_ARGUMENTS:
        for arg in node.args[: node.op.arity]:
            _back(arg, (), env)
    elif isinstance(node, Shift):
        _back(node.offset, () if node.axis.dim == 1 else (node.axis.dim,), env)


_SCALAR_ARGUMENTS = {"pow", "exp", "sin", "cos"}


def _partial(expr: Expr, env: Env) -> Shape | None:
    """The shape of `expr` from declared and inferred shapes, or None if unknown."""
    if isinstance(expr, FunctionSymbol):
        return expr.shape if expr.shape is not None else env.get(expr)
    if isinstance(expr, (Restrict, Integral)):
        return _partial(expr.expr, env)
    if isinstance(expr, Shift):
        _check_offset(expr, env)
        return _partial(expr.expr, env)
    if isinstance(expr, Apply):
        return _partial_apply(expr, env)
    return shape_of(expr)


def _check_offset(shift: Shift, env: Env) -> None:
    offset = _partial(shift.offset, env)
    expected: Shape = () if shift.axis.dim == 1 else (shift.axis.dim,)
    if offset is not None and offset != expected:
        raise ShapeError(f"In {shift}: the offset needs shape {expected}, got {offset}.")


def _partial_apply(expr: Apply, env: Env) -> Shape | None:
    shapes = tuple(_partial(arg, env) for arg in expr.args[: expr.op.arity])
    if any(shape is None for shape in shapes):
        return None
    axes = expr.args[expr.op.arity] if expr.op.takes_axes else None
    try:
        return expr.op.shape_rule(shapes, axes)
    except ShapeError as error:
        raise ShapeError(f"In {expr}: {error}") from error


def _assign(symbol: FunctionSymbol, shape: Shape, env: Env) -> None:
    if symbol.shape is not None:
        return
    if env.setdefault(symbol, shape) != shape:
        raise ShapeError(
            f"{symbol.name!r} is used with shapes {env[symbol]} and {shape}."
        )


def _back(expr: Expr, target: Shape, env: Env) -> None:
    """Record what `expr` having shape `target` says about the symbols inside it."""
    if isinstance(expr, FunctionSymbol):
        _assign(expr, target, env)
    elif isinstance(expr, (Restrict, Integral, Shift)):
        _back(expr.expr, target, env)
    elif isinstance(expr, Apply) and expr.op.name in _BACKWARD:
        _BACKWARD[expr.op.name](expr, target, env)


def _back_same(expr: Apply, target: Shape, env: Env) -> None:
    _back(expr.args[0], target, env)


def _back_grad(expr: Apply, target: Shape, env: Env) -> None:
    dim = expr.args[1].dim
    if not target or target[-1] != dim:
        raise ShapeError(
            f"In {expr}: a gradient along {dim} ends in {dim}, got {target}."
        )
    _back(expr.args[0], target[:-1], env)


def _back_div(expr: Apply, target: Shape, env: Env) -> None:
    _back(expr.args[0], target + (expr.args[1].dim,), env)


def _back_sum(expr: Apply, target: Shape, env: Env) -> None:
    # With the other side scalar, or a scalar result, the unknown side is the result.
    # Otherwise it could be a broadcast scalar, so nothing is decided.
    for arg, other in _each_with_other(expr):
        shape = _partial(other, env)
        if (
            shape is not None
            and _partial(arg, env) is None
            and (shape == () or target == ())
        ):
            _back(arg, target, env)


def _back_product(expr: Apply, target: Shape, env: Env) -> None:
    # One factor is a scalar: the unknown one is the result if the other is scalar,
    # and a scalar if the other is not.
    for arg, other in _each_with_other(expr):
        shape = _partial(other, env)
        if shape is not None and _partial(arg, env) is None:
            _back(arg, target if shape == () else (), env)


def _back_inner(expr: Apply, _target: Shape, env: Env) -> None:
    for arg, other in _each_with_other(expr):
        shape = _partial(other, env)
        if shape is not None and _partial(arg, env) is None:
            _back(arg, shape, env)


def _each_with_other(expr: Apply) -> tuple[tuple[Expr, Expr], tuple[Expr, Expr]]:
    first, second = expr.args[:2]
    return (first, second), (second, first)


_BACKWARD: dict[str, Callable[[Apply, Shape, Env], None]] = {
    "neg": _back_same,
    "d": _back_same,
    "lap": _back_same,
    "grad": _back_grad,
    "div": _back_div,
    "add": _back_sum,
    "sub": _back_sum,
    "mul": _back_product,
    "inner": _back_inner,
}


def _check(constraint: Constraint, env: Env) -> None:
    left, right = _partial(constraint.lhs, env), _partial(constraint.rhs, env)
    if (
        left == right
        or _scalar_const(constraint.lhs, left)
        or _scalar_const(constraint.rhs, right)
    ):
        return
    raise ShapeError(f"In {constraint}: the sides have shapes {left} and {right}.")
