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
a ShapeError. What a node says about shapes is defined on the node classes
(`Expr.shape_with`, `require_shape`, `require_operand_shapes`); this module only
drives the propagation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from qewton.equations.expr.base import Expr
from qewton.equations.expr.leaves import Const, FunctionSymbol
from qewton.equations.expr.ops import Shape, ShapeError
from qewton.equations.expr.wrappers import Index
from qewton.equations.model.constraints import Constraint
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
    solver = _Solver({})
    defaulted: list[FunctionSymbol] = []
    while True:
        _propagate(problem, solver)
        pending = [
            s
            for s in _symbols_in_order(problem)
            if s.shape is None and s not in solver.env
        ]
        if not pending:
            break
        _reject_indexed(problem, pending[0])
        solver.env[pending[0]] = ()
        defaulted.append(pending[0])
    for constraint in problem.constraints:
        _check(constraint, solver.env)
    shapes = {
        symbol: symbol.shape if symbol.shape is not None else solver.env[symbol]
        for symbol in _symbols_in_order(problem)
    }
    return Inference(shapes, tuple(defaulted))


def _reject_indexed(problem: Problem, symbol: FunctionSymbol) -> None:
    """A component does not say how many components there are, so it cannot default."""
    for constraint in problem.constraints:
        for node in constraint.nodes():
            if isinstance(node, Index) and node.expr == symbol:
                raise ShapeError(
                    f"The output shape of {symbol.name!r} cannot be inferred from "
                    f"{node}; declare it, e.g. Variable({symbol.name!r}, dim=2)."
                )


def constraint_shape(
    constraint: Constraint, shapes: Mapping[FunctionSymbol, Shape]
) -> Shape:
    """The shape of both sides of `constraint`, when the shape of every function is
    given; a scalar constant side fits any shape."""
    left, right = (_determined(side, shapes) for side in constraint.sides)
    if left == right:
        return left
    if left == () and isinstance(constraint.lhs, Const):
        return right
    if right == () and isinstance(constraint.rhs, Const):
        return left
    raise ShapeError(f"Cannot determine scalar equation components for {constraint}.")


def _determined(expr: Expr, shapes: Mapping[FunctionSymbol, Shape]) -> Shape:
    shape = expr.shape_with(shapes)
    if shape is None:
        raise ShapeError(f"The shape of {expr} is not determined by the given shapes.")
    return shape


def _symbols_in_order(problem: Problem) -> list[FunctionSymbol]:
    found = (s for constraint in problem.constraints for s in constraint.symbols())
    return list(dict.fromkeys(found))


def _propagate(problem: Problem, solver: _Solver) -> None:
    """Apply steps 1 and 2 until no more shapes follow."""
    while True:
        known = len(solver.env)
        for constraint in problem.constraints:
            _constrain(constraint.lhs, constraint.rhs, solver)
            for node in constraint.nodes():
                node.require_operand_shapes(solver)
        if len(solver.env) == known:
            return


def _constrain(lhs: Expr, rhs: Expr, solver: _Solver) -> None:
    """Equal sides have equal shapes, except that a scalar constant fits any shape."""
    left, right = solver.known(lhs), solver.known(rhs)
    if left is None and right is not None and not _scalar_const(rhs, right):
        lhs.require_shape(right, solver)
    elif right is None and left is not None and not _scalar_const(lhs, left):
        rhs.require_shape(left, solver)


def _scalar_const(expr: Expr, shape: Shape | None) -> bool:
    return isinstance(expr, Const) and shape == ()


@dataclass(frozen=True)
class _Solver:
    """The shapes inferred so far, and what the nodes may ask of them."""

    env: Env

    def known(self, expr: Expr) -> Shape | None:
        return expr.shape_with(self.env)

    def back(self, expr: Expr, target: Shape) -> None:
        expr.require_shape(target, self)

    def record(self, symbol: FunctionSymbol, shape: Shape) -> None:
        if self.env.setdefault(symbol, shape) != shape:
            raise ShapeError(
                f"{symbol.name!r} is used with shapes {self.env[symbol]} and {shape}."
            )


def _check(constraint: Constraint, env: Env) -> None:
    left, right = (side.shape_with(env) for side in constraint.sides)
    if (
        left == right
        or _scalar_const(constraint.lhs, left)
        or _scalar_const(constraint.rhs, right)
    ):
        return
    raise ShapeError(f"In {constraint}: the sides have shapes {left} and {right}.")
