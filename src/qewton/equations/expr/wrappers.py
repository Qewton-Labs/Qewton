"""Nodes that wrap one expression: evaluate it on a set, integrate it, shift it, or
take one of its components."""

from __future__ import annotations

from dataclasses import dataclass, replace

from qewton.config.variables import Variable
from qewton.config.data_configurations.base import DataConfiguration
from qewton.equations.expr.base import Axes, Expr, SymbolConfigs, axes_text
from qewton.equations.expr.leaves import Coord, FunctionSymbol, as_expr
from qewton.equations.expr.ops import (
    AxisError,
    Linearity,
    Shape,
    ShapeError,
    describe,
    fixed,
    known_rank,
    unify,
)
from qewton.equations.expr.traversal import walk
from qewton.equations.geometry.coordinates import check_coordinate
from qewton.equations.geometry.sets import SetNode, sliced_axes


def _check_expr(value: object) -> None:
    if not isinstance(value, Expr):
        raise TypeError(f"Expected an expression, got {type(value).__name__}.")


def _check_set(value: object) -> None:
    if not isinstance(value, SetNode):
        raise TypeError(f"Expected a set, got {type(value).__name__}.")


@dataclass(frozen=True)
class Restrict(Expr):
    """`expr` evaluated on the set `on`, e.g. the trace of u on Gamma_D."""

    expr: Expr
    on: SetNode

    def __post_init__(self):
        _check_expr(self.expr)
        _check_set(self.on)

    def sets(self) -> tuple[SetNode, ...]:
        return (self.on,)

    def children(self) -> tuple[Expr, ...]:
        return (self.expr,)

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        return replace(self, expr=new[0])

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        return self.expr.config(env)

    def _free_axes(self) -> Axes:
        # A slice fixes its coordinate; any other set keeps the coordinates.
        return self.expr.free_axes - sliced_axes(self.on)


@dataclass(frozen=True)
class Integral(Expr):
    """The integral of `expr` over the set `over`, e.g. the mean of a pressure."""

    expr: Expr
    over: SetNode

    def __post_init__(self):
        _check_expr(self.expr)
        _check_set(self.over)

    def sets(self) -> tuple[SetNode, ...]:
        return (self.over,)

    def children(self) -> tuple[Expr, ...]:
        return (self.expr,)

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        return replace(self, expr=new[0])

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        return self.expr.config(env)

    def _free_axes(self) -> Axes:
        return self.expr.free_axes - self.over.axes


@dataclass(frozen=True)
class Shift(Expr):
    """`expr` evaluated at a shifted coordinate: u(t - 1), or u(x + (L, 0)).

    `axis` is a coordinate and `offset` is how far it moves: a number for a
    one-dimensional axis, a vector for a multi-component one. The offset is the same
    everywhere, so it may be a constant or a constant function such as a delay tau,
    but not depend on any coordinate. Delays and periodic conditions use this.
    """

    expr: Expr
    axis: Variable
    offset: Expr

    def __post_init__(self):
        _check_expr(self.expr)
        check_coordinate(self.axis)
        object.__setattr__(self, "offset", as_expr(self.offset))
        for node in walk(self.offset):
            if isinstance(node, Coord) or (
                isinstance(node, FunctionSymbol) and node.inputs is not None
            ):
                raise ValueError(
                    "A shift offset must be constant: it cannot depend on a coordinate."
                )
        self._check()

    @property
    def offset_shape(self) -> Shape:
        """The shape the offset needs: a number for a one-dimensional axis, a vector
        of its length otherwise."""
        dim = self.axis.dim
        if isinstance(dim, tuple):
            return dim
        return () if dim == 1 or not dim else (dim,)

    def coordinates(self) -> tuple[Variable, ...]:
        return (self.axis.whole,)

    def children(self) -> tuple[Expr, ...]:
        return (self.expr, self.offset)

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        return Shift(new[0], self.axis, new[1])

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        offset = self.offset.config(env)
        expected = self.offset_shape
        message = f"In {self}: the offset needs shape {expected}, got {describe(offset)}."
        unify(offset, fixed(expected), message)
        return self.expr.config(env)

    def _free_axes(self) -> Axes:
        operand = self.expr.free_axes
        axis = self.axis.whole
        if axis not in operand:
            raise AxisError(
                f"Cannot shift along {axis.name!r}: {self.expr} depends on "
                f"{axes_text(operand)}."
            )
        return operand | self.offset.free_axes

    def linearity(self, unknowns: frozenset[FunctionSymbol]) -> Linearity:
        # A delay that depends on an unknown makes the term nonlinear.
        if self.offset.linearity(unknowns) is not Linearity.NONE:
            return Linearity.NONLINEAR
        return self.expr.linearity(unknowns)


@dataclass(frozen=True)
class Index(Expr):
    """Component `index` of the first output axis of `expr`: u[0], or A[0][1].

    It removes the leading axis of the shape, so a vector gives a scalar and a matrix
    gives a row. The component must exist, which is checked once the shape is known.
    Several axes are indexed one after another, and `A[0, 1]` builds that chain.
    """

    expr: Expr
    index: int

    def __post_init__(self):
        _check_expr(self.expr)
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError(
                f"Expected an integer index, got {type(self.index).__name__}."
            )
        if self.index < 0:
            raise ValueError(f"Component indices count from 0, got {self.index}.")
        self._check()

    def children(self) -> tuple[Expr, ...]:
        return (self.expr,)

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        return replace(self, expr=new[0])

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        operand = self.expr.config(env)
        if known_rank(operand) == 0:
            raise ShapeError(
                f"Cannot take component {self.index} of the scalar {self.expr}."
            )
        size = operand.dim(0).size
        if size is not None and self.index >= size:
            raise ShapeError(
                f"Component {self.index} is out of range for {self.expr} of shape "
                f"{describe(operand)}."
            )
        return operand.drop(0)
