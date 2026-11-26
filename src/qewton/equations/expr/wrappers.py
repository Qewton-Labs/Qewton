"""Nodes that wrap one expression: evaluate it on a set, integrate it, shift it, or
take one of its components."""

from __future__ import annotations

from dataclasses import dataclass, replace

from qewton.config.variables import Variable
from qewton.equations.expr.base import Axes, Expr, SymbolShapes
from qewton.equations.expr.leaves import Coord, FunctionSymbol, as_expr
from qewton.equations.expr.ops import AxisError, Linearity, Shape, ShapeError, Solver
from qewton.equations.geometry.coordinates import check_coordinate
from qewton.equations.geometry.sets import SetNode, sliced_axes


class Wrapper(Expr):
    """A node around one sub-expression, `expr`, the first field of each subclass.

    By default it has the shape of `expr` and passes a known shape on to it; the
    coordinates and the linearity follow `expr` through the defaults of `Expr`.
    """

    expr: Expr

    def __post_init__(self):
        if not isinstance(self.expr, Expr):
            raise TypeError(f"Expected an expression, got {type(self.expr).__name__}.")

    @staticmethod
    def _require_set(value: object) -> None:
        if not isinstance(value, SetNode):
            raise TypeError(f"Expected a set, got {type(value).__name__}.")

    def children(self) -> tuple[Expr, ...]:
        return (self.expr,)

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        return replace(self, expr=new[0])  # type: ignore[type-var]

    def _shape(self, shapes: SymbolShapes) -> Shape | None:
        return self.expr.shape_with(shapes)

    def require_shape(self, target: Shape, solver: Solver) -> None:
        self.expr.require_shape(target, solver)


@dataclass(frozen=True)
class Restrict(Wrapper):
    """`expr` evaluated on the set `on`, e.g. the trace of u on Gamma_D."""

    expr: Expr
    on: SetNode

    def __post_init__(self):
        super().__post_init__()
        self._require_set(self.on)

    def sets(self) -> tuple[SetNode, ...]:
        return (self.on,)

    def _free_axes(self) -> Axes:
        # A slice fixes its coordinate; any other set keeps the coordinates.
        return self.expr.free_axes - sliced_axes(self.on)


@dataclass(frozen=True)
class Integral(Wrapper):
    """The integral of `expr` over the set `over`, e.g. the mean of a pressure."""

    expr: Expr
    over: SetNode

    def __post_init__(self):
        super().__post_init__()
        self._require_set(self.over)

    def sets(self) -> tuple[SetNode, ...]:
        return (self.over,)

    def _free_axes(self) -> Axes:
        return self.expr.free_axes - self.over.axes


@dataclass(frozen=True)
class Shift(Wrapper):
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
        super().__post_init__()
        check_coordinate(self.axis)
        object.__setattr__(self, "offset", as_expr(self.offset))
        for node in self.offset.walk():
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

    def _shape(self, shapes: SymbolShapes) -> Shape | None:
        offset = self.offset.shape_with(shapes)
        expected = self.offset_shape
        if offset is not None and offset != expected:
            raise ShapeError(
                f"In {self}: the offset needs shape {expected}, got {offset}."
            )
        return self.expr.shape_with(shapes)

    def _free_axes(self) -> Axes:
        operand = self.expr.free_axes
        axis = self.axis.whole
        if axis not in operand:
            raise AxisError(
                f"Cannot shift along {axis.name!r}: {self.expr} depends on "
                f"{self._axes_text(operand)}."
            )
        return operand | self.offset.free_axes

    def linearity(self, unknowns: frozenset[FunctionSymbol]) -> Linearity:
        # A delay that depends on an unknown makes the term nonlinear.
        if self.offset.linearity(unknowns) is not Linearity.NONE:
            return Linearity.NONLINEAR
        return self.expr.linearity(unknowns)

    def require_operand_shapes(self, solver: Solver) -> None:
        self.offset.require_shape(self.offset_shape, solver)


@dataclass(frozen=True)
class Index(Wrapper):
    """Component `index` of the first output axis of `expr`: u[0], or A[0][1].

    It removes the leading axis of the shape, so a vector gives a scalar and a matrix
    gives a row. The component must exist, which is checked once the shape is known.
    Several axes are indexed one after another, and `A[0, 1]` builds that chain.
    """

    expr: Expr
    index: int

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError(
                f"Expected an integer index, got {type(self.index).__name__}."
            )
        if self.index < 0:
            raise ValueError(f"Component indices count from 0, got {self.index}.")
        self._check()

    def _shape(self, shapes: SymbolShapes) -> Shape | None:
        operand = self.expr.shape_with(shapes)
        if operand is None:
            return None
        if not operand:
            raise ShapeError(
                f"Cannot take component {self.index} of the scalar {self.expr}."
            )
        if self.index >= operand[0]:
            raise ShapeError(
                f"Component {self.index} is out of range for {self.expr} of shape "
                f"{operand}."
            )
        return operand[1:]

    def require_shape(self, target: Shape, solver: Solver) -> None:
        """Nothing follows: a component does not say how many components there are."""
