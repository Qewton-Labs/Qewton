"""`Expr`: the base class of every expression node.

A node class says in its own methods what it is made of and what follows from that:

- structure: `children`, `with_children`, `sets`, `coordinates`;
- its output shape: `_shape(shapes)`, read through `shape` and `shape_with`;
- the coordinates it varies with: `_free_axes()`, read through `free_axes`;
- how it depends on given unknowns: `linearity(unknowns)`;
- what a known result shape says about its operands, for shape inference:
  `require_shape` and `require_operand_shapes`.

The base class has a default for everything except `_shape`, so a new node class is
complete when it defines its own; nothing is registered anywhere else. Building an
operator node checks its shape and axes at once (`_check`).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING

from qewton.config.variables import Variable
from qewton.equations.expr.ops import OPS, Linearity, OpLinearity, OpSpec, Shape, Solver
from qewton.equations.geometry.sets import SetNode
from qewton.equations.printable import Printable

if TYPE_CHECKING:
    from qewton.equations.expr.leaves import FunctionSymbol

Axes = frozenset[Variable]
SymbolShapes = Mapping["FunctionSymbol", Shape]


def axes_text(axes: Axes) -> str:
    """Coordinates as messages show them: {t, x}."""
    return "{" + ", ".join(sorted(str(axis.name) for axis in axes)) + "}"


def _build(op: OpSpec, *operands: object) -> Expr:
    """`Apply(op, operands)`, with Python numbers turned into constants."""
    # Apply and the leaves are subclasses of Expr, so they are imported on use.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from qewton.equations.expr.apply import Apply

    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from qewton.equations.expr.leaves import as_expr

    return Apply(op, tuple(as_expr(operand) for operand in operands))


class Expr(Printable):
    """Base class of expressions: the sides of equations, conditions and so on."""

    # region Structure
    def children(self) -> tuple[Expr, ...]:
        """The direct sub-expressions. Coordinates and sets are not included."""
        return ()

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        """This node rebuilt around new sub-expressions, in the order of `children()`."""
        if new:
            raise ValueError(f"{type(self).__name__} has no sub-expressions.")
        return self

    def sets(self) -> tuple[SetNode, ...]:
        """The sets this node itself refers to, not those of its sub-expressions."""
        return ()

    def coordinates(self) -> tuple[Variable, ...]:
        """The coordinates this node itself uses, not those of its sub-expressions."""
        return ()

    # endregion

    # region Shape
    def _shape(self, shapes: SymbolShapes) -> Shape | None:
        """The output shape of this node; every node class defines it.

        `shapes` gives the shapes of symbols whose shape is not declared. Returns
        None while the shape of a symbol in the node is unknown, and raises
        ShapeError where known shapes do not fit together.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define its shape.")

    @cached_property
    def shape(self) -> Shape | None:
        """The output shape from the declared shapes alone: () for a scalar, (n,) for
        a vector, None while the shape of a symbol in it is not declared."""
        return self._shape({})

    def shape_with(self, shapes: SymbolShapes) -> Shape | None:
        """The output shape when the undeclared symbols have the given `shapes`."""
        known = self.shape
        if known is not None or not shapes:
            return known
        return self._shape(shapes)

    # endregion

    # region Coordinates
    def _free_axes(self) -> Axes:
        """The coordinates this node varies with. By default those of its children;
        a node that adds or removes coordinates overrides this."""
        return frozenset().union(*(child.free_axes for child in self.children()))

    @cached_property
    def free_axes(self) -> Axes:
        """The coordinates the value of this expression varies with, e.g. {x, t}.

        A function depends on its declared inputs, a constant on none; restricting to
        a slice removes the sliced coordinate, integrating over a set its coordinates.
        """
        return self._free_axes()

    # endregion

    # region Dependence on unknowns
    def linearity(self, unknowns: frozenset[FunctionSymbol]) -> Linearity:
        """How this expression depends on `unknowns`. By default linearly through its
        children, which fits restrictions, integrals and components."""
        parts = [child.linearity(unknowns) for child in self.children()]
        return OpLinearity.LINEAR.of_result(parts)

    # endregion

    # region Shape inference
    def require_shape(self, target: Shape, solver: Solver) -> None:
        """Record what this node having shape `target` says about the symbols in it.

        By default nothing: the result does not determine the operands.
        """

    def require_operand_shapes(self, solver: Solver) -> None:
        """Record the shapes this node needs of its operands, whatever its result.

        By default none.
        """

    # endregion

    def _check(self) -> None:
        """Check the shape and the coordinates as far as they are known; operator
        nodes call this when they are built."""
        _ = self.shape
        _ = self.free_axes

    # region Building expressions
    def __add__(self, other):
        return _build(OPS.add, self, other)

    def __radd__(self, other):
        return _build(OPS.add, other, self)

    def __sub__(self, other):
        return _build(OPS.sub, self, other)

    def __rsub__(self, other):
        return _build(OPS.sub, other, self)

    def __mul__(self, other):
        return _build(OPS.mul, self, other)

    def __rmul__(self, other):
        return _build(OPS.mul, other, self)

    # Division is multiplication by a power of -1, so there is no separate operator.
    def __truediv__(self, other):
        return self * _build(OPS.pow, other, -1)

    def __rtruediv__(self, other):
        return _build(OPS.mul, other, self**-1)

    def __pow__(self, other):
        return _build(OPS.pow, self, other)

    def __rpow__(self, other):
        return _build(OPS.pow, other, self)

    def __neg__(self):
        return _build(OPS.neg, self)

    # Without this, Python would try to iterate over `u` through `__getitem__`.
    __iter__ = None

    def __getitem__(self, index: int | tuple[int, ...]) -> Expr:
        """Components of a vector or tensor: `u[0]`, or `A[0, 1]` for `A[0][1]`."""
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from qewton.equations.expr.wrappers import Index

        result = self
        for position in index if isinstance(index, tuple) else (index,):
            result = Index(result, position)
        return result

    # endregion
