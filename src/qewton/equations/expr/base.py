"""`Expr`: the base class of every expression node.

A node class says in its own methods what it is made of and what follows from that:

- structure: `children`, `with_children`, `sets`, `coordinates`;
- its output shape: `_config(env)`, read through `shape`, `shape_with` and `config`;
- the coordinates it varies with: `_free_axes()`, read through `free_axes`;
- how it depends on given unknowns: `linearity(unknowns)`.

The base class has a default for everything except `_config`, so a new node class is
complete when it defines its own; nothing is registered anywhere else. Building an
operator node checks its shape and axes at once (`_check`).

`_config` describes the shape as a data configuration (see `ops.py`), which relates
it to the shapes of the sub-expressions in both directions: shape inference
(`analysis/prepare/infer.py`) learns undeclared shapes from that alone.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from functools import cached_property
from typing import TYPE_CHECKING

from qewton.config.variables import Variable
from qewton.config.data_configurations.base import DataConfiguration
from qewton.config.errors import StructureUnknown
from qewton.equations.expr.ops import (
    OPS,
    Linearity,
    OpLinearity,
    OpSpec,
    Shape,
    fixed,
    known_shape,
)
from qewton.equations.geometry.sets import SetNode
from qewton.equations.printable import Printable

if TYPE_CHECKING:
    from qewton.equations.expr.leaves import FunctionSymbol

Axes = frozenset[Variable]
SymbolShapes = Mapping["FunctionSymbol", Shape]
# One configuration per symbol whose shape is not declared, shared by its occurrences.
SymbolConfigs = MutableMapping["FunctionSymbol", DataConfiguration]


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
    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        """The output shape of this node as a data configuration, related to the
        configurations of its sub-expressions; every node class defines it.

        `env` holds a configuration for each symbol whose shape is not declared.
        Raises ShapeError where the shapes contradict, and StructureUnknown while an
        operator cannot decide yet (see `ops.Relation`).
        """
        raise NotImplementedError(f"{type(self).__name__} does not define its shape.")

    def config(self, env: SymbolConfigs) -> DataConfiguration:
        """The output shape as a data configuration; see `_config`."""
        known = self.shape
        return fixed(known) if known is not None else self._config(env)

    @cached_property
    def shape(self) -> Shape | None:
        """The output shape from the declared shapes alone: () for a scalar, (n,) for
        a vector, None while the shape of a symbol in it is not declared."""
        if any(child.shape is None for child in self.children()):
            return None
        return known_shape(self._config({}))

    def shape_with(self, shapes: SymbolShapes) -> Shape | None:
        """The output shape when the undeclared symbols have the given `shapes`."""
        known = self.shape
        if known is not None or not shapes:
            return known
        env = {symbol: fixed(shape) for symbol, shape in shapes.items()}
        try:
            return known_shape(self._config(env))
        except StructureUnknown:
            return None

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
