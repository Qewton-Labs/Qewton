"""`Apply`: an operator from the registry (`ops.py`) applied to its arguments."""

from __future__ import annotations

from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.config.data_configurations.base import DataConfiguration
from qewton.equations.expr.base import Axes, Expr, SymbolConfigs, axes_text
from qewton.equations.expr.ops import AxisError, Linearity, OpSpec, ShapeError
from qewton.equations.geometry.coordinates import (
    check_coordinate_product,
    coordinate_factors,
)


@dataclass(frozen=True)
class Apply(Expr):
    """An operator applied to its arguments, e.g. grad(u, x) or u + 1.

    `args` are the expression arguments in order. An operator that takes derivative
    axes has one more argument at the end: a coordinate product, e.g. x or x * t.
    What the operator does with shapes, orders and linearity is data of its `OpSpec`.
    """

    op: OpSpec
    args: tuple[Expr | Variable, ...]

    def __post_init__(self):
        if not isinstance(self.op, OpSpec):
            raise TypeError(f"Expected an OpSpec, got {type(self.op).__name__}.")
        arity = self.op.arity
        expected = arity + (1 if self.op.takes_axes else 0)
        if len(self.args) != expected:
            raise ValueError(
                f"{self.op.name} takes {expected} arguments, got {len(self.args)}."
            )
        for arg in self.args[:arity]:
            if not isinstance(arg, Expr):
                raise TypeError(
                    f"{self.op.name} takes expressions, got {type(arg).__name__}."
                )
        if self.op.takes_axes:
            check_coordinate_product(self.args[arity])
        self._check()

    @property
    def operands(self) -> tuple[Expr, ...]:
        """The expression arguments, without the derivative axes."""
        return self.args[: self.op.arity]  # type: ignore[return-value]

    @property
    def axes(self) -> Variable | None:
        """The derivative axes, e.g. x for grad(u, x), or None."""
        return self.args[self.op.arity] if self.op.takes_axes else None  # type: ignore

    def coordinates(self) -> tuple[Variable, ...]:
        return () if self.axes is None else coordinate_factors(self.axes)

    def children(self) -> tuple[Expr, ...]:
        return self.operands

    def with_children(self, new: tuple[Expr, ...]) -> Expr:
        # The derivative axes are not expressions and stay where they are.
        rest = () if self.axes is None else (self.axes,)
        return Apply(self.op, tuple(new) + rest)

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        configs = [arg.config(env) for arg in self.operands]
        try:
            return self.op.relate(configs, self.axes)
        except ShapeError as error:
            raise ShapeError(f"In {self}: {error}") from error

    def _free_axes(self) -> Axes:
        result = super()._free_axes()
        if self.axes is not None:
            missing = frozenset(coordinate_factors(self.axes)) - result
            if missing:
                raise AxisError(
                    f"Cannot differentiate along {axes_text(missing)} in {self}: the "
                    f"operand depends on {axes_text(result)}, so the derivative is zero."
                )
        return result

    def linearity(self, unknowns) -> Linearity:
        parts = [arg.linearity(unknowns) for arg in self.operands]
        return self.op.linearity.of_result(parts)
