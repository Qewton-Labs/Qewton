"""The leaves of the expression tree: functions, coordinates, constants and normals."""

from __future__ import annotations

from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.config.data_configurations.base import DataConfiguration
from qewton.equations.expr.base import Axes, Expr, SymbolConfigs
from qewton.equations.expr.ops import Linearity, Shape, ShapeError, fixed, unknown
from qewton.equations.geometry.coordinates import (
    check_coordinate,
    check_coordinate_product,
    coordinate_factors,
)
from qewton.equations.geometry.sets import SetNode, has_normal, set_free_axes


def _check_output(variable: Variable) -> None:
    dim = variable.dim
    if dim is None:
        return
    dims = dim if isinstance(dim, tuple) else (dim,)
    if not all(isinstance(d, int) and d >= 1 for d in dims):
        raise ValueError(
            f"The output {variable.name!r} needs positive dims, got {dim!r}."
        )


def _check_inputs(inputs: Variable) -> None:
    if inputs.dim == 0:
        raise ValueError("A constant has inputs=None, not a zero-dimensional Variable.")
    check_coordinate_product(inputs)


def _freeze(value: object) -> float | tuple:
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise TypeError(f"A constant must be numbers in nested lists, got {value!r}.")
    if isinstance(value, (int, float)):
        return value
    if not value:
        raise ValueError("A constant vector or matrix cannot be empty.")
    return tuple(_freeze(item) for item in value)


def _value_shape(value: int | float | tuple) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        return ()
    shapes = {_value_shape(item) for item in value}
    if len(shapes) != 1:
        raise ValueError(f"A constant must be rectangular, got {value!r}.")
    return (len(value),) + shapes.pop()


def _dim_shape(dim: int | tuple[int, ...] | None) -> Shape:
    """The shape a dim stands for: () for 1 or None, (n,) for n, a tuple as it is."""
    if isinstance(dim, tuple):
        return dim
    return () if dim in (1, None) else (dim,)


@dataclass(frozen=True, eq=False)
class FunctionSymbol(Expr):
    """A function such as u or f. Two symbols are different functions even if named alike.

    `variable` gives the name and the output: dim 1 is a scalar, n a vector of n
    components, a tuple a tensor, and None a shape the analysis infers.
    `inputs` is always given: the product of the coordinates the function depends
    on, e.g. x * t, in the order of the input layout. None means the function is a
    constant. Inputs are never inferred: a function that is not differentiated along
    x may be constant, or may depend on x, and usage cannot tell the two apart.
    """

    variable: Variable
    inputs: Variable | None

    def __post_init__(self):
        if not isinstance(self.variable, Variable):
            raise TypeError(f"Expected a Variable, got {type(self.variable).__name__}.")
        _check_output(self.variable)
        if self.inputs is not None:
            if not isinstance(self.inputs, Variable):
                raise TypeError(
                    "Inputs must be a Variable such as x * t, got "
                    f"{type(self.inputs).__name__}."
                )
            _check_inputs(self.inputs)

    @property
    def name(self) -> str | None:
        return self.variable.name

    def coordinates(self) -> tuple[Variable, ...]:
        return coordinate_factors(self.inputs) if self.inputs is not None else ()

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        if self.variable.dim is None:
            # One configuration for every occurrence, so they share what is learned.
            return env.setdefault(self, unknown())
        return fixed(_dim_shape(self.variable.dim))

    def _free_axes(self) -> Axes:
        return frozenset(self.coordinates())

    def linearity(self, unknowns: frozenset[FunctionSymbol]) -> Linearity:
        return Linearity.LINEAR if self in unknowns else Linearity.NONE


@dataclass(frozen=True)
class Coord(Expr):
    """The coordinate function x -> x_i, component `component` of `variable`."""

    variable: Variable
    component: int

    def __post_init__(self):
        check_coordinate(self.variable)
        assert isinstance(
            self.variable.dim, int
        ), f"Expected an integer component, got {type(self.component).__name__}."
        if not 0 <= self.component < self.variable.dim:
            raise ValueError(
                f"Component {self.component} is out of range for {self.variable.name!r} "
                f"with dim {self.variable.dim}."
            )

    def coordinates(self) -> tuple[Variable, ...]:
        return (self.variable.whole,)

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        return fixed(())

    def _free_axes(self) -> Axes:
        return frozenset(self.coordinates())


@dataclass(frozen=True)
class Const(Expr):
    """A number, vector or matrix with fixed values, such as -Δu = 1.0 or g = (0, -9.81).

    Lists are stored as tuples, so constants hash and compare by value. The shape
    follows the nesting: () for a number, (n,) for a vector, (n, m) for a matrix.
    Data supplied later, such as a coefficient matrix, is a FunctionSymbol instead.
    """

    value: float | tuple

    def __post_init__(self):
        # Frozen dataclass: store the normalized value.
        object.__setattr__(self, "value", _freeze(self.value))
        _value_shape(self.value)  # raises for ragged nesting

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        return fixed(_value_shape(self.value))


@dataclass(frozen=True)
class Normal(Expr):
    """The outward unit normal of a boundary, or of an interface (first to second)."""

    of: SetNode

    def __post_init__(self):
        if not has_normal(self.of):
            raise ValueError(
                "A normal only exists on a boundary, a part of one, or an interface."
            )

    def sets(self) -> tuple[SetNode, ...]:
        return (self.of,)

    def _config(self, env: SymbolConfigs) -> DataConfiguration:
        axes = self.of.axes
        if len(axes) != 1:
            raise ShapeError(f"A normal needs a single coordinate, got {len(axes)}.")
        (axis,) = axes
        dim = axis.dim
        if isinstance(dim, tuple):
            return fixed(dim)
        return fixed((dim,) if dim is not None else ())

    def _free_axes(self) -> Axes:
        return set_free_axes(self.of)


def as_expr(value: object) -> Expr:
    """Return `value` as an expression.

    Python numbers, lists and tuples become a Const. A one-dimensional coordinate
    becomes its Coord.
    """
    if isinstance(value, Expr):
        return value
    if isinstance(value, Variable):
        if value.dim != 1:
            raise TypeError(
                f"{value.name!r} has dim {value.dim!r}; use Coord({value.name}, i) "
                "for one of its components."
            )
        return Coord(value, 0)
    if isinstance(value, (int, float, list, tuple)) and not isinstance(value, bool):
        return Const(value) if not isinstance(value, list) else Const(tuple(value))
    raise TypeError(f"Cannot use {value!r} in an expression.")
