from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from qewton.config.variables import Variable
from qewton.equations.expr.ops import OPS, OpSpec
from qewton.equations.printable import Printable
from qewton.equations.geometry.coordinates import (
    check_coordinate,
    check_coordinate_product,
)
from qewton.equations.geometry.sets import Boundary, Part, SetNode


class Expr(Printable):
    """Base class of expressions.

    Subclasses are frozen dataclasses, so they compare and hash by their fields.
    This base is not a dataclass on purpose: a generated __eq__ here would be
    inherited by FunctionSymbol, which must compare by identity.
    """

    def __add__(self, other):
        return Apply(OPS["add"], (self, as_expr(other)))

    def __radd__(self, other):
        return Apply(OPS["add"], (as_expr(other), self))

    def __sub__(self, other):
        return Apply(OPS["sub"], (self, as_expr(other)))

    def __rsub__(self, other):
        return Apply(OPS["sub"], (as_expr(other), self))

    def __mul__(self, other):
        return Apply(OPS["mul"], (self, as_expr(other)))

    def __rmul__(self, other):
        return Apply(OPS["mul"], (as_expr(other), self))

    # Division is multiplication by a power of -1, so there is no separate operator.
    def __truediv__(self, other):
        return self * as_expr(other) ** -1

    def __rtruediv__(self, other):
        return as_expr(other) * self**-1

    def __pow__(self, other):
        return Apply(OPS["pow"], (self, as_expr(other)))

    def __rpow__(self, other):
        return Apply(OPS["pow"], (as_expr(other), self))

    def __neg__(self):
        return Apply(OPS["neg"], (self,))


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
        return Const(value)
    raise TypeError(f"Cannot use {value!r} in an expression.")


def _is_boundary(of: SetNode) -> bool:
    return isinstance(of, Boundary) or (isinstance(of, Part) and _is_boundary(of.of))


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
    def name(self) -> str:
        return self.variable.name

    @property
    def shape(self) -> tuple[int, ...] | None:
        """The output shape: () for a scalar, (n,) for a vector, None if not yet known."""
        dim = self.variable.dim
        if dim is None:
            return None
        if isinstance(dim, tuple):
            return dim
        return () if dim == 1 else (dim,)


@dataclass(frozen=True)
class Coord(Expr):
    """The coordinate function x -> x_i, component `component` of `variable`."""

    variable: Variable
    component: int

    def __post_init__(self):
        check_coordinate(self.variable)
        if not 0 <= self.component < self.variable.dim:
            raise ValueError(
                f"Component {self.component} is out of range for {self.variable.name!r} "
                f"with dim {self.variable.dim}."
            )


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
        _shape_of(self.value)  # raises for ragged nesting

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape_of(self.value)


def _freeze(value: object) -> float | tuple:
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise TypeError(f"A constant must be numbers in nested lists, got {value!r}.")
    if isinstance(value, (int, float)):
        return value
    if not value:
        raise ValueError("A constant vector or matrix cannot be empty.")
    return tuple(_freeze(item) for item in value)


def _shape_of(value: float | tuple) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        return ()
    shapes = {_shape_of(item) for item in value}
    if len(shapes) != 1:
        raise ValueError(f"A constant must be rectangular, got {value!r}.")
    return (len(value),) + shapes.pop()


@dataclass(frozen=True)
class Normal(Expr):
    """The outward unit normal of a boundary."""

    of: SetNode

    def __post_init__(self):
        if not _is_boundary(self.of):
            raise ValueError("A normal only exists on a boundary or a part of one.")


@dataclass(frozen=True)
class Restrict(Expr):
    """`expr` evaluated on the set `on`, e.g. the trace of u on Gamma_D.

    Whether `on` is lower-dimensional than `expr` needs the signatures, so the
    validate and infer passes check it.
    """

    expr: Expr
    on: SetNode

    def __post_init__(self):
        if not isinstance(self.expr, Expr):
            raise TypeError(f"Expected an expression, got {type(self.expr).__name__}.")
        if not isinstance(self.on, SetNode):
            raise TypeError(f"Expected a set, got {type(self.on).__name__}.")


@dataclass(frozen=True)
class Apply(Expr):
    """An operator applied to its arguments, e.g. grad(u, x) or u + 1.

    `args` are the expression arguments in order. An operator that takes derivative
    axes has one more argument at the end: a coordinate product, e.g. x or x * t.
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


def children(expr: Expr) -> tuple[Expr, ...]:
    """The direct sub-expressions of `expr`. Coordinates and sets are excluded."""
    if isinstance(expr, Apply):
        return tuple(arg for arg in expr.args if isinstance(arg, Expr))
    if isinstance(expr, Restrict):
        return (expr.expr,)
    return ()


def walk(expr: Expr) -> Iterator[Expr]:
    """Yield `expr` and all its sub-expressions, depth first."""
    yield expr
    for child in children(expr):
        yield from walk(child)
