"""Operator registry: operators as data.

Each OpSpec says how many expression arguments an operator takes, whether it takes
derivative axes after them, how the result shape follows from the argument shapes,
its derivative order and its linearity. It can also say how a known result shape
determines the shapes of the arguments (`back_rule`), which shape inference uses.
Nodes refer to these specs; nothing here depends on the nodes.

`OPS` holds every operator: by name (`OPS["grad"]`, e.g. to load saved expressions),
and, for the built-in ones, as attributes (`OPS.grad`), which is how code refers to
them; see `OperatorRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Protocol

from qewton.config.variables import Variable

Shape = tuple[int, ...]


class ShapeError(ValueError):
    """Raised when the arguments of an operator have incompatible shapes."""


class AxisError(ValueError):
    """Raised when an expression or constraint uses coordinates inconsistently."""


class Linearity(str, Enum):
    """How an expression depends on a set of unknowns.

    - NONE: no unknown occurs.
    - LINEAR: a linear function of the unknowns, e.g. nu * grad(u).
    - AFFINE: linear plus something that does not depend on them, e.g. u + 1.
    - NONLINEAR: anything else, e.g. u * u or sin(u).
    """

    NONE = "none"
    LINEAR = "linear"
    AFFINE = "affine"
    NONLINEAR = "nonlinear"

    def __str__(self) -> str:
        return self.value

    def plus(self, other: Linearity) -> Linearity:
        """The dependence of the sum of two expressions."""
        if Linearity.NONLINEAR in (self, other):
            return Linearity.NONLINEAR
        if self is Linearity.NONE:
            return other
        if other is Linearity.NONE:
            return self
        return Linearity.LINEAR if self is other is Linearity.LINEAR else Linearity.AFFINE


class OpLinearity(str, Enum):
    """How an operator depends on its arguments.

    - LINEAR: linear in every argument jointly, like a sum or a derivative.
    - MULTILINEAR: linear in each argument alone, like a product.
    - NONLINEAR: neither, like sin.
    """

    LINEAR = "linear"
    MULTILINEAR = "multilinear"
    NONLINEAR = "nonlinear"

    def __str__(self) -> str:
        return self.value

    def of_result(self, arguments: list[Linearity]) -> Linearity:
        """The dependence of the result, given that of each expression argument."""
        involved = [part for part in arguments if part is not Linearity.NONE]
        if not involved:
            return Linearity.NONE
        if self is OpLinearity.NONLINEAR:
            return Linearity.NONLINEAR
        if self is OpLinearity.MULTILINEAR:
            return Linearity.NONLINEAR if len(involved) > 1 else involved[0]
        result = Linearity.NONE
        for part in arguments:
            result = result.plus(part)
        # A term without unknowns next to one with is a constant offset.
        if len(involved) < len(arguments) and result is Linearity.LINEAR:
            return Linearity.AFFINE
        return result


class Solver(Protocol):
    """What back rules and the nodes' `require_shape` may ask of shape inference."""

    def known(self, expr: Any) -> Shape | None:
        """The shape of `expr` if it is already determined, else None."""

    def back(self, expr: Any, target: Shape) -> None:
        """Record that `expr` has shape `target`."""

    def record(self, symbol: Any, shape: Shape) -> None:
        """Record that the function `symbol`, whose shape is not declared, has
        `shape`; raises ShapeError if it already has another one."""


# (arguments, axes, result shape, solver) -> records the shapes the arguments need
BackRule = Callable[[tuple[Any, ...], Variable | None, Shape, Solver], None]


@dataclass(frozen=True)
class OpSpec:
    """The semantics of one operator.

    Attributes:
        arity: The number of expression arguments.
        shape_rule: Maps the argument shapes and the derivative axes (or None) to
            the result shape. Raises ShapeError on incompatible shapes.
        takes_axes: Whether a coordinate product follows the expression arguments.
        order: The derivative order along each leaf of the axes argument.
        linearity: How the operator depends on its arguments.
        scalar_args: Whether every expression argument must be a scalar.
        back_rule: Infers argument shapes from a known result shape, or None if the
            result says nothing about the arguments.
        name: The registry key, e.g. "grad". An operator defined as an attribute of
            `OperatorRegistry` takes the attribute's name, so it is not given.
    """

    arity: int
    shape_rule: Callable[[tuple[Shape, ...], Variable | None], Shape]
    takes_axes: bool = False
    order: int = 0
    linearity: OpLinearity = OpLinearity.LINEAR
    scalar_args: bool = False
    back_rule: BackRule | None = None
    name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Take the name of the class attribute this operator is defined as."""
        if self.name and self.name != name:
            raise ValueError(f"The operator {self.name!r} is defined as {name!r}.")
        object.__setattr__(self, "name", name)


###
# Functions for shape rules of operators.
###
def _elementwise(shapes: tuple[Shape, ...], _axes: Variable | None) -> Shape:
    first, second = shapes
    if first == second:
        return first
    if first == ():
        return second
    if second == ():
        return first
    raise ShapeError(f"Cannot combine shapes {first} and {second} elementwise.")


def _same_shape(shapes: tuple[Shape, ...], _axes: Variable | None) -> Shape:
    return shapes[0]


def _scalar_only(shapes: tuple[Shape, ...], _axes: Variable | None) -> Shape:
    if any(shape != () for shape in shapes):
        raise ShapeError(f"Expected scalar arguments, got shapes {shapes}.")
    return ()


def scalar_op(arity: int, name: str = "") -> OpSpec:
    """A nonlinear operator whose arguments and result are all scalars."""
    return OpSpec(
        arity, _scalar_only, linearity=OpLinearity.NONLINEAR, scalar_args=True, name=name
    )


def _mul(shapes: tuple[Shape, ...], _axes: Variable | None) -> Shape:
    first, second = shapes
    if first == ():
        return second
    if second == ():
        return first
    raise ShapeError(
        f"mul needs one scalar factor, got {first} and {second}. "
        "Use dot or inner for two vectors."
    )


def _contract(shapes: tuple[Shape, ...], _axes: Variable | None) -> Shape:
    first, second = shapes
    if not first or not second:
        raise ShapeError(f"dot needs two non-scalar arguments, got {first} and {second}.")
    if first[-1] != second[0]:
        raise ShapeError(
            f"dot contracts the last axis of {first} with the first axis of {second}."
        )
    return first[:-1] + second[1:]


def _inner(shapes: tuple[Shape, ...], _axes: Variable | None) -> Shape:
    first, second = shapes
    if first != second:
        raise ShapeError(f"inner needs equal shapes, got {first} and {second}.")
    return ()


def _partial(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    if axes is not None and axes.dim != 1:
        raise ShapeError("d needs a single coordinate; use grad for several.")
    return shapes[0]


def _grad(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    return shapes[0] + (axes.dim,)  # type: ignore


def _div(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    shape = shapes[0]
    if axes is None:
        raise ShapeError("div needs a coordinate axis.")
    if not shape or shape[-1] != axes.dim:
        raise ShapeError(
            f"div needs a last axis of length {axes.dim}, got shape {shape}."
        )
    return shape[:-1]


def _curl(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    if axes is None:
        raise ShapeError("curl needs a coordinate axis.")
    if shapes[0] != (3,) or axes.dim != 3:
        raise ShapeError("curl needs a 3-vector and three coordinates.")
    return (3,)


###
# Back rules: a known result shape determines (part of) the argument shapes.
###
def _back_same(args: tuple[Any, ...], _axes, target: Shape, solver: Solver) -> None:
    solver.back(args[0], target)


def _back_grad(args: tuple[Any, ...], axes, target: Shape, solver: Solver) -> None:
    if not target or target[-1] != axes.dim:
        raise ShapeError(f"a gradient along {axes.dim} ends in {axes.dim}, got {target}.")
    solver.back(args[0], target[:-1])


def _back_div(args: tuple[Any, ...], axes, target: Shape, solver: Solver) -> None:
    solver.back(args[0], target + (axes.dim,))


def _with_other(args: tuple[Any, ...]):
    first, second = args
    return (first, second), (second, first)


def _back_sum(args: tuple[Any, ...], _axes, target: Shape, solver: Solver) -> None:
    # With the other side scalar, or a scalar result, the unknown side is the result.
    # Otherwise it could be a broadcast scalar, so nothing is decided.
    for arg, other in _with_other(args):
        shape = solver.known(other)
        if (
            shape is not None
            and solver.known(arg) is None
            and (shape == () or not target)
        ):
            solver.back(arg, target)


def _back_product(args: tuple[Any, ...], _axes, target: Shape, solver: Solver) -> None:
    # One factor is a scalar: the unknown one is the result if the other is scalar,
    # and a scalar if the other is not.
    for arg, other in _with_other(args):
        shape = solver.known(other)
        if shape is not None and solver.known(arg) is None:
            solver.back(arg, target if shape == () else ())


def _back_inner(args: tuple[Any, ...], _axes, _target: Shape, solver: Solver) -> None:
    for arg, other in _with_other(args):
        shape = solver.known(other)
        if shape is not None and solver.known(arg) is None:
            solver.back(arg, shape)


class OperatorRegistry(Mapping[str, OpSpec]):
    """Every operator by name. The built-in ones are also attributes: `OPS.grad`.

    Code refers to an operator by its attribute (`expr.op is OPS.lap`), never by its
    name, so pylint and editors flag a misspelled one. A built-in operator is one more
    attribute below, and its name is the attribute's name. An operator defined
    elsewhere is registered once:

        TANH = OPS.register(scalar_op(1, name="tanh"))

    after which it is also `OPS["tanh"]` and `OPS.tanh`.
    """

    add = OpSpec(2, _elementwise, back_rule=_back_sum)
    sub = OpSpec(2, _elementwise, back_rule=_back_sum)
    neg = OpSpec(1, _same_shape, back_rule=_back_same)
    mul = OpSpec(2, _mul, linearity=OpLinearity.MULTILINEAR, back_rule=_back_product)
    dot = OpSpec(2, _contract, linearity=OpLinearity.MULTILINEAR)
    inner = OpSpec(2, _inner, linearity=OpLinearity.MULTILINEAR, back_rule=_back_inner)
    pow = scalar_op(2)
    exp = scalar_op(1)
    sin = scalar_op(1)
    cos = scalar_op(1)
    d = OpSpec(1, _partial, takes_axes=True, order=1, back_rule=_back_same)
    grad = OpSpec(1, _grad, takes_axes=True, order=1, back_rule=_back_grad)
    div = OpSpec(1, _div, takes_axes=True, order=1, back_rule=_back_div)
    curl = OpSpec(1, _curl, takes_axes=True, order=1)
    lap = OpSpec(1, _same_shape, takes_axes=True, order=2, back_rule=_back_same)

    def __init__(self) -> None:
        self._specs: dict[str, OpSpec] = {
            spec.name: spec
            for spec in vars(type(self)).values()
            if isinstance(spec, OpSpec)
        }

    def register(self, spec: OpSpec) -> OpSpec:
        """Add an operator defined outside this class, and return it."""
        if not spec.name:
            raise ValueError("An operator needs a name, e.g. OpSpec(..., name='tanh').")
        if spec.name in self._specs:
            raise ValueError(f"Operator {spec.name!r} is already registered.")
        if hasattr(self, spec.name):
            raise ValueError(f"{spec.name!r} cannot name an operator: OPS uses it.")
        self._specs[spec.name] = spec
        setattr(self, spec.name, spec)
        return spec

    def __getitem__(self, name: str) -> OpSpec:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)


OPS = OperatorRegistry()
