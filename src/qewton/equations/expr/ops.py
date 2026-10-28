"""Operator registry: operators as data.

Each OpSpec says how many expression arguments an operator takes, whether it takes
derivative axes after them, how the result shape follows from the argument shapes,
its derivative order and its linearity. Nodes refer to these specs; nothing here
depends on the nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from qewton.config.variables import Variable

Shape = tuple[int, ...]


class ShapeError(ValueError):
    """Raised when the arguments of an operator have incompatible shapes."""


@dataclass(frozen=True)
class OpSpec:
    """The semantics of one operator.

    Attributes:
        name: The registry key, e.g. "grad".
        arity: The number of expression arguments.
        shape_rule: Maps the argument shapes and the derivative axes (or None) to
            the result shape. Raises ShapeError on incompatible shapes.
        takes_axes: Whether a coordinate product follows the expression arguments.
        order: The derivative order along each leaf of the axes argument.
        linearity: How the operator depends on its arguments.
    """

    name: str
    arity: int
    shape_rule: Callable[[tuple[Shape, ...], Variable | None], Shape]
    takes_axes: bool = False
    order: int = 0
    linearity: Literal["linear", "multilinear", "nonlinear"] = "linear"


OPS: dict[str, OpSpec] = {}


def register(spec: OpSpec) -> OpSpec:
    if spec.name in OPS:
        raise ValueError(f"Operator {spec.name!r} is already registered.")
    OPS[spec.name] = spec
    return spec


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
    if axes.dim != 1:
        raise ShapeError("d needs a single coordinate; use grad for several.")
    return shapes[0]


def _grad(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    return shapes[0] + (axes.dim,)


def _div(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    shape = shapes[0]
    if not shape or shape[-1] != axes.dim:
        raise ShapeError(
            f"div needs a last axis of length {axes.dim}, got shape {shape}."
        )
    return shape[:-1]


def _curl(shapes: tuple[Shape, ...], axes: Variable | None) -> Shape:
    if shapes[0] != (3,) or axes.dim != 3:
        raise ShapeError("curl needs a 3-vector and three coordinates.")
    return (3,)


register(OpSpec("add", 2, _elementwise))
register(OpSpec("sub", 2, _elementwise))
register(OpSpec("neg", 1, _same_shape))
register(OpSpec("mul", 2, _mul, linearity="multilinear"))
register(OpSpec("dot", 2, _contract, linearity="multilinear"))
register(OpSpec("inner", 2, _inner, linearity="multilinear"))
register(OpSpec("pow", 2, _scalar_only, linearity="nonlinear"))
register(OpSpec("exp", 1, _scalar_only, linearity="nonlinear"))
register(OpSpec("sin", 1, _scalar_only, linearity="nonlinear"))
register(OpSpec("cos", 1, _scalar_only, linearity="nonlinear"))
register(OpSpec("d", 1, _partial, takes_axes=True, order=1))
register(OpSpec("grad", 1, _grad, takes_axes=True, order=1))
register(OpSpec("div", 1, _div, takes_axes=True, order=1))
register(OpSpec("curl", 1, _curl, takes_axes=True, order=1))
register(OpSpec("lap", 1, _same_shape, takes_axes=True, order=2))
