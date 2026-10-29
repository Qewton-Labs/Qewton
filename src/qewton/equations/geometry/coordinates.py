"""Coordinates: config Variables with a positive integer dimension."""

from __future__ import annotations

from qewton.config.variables import Variable


def check_coordinate(var: object) -> Variable:
    """Return `var` if it can be a coordinate, otherwise raise ValueError.

    Missing dims and tuple dims (several coordinates at once) are rejected.
    """
    if not isinstance(var, Variable):
        raise ValueError(f"Expected a Variable, got {type(var).__name__}.")
    if not isinstance(var.dim, int) or var.dim < 1:
        raise ValueError(
            f"Coordinate {var.name!r} needs a positive integer dim, got dim={var.dim!r}."
        )
    return var


def check_coordinate_product(var: object) -> Variable:
    """Return `var` if it is one coordinate or a product of coordinates, e.g. x * t.

    The leaves are the coordinate components, in order. Anything else raises ValueError.
    """
    if not isinstance(var, Variable):
        raise ValueError(f"Expected a Variable, got {type(var).__name__}.")
    if not isinstance(var.dim, int) or var.dim < 1:
        raise ValueError(f"{var.name!r} is not a coordinate or a product of coordinates.")
    if any(leaf.dim != 1 for leaf in var.leaves):
        raise ValueError(f"{var.name!r} is not a coordinate or a product of coordinates.")
    return var


def _is_atomic(var: Variable) -> bool:
    """A single coordinate: no children, or only its own components x_1, x_2, ...

    A product such as x * t has whole variables as children instead. The `parent`
    attribute cannot tell the two apart, since a variable's parent is overwritten
    by every product it is multiplied into.
    """
    names = [child.name for child in var.children]
    return not names or names == [f"{var.name}_{i + 1}" for i in range(var.dim)]


def root_of(var: Variable) -> Variable:
    """The coordinate a component belongs to: x for x_1, and `var` itself otherwise."""
    parent = var.parent
    if parent is not None and _is_atomic(parent):
        if any(child is var for child in parent.children):
            return parent
    return var


def coordinate_factors(var: Variable) -> tuple[Variable, ...]:
    """The coordinates a product is made of: (x, t) for x * t, and (x,) for x.

    A component such as x_1 stands for its coordinate x.
    """
    if _is_atomic(var):
        return (root_of(var),)
    return tuple(factor for child in var.children for factor in coordinate_factors(child))
