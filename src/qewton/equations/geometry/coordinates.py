"""Coordinates are config `Variable`s with a positive integer dim.

What a variable is made of comes from `Variable` itself: `x[0].whole` is x, and
`(x * t).quantities` is (x, t). This module adds what the equations require of a
coordinate, and `coordinate_factors`, which stands each component for its coordinate.
"""

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


def coordinate_factors(var: Variable) -> tuple[Variable, ...]:
    """The coordinates a product is made of: (x, t) for x * t, and (x,) for x.

    A component such as x_1 stands for its coordinate x.
    """
    return tuple(quantity.whole for quantity in var.quantities)
