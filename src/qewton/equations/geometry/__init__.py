"""Geometry layer: coordinates and the sets they span.

The set classes live in `sets.py`, together with the queries the other layers ask
about them (`boundary_of`, `depends_on`, ...); `coordinates.py` has the helpers for
coordinates, which are config `Variable`s.
"""

from qewton.equations.geometry.sets import (
    Boundary,
    Domain,
    Interface,
    Part,
    Partition,
    Periodic,
    Product,
    SetNode,
    Slice,
    is_subset,
    set_from_geometry,
)

__all__ = [
    "Boundary",
    "Domain",
    "Interface",
    "Part",
    "Partition",
    "Periodic",
    "Product",
    "SetNode",
    "Slice",
    "is_subset",
    "set_from_geometry",
]
