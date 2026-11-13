"""Sets on which constraints hold: domains, boundaries, parts, interfaces, products
and slices, and the questions asked about them.

The set classes describe the structure symbolically: a boundary knows what it is the
boundary of, a product knows its factors. The functions at the end answer what the
other layers need to know from that structure, e.g. `boundary_of(region)` or
`depends_on(node, source)`, so that no other module walks the set classes itself.

TODO: A direct connection to the underlying geometry and boundaries
might be added later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Sequence

from qewton.config.variables import Variable
from qewton.equations.geometry.coordinates import check_coordinate
from qewton.equations.printable import Printable
from qewton.geometries.base import Geometry
from qewton.geometries.continuous.base import ContinuousGeometry
from qewton.geometries.product import ProductGeometry


class SetNode(Printable, ABC):
    """A set inside the coordinates given by `axes`."""

    @property
    @abstractmethod
    def axes(self) -> frozenset[Variable]:
        """The coordinates the set lives in."""

    @property
    @abstractmethod
    def codim(self) -> int:
        """How many dimensions the set is below its coordinates.
        0 is just the full-dimensional set.
        1 is a co-dimension-one set, e.g., a boundary.
        and so on...
        """

    def built_from(self) -> tuple[SetNode, ...]:
        """The sets this one is defined from: the set of a boundary or a part, the
        factors of a product, the two sides of an interface. A domain or a slice is
        built from nothing."""
        return ()

    def __mul__(self, other: object) -> SetNode:
        if not isinstance(other, SetNode):
            return NotImplemented
        return Product(self, other)


def _check_set(obj: object) -> SetNode:
    if not isinstance(obj, SetNode):
        raise TypeError(f"Expected a set, got {type(obj).__name__}.")
    return obj


@dataclass(frozen=True)
class Domain(SetNode):
    """An open set named `name`, e.g. Omega.

    Give either `ambient`, the coordinate, or a continuous `geometry`, whose
    coordinate then becomes `ambient`. The geometry is left out of equality and
    hashing: the same named set is the same set however it is discretized.
    Meshes/Geometries are bound in the Task, not here.
    """

    name: str
    ambient: Variable | None = None
    geometry: ContinuousGeometry | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.geometry is not None:
            if not isinstance(self.geometry, ContinuousGeometry):
                raise TypeError(
                    f"Domain {self.name!r} needs a continuous geometry, got "
                    f"{type(self.geometry).__name__}. Bind meshes in the Task instead."
                )
            if self.ambient is not None and self.ambient != self.geometry.variable:
                raise ValueError(
                    f"Domain {self.name!r}: ambient {self.ambient.name!r} does not match "
                    f"the geometry's coordinate {self.geometry.variable.name!r}."
                )
            # Frozen dataclass: the derived field is set with object.__setattr__.
            object.__setattr__(self, "ambient", self.geometry.variable)
        if self.ambient is None:
            raise ValueError(
                f"Domain {self.name!r} needs an ambient coordinate or a geometry."
            )
        check_coordinate(self.ambient)

    @property
    def axes(self) -> frozenset[Variable]:
        return frozenset({self.ambient})  # type: ignore

    @property
    def codim(self) -> int:
        return 0


@dataclass(frozen=True)
class Boundary(SetNode):
    """The boundary of `of`, one dimension lower."""

    of: SetNode

    def __post_init__(self):
        _check_set(self.of)

    @property
    def axes(self) -> frozenset[Variable]:
        return self.of.axes

    @property
    def codim(self) -> int:
        return self.of.codim + 1

    def built_from(self) -> tuple[SetNode, ...]:
        return (self.of,)


@dataclass(frozen=True)
class Part(SetNode):
    """A named piece of `of`, e.g. Gamma_D inside the boundary of Omega."""

    name: str
    of: SetNode

    def __post_init__(self):
        _check_set(self.of)

    @property
    def axes(self) -> frozenset[Variable]:
        return self.of.axes

    @property
    def codim(self) -> int:
        return self.of.codim

    def built_from(self) -> tuple[SetNode, ...]:
        return (self.of,)


@dataclass(frozen=True)
class Interface(SetNode):
    """The common boundary of two subdomains `first` and `second` of one domain.

    It is one dimension below the domain, like a boundary. The normal of an interface
    points from `first` into `second`. A problem declares its interfaces with
    `Problem(interfaces=...)`, so that `classify` can tell when conditions are missing.
    """

    first: Part
    second: Part

    def __post_init__(self):
        for side in (self.first, self.second):
            if not isinstance(side, Part):
                raise TypeError(
                    f"An interface lies between two subdomains (parts), got {side!r}."
                )
        if self.first == self.second:
            raise ValueError("An interface needs two different subdomains.")
        if self.first.of != self.second.of:
            raise ValueError("The two subdomains of an interface must share one parent.")
        if isinstance(self.first.of, Boundary):
            raise ValueError(
                "Interfaces lie between subdomains, not between boundary parts."
            )

    @property
    def axes(self) -> frozenset[Variable]:
        return self.first.axes

    @property
    def codim(self) -> int:
        return self.first.codim + 1

    def built_from(self) -> tuple[SetNode, ...]:
        return (self.first, self.second)


@dataclass(frozen=True, init=False)
class Product(SetNode):
    """A product of sets on disjoint coordinates, e.g. Omega x I.

    Write `Product(Omega, I)`, `Product((Omega, I))` or `Omega * I`. Nested products
    are flattened, so `(A * B) * C == Product(A, B, C)`.
    """

    factors: tuple[SetNode, ...]

    def __init__(self, *given: SetNode | Iterable[SetNode]):
        if len(given) == 1 and isinstance(given[0], (tuple, list)):
            given = tuple(given[0])
        flat: list[SetNode] = []
        for factor in given:
            _check_set(factor)
            flat.extend(factor.factors if isinstance(factor, Product) else (factor,))
        object.__setattr__(self, "factors", tuple(flat))
        self.__post_init__()

    def __post_init__(self):
        if not self.factors:
            raise ValueError("A product needs at least one factor.")
        seen: frozenset[Variable] = frozenset()
        for factor in self.factors:
            _check_set(factor)
            shared = seen & factor.axes
            if shared:
                names = list(v.name for v in shared)
                raise ValueError(f"Product factors share coordinates {names}.")
            seen |= factor.axes

    @property
    def axes(self) -> frozenset[Variable]:
        return frozenset().union(*(factor.axes for factor in self.factors))

    @property
    def codim(self) -> int:
        return sum(factor.codim for factor in self.factors)

    def built_from(self) -> tuple[SetNode, ...]:
        return self.factors


@dataclass(frozen=True)
class Slice(SetNode):
    """The slice {axis = value}, e.g. {t = 0}. It drops one dimension."""

    axis: Variable
    value: float

    def __post_init__(self):
        check_coordinate(self.axis)
        if self.axis.dim != 1:
            raise ValueError(
                "A slice fixes one value, so its coordinate must have dim 1."
            )

    @property
    def axes(self) -> frozenset[Variable]:
        return frozenset({self.axis})

    @property
    def codim(self) -> int:
        return 1


@dataclass(frozen=True)
class Partition(Printable):
    """Declares that `parts` cover `whole` without overlap, checked by validate."""

    whole: SetNode
    parts: tuple[Part, ...]


@dataclass(frozen=True)
class Periodic(Printable):
    """Declares that `second` is `first` moved by `offset` along `axis`.

    Use it for periodic boundary conditions: the two sets are the same place, so the
    functions agree there. For a box [0, L] x [0, H] with Gamma_L at x_1 = 0 and
    Gamma_R at x_1 = L, Periodic(Gamma_L, Gamma_R, x, (L, 0)) says
    Gamma_R = Gamma_L + (L, 0).
    The condition on the functions is still written as a constraint, with a shift.
    `offset` is a number for a one-dimensional axis and a tuple otherwise.
    """

    first: SetNode
    second: SetNode
    axis: Variable
    offset: float | tuple[float, ...]

    def __post_init__(self):
        _check_set(self.first)
        _check_set(self.second)
        check_coordinate(self.axis)
        if self.first == self.second:
            raise ValueError("A set cannot be periodic with itself.")
        if self.first.axes != self.second.axes or self.first.codim != self.second.codim:
            raise ValueError("Periodic sets must live in the same coordinates and codim.")
        if self.axis not in self.first.axes:
            raise ValueError(
                f"The axis {self.axis.name!r} is not a coordinate of the sets."
            )
        offset = (
            tuple(self.offset) if isinstance(self.offset, (list, tuple)) else self.offset
        )
        values = offset if isinstance(offset, tuple) else (offset,)
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            raise TypeError(f"The offset must be numbers, got {self.offset!r}.")
        expected = 1 if self.axis.dim == 1 else self.axis.dim
        if len(values) != expected or (isinstance(offset, tuple) != (self.axis.dim != 1)):
            raise ValueError(
                f"The offset along {self.axis.name!r} needs {expected} number(s), "
                f"got {self.offset!r}."
            )
        if not any(values):
            raise ValueError("The offset must not be zero.")
        object.__setattr__(self, "offset", offset)


def is_subset(a: SetNode, b: SetNode) -> bool:
    """Whether `a` is provably inside `b`, from the declared structure.

    This is a sufficient test, not a complete one. There is no set algebra, and
    Domain bounds are not modelled, so a slice of a coordinate counts as inside
    that coordinate's domain.
    """
    if a == b:
        return True
    if isinstance(a, Part):
        return is_subset(a.of, b)
    if isinstance(a, Slice) and isinstance(b, Domain):
        return a.axis == b.ambient
    if isinstance(a, Product) and isinstance(b, Product):
        return len(a.factors) == len(b.factors) and all(
            any(is_subset(fa, fb) for fb in b.factors) for fa in a.factors
        )
    return False


###
# Queries about the structure of sets.
###
def factors(region: SetNode | None) -> tuple[SetNode, ...]:
    """The factors of a product (products are flat); any other set is its own factor."""
    if isinstance(region, Product):
        return region.factors
    return () if region is None else (region,)


def with_factors(sets: Iterable[SetNode]) -> set[SetNode]:
    """The sets, together with the factors of every product among them."""
    return {found for node in sets for found in (node, *factors(node))}


def lies_on_boundary(region: SetNode) -> bool:
    """Whether `region` is a boundary, or a part of one; a part of a domain is not."""
    while isinstance(region, Part):
        region = region.of
    return isinstance(region, Boundary)


def has_normal(region: SetNode) -> bool:
    """Whether `region` has a normal: a boundary, an interface, or a part of either."""
    while isinstance(region, Part):
        region = region.of
    return isinstance(region, (Boundary, Interface))


def base_domain(region: SetNode | None) -> SetNode | None:
    """The domain a subdomain (a `Part` of a domain) belongs to; other sets stay."""
    while isinstance(region, Part):
        region = region.of
    return region


def interface_of(region: SetNode | None) -> Interface | None:
    """The interface a constraint set contains, e.g. Gamma for Gamma x I."""
    return next((f for f in factors(region) if isinstance(f, Interface)), None)


def boundary_of(region: SetNode | None) -> Boundary | None:
    """The boundary a constraint set lies on, e.g. d(Omega) for Gamma x I."""
    for factor in factors(region):
        if isinstance(factor, Boundary):
            return factor
        if isinstance(factor, Part) and isinstance(factor.of, Boundary):
            return factor.of
    return None


def spatial_domain(region: SetNode | None, time: Variable | None) -> SetNode | None:
    """The factor of a set that is not the time interval."""
    for factor in factors(region):
        if factor.axes != frozenset({time}):
            return factor
    return None


def sliced_axes(region: SetNode) -> frozenset[Variable]:
    """The coordinates a set fixes to a single value, such as t in Omega x {t = 0}."""
    if isinstance(region, Slice):
        return frozenset({region.axis})
    return frozenset().union(*(sliced_axes(p) for p in region.built_from()))


def set_free_axes(region: SetNode) -> frozenset[Variable]:
    """The coordinates a point of the set still varies in: its axes minus the sliced."""
    return region.axes - sliced_axes(region)


def depends_on(
    node: SetNode, source: SetNode, partitions: Sequence[Partition] = ()
) -> bool:
    """Whether `node` changes when `source` changes, e.g. when `source` varies.

    It does if it is `source`, if a set it is built from depends on `source` (the
    boundary of a moving domain), or if both are parts of one of the `partitions`:
    the parts fill the whole, so one cannot move alone. A whole with a moving part
    stays where it is.
    """
    if node == source:
        return True
    if any(depends_on(p, source, partitions) for p in node.built_from()):
        return True
    return any(node in p.parts and source in p.parts for p in partitions)


def set_from_geometry(geometry: Geometry, name: str | None = None) -> SetNode:
    """The set described by a geometry.

    A continuous geometry becomes a Domain called `name`. A ProductGeometry becomes a
    Product with one Domain per factor, named after the factor's coordinate, because a
    product has no name of its own. Other geometries, such as meshes, are rejected:
    they are bound in the Task.
    """
    if isinstance(geometry, ProductGeometry):
        if name is not None:
            raise ValueError(
                "A product has no name. Its factors are named after their coordinates."
            )
        domain_list = []
        for factor in geometry.factors:
            if not isinstance(factor, ContinuousGeometry):
                raise TypeError("All factors of a product must be continuous geometries.")
            if factor.variable.name is None:
                raise ValueError("Each factor must have a variable name.")
            domain_list.append(Domain(factor.variable.name, geometry=factor))
        return Product(tuple(domain_list))
    if name is None:
        raise ValueError("A continuous geometry needs a name for its domain.")
    if not isinstance(geometry, ContinuousGeometry):
        raise ValueError("Only continuous geometries can be converted to a Domain.")
    return Domain(name, geometry=geometry)
