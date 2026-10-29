"""Sets on which constraints hold: domains, boundaries, parts, products and slices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qewton.config.variables import Variable
from qewton.equations.geometry.coordinates import check_coordinate
from qewton.equations.printing.printable import Printable
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
        """How many dimensions the set is below its coordinates. 0 for an open set."""


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
    Meshes are bound in the Task, not here.
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
        return frozenset({self.ambient})

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


@dataclass(frozen=True)
class Product(SetNode):
    """A product of sets on disjoint coordinates, e.g. Omega x I."""

    factors: tuple[SetNode, ...]

    def __post_init__(self):
        if not self.factors:
            raise ValueError("A product needs at least one factor.")
        seen: frozenset[Variable] = frozenset()
        for factor in self.factors:
            _check_set(factor)
            shared = seen & factor.axes
            if shared:
                names = sorted(v.name for v in shared)
                raise ValueError(f"Product factors share coordinates {names}.")
            seen |= factor.axes

    @property
    def axes(self) -> frozenset[Variable]:
        return frozenset().union(*(factor.axes for factor in self.factors))

    @property
    def codim(self) -> int:
        return sum(factor.codim for factor in self.factors)


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
        return Product(
            tuple(
                Domain(factor.variable.name, geometry=factor)
                for factor in geometry.factors
            )
        )
    if name is None:
        raise ValueError("A continuous geometry needs a name for its domain.")
    return Domain(name, geometry=geometry)
