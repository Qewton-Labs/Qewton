from __future__ import annotations
from types import EllipsisType
from typing import Any

from ..config.errors import DataConfigMismatchError
from ..config.variables import Variable


# TODO: General domain classes to define underlying geometry, maybe later include again
# func. such as union, differences,....


class Geometry:
    """
    Represents a geometric shape.
    """

    def __init__(
        self,
        variable: Variable | None = None,
        dim: int | None = None,
        shape: tuple[int | None, ...] | EllipsisType = ...,
    ):
        self.variable = variable if variable is not None else Variable()
        self._dim = dim
        self.shape: tuple[int | None, ...] | EllipsisType = shape
        if variable is not None and dim is not None:
            assert variable.dim == dim
        self.markers = {}

    @property
    def dim(self) -> int | None:
        return self.variable.dim if not self.variable.is_empty else self._dim

    def unify_with(self, other: Geometry) -> Geometry:
        """
        Unifies this `Geometry` object with another `Geometry` object.

        Placeholder values may appear for `variable` and `shape`.
        If one geometry is more concrete than the other, the concrete
        representation is returned when the two are compatible.

        When possible, the more specific object is refined in-place rather
        than creating a new base `Geometry` instance.
        """
        if other is self or other == self:
            return self

        if self.is_discretization_of(other):
            return self
        if other.is_discretization_of(self):
            return other

        if self.dim is not None and other.dim is not None and self.dim != other.dim:
            raise DataConfigMismatchError(
                f"Cannot unify geometries with different dims: {self.dim} != {other.dim}"
            )

        try:
            unified_variable = self.variable.unify(other.variable)
        except ValueError as error:
            raise DataConfigMismatchError(
                f"Cannot unify geometry variables: {error}"
            ) from error

        unified_dim = self.dim if self.dim is not None else other.dim
        if unified_dim is None and not unified_variable.is_empty:
            unified_dim = unified_variable.dim

        unified_shape = self._unify_shapes(self.shape, other.shape)

        target: Geometry = self._preferred_refinement_target(other)
        target.apply_unified(unified_variable, unified_dim, unified_shape)
        return target

    @staticmethod
    def _unify_shapes(
        shape_a: tuple[int | None, ...] | EllipsisType,
        shape_b: tuple[int | None, ...] | EllipsisType,
    ) -> tuple[int | None, ...] | EllipsisType:
        if shape_a is ...:
            return shape_b
        if shape_b is ...:
            return shape_a

        if len(shape_a) != len(shape_b):
            raise DataConfigMismatchError(
                f"Cannot unify geometry shapes with different lengths: \
                    {len(shape_a)} != {len(shape_b)}"
            )

        unified: list[int | None] = []
        for index, (dim_a, dim_b) in enumerate(zip(shape_a, shape_b)):
            if dim_a is None:
                unified.append(dim_b)
            elif dim_b is None:
                unified.append(dim_a)
            elif dim_a == dim_b:
                unified.append(dim_a)
            else:
                raise DataConfigMismatchError(
                    f"Cannot unify geometry shapes at index {index}: \
                        {dim_a} != {dim_b}"
                )
        return tuple(unified)

    def _shape_is_placeholder(self, shape: tuple[int | None, ...] | EllipsisType) -> bool:
        if shape is ...:
            return True
        return any(dim is None for dim in shape)

    def _shape_needs_update(
        self,
        current_shape: tuple[int | None, ...] | EllipsisType,
        unified_shape: tuple[int | None, ...] | EllipsisType,
    ) -> bool:
        if current_shape is ...:
            return unified_shape is not ...
        if isinstance(current_shape, tuple) and isinstance(unified_shape, tuple):
            return current_shape != unified_shape
        return False

    def apply_unified(
        self,
        unified_variable: Variable,
        unified_dim: int | None,
        unified_shape: tuple[int | None, ...] | EllipsisType,
    ) -> None:
        if not unified_variable.is_empty and self.variable.is_empty:
            self.variable = unified_variable
        if unified_dim is not None and self._dim is None:
            self._dim = unified_dim
        if self._shape_needs_update(self.shape, unified_shape):
            self.shape = unified_shape

    def _preferred_refinement_target(self, other: Geometry) -> Geometry:
        assert Geometry in [self.__class__, other.__class__]
        if type(self) is not type(other):
            if isinstance(self, type(other)):
                return self
            if isinstance(other, type(self)):
                return other
        return self

    def discretize(
        self,
        discretization_points: Any | None = None,
        discretization_shape: tuple[int, ...] | None = None,
    ) -> DiscreteGeometry:
        if discretization_points is not None:
            shape = discretization_points.shape[:-1]
            if discretization_shape is not None:
                assert shape == discretization_shape
            if self.dim is not None:
                assert (
                    discretization_points.shape[-1] == self.dim
                ), "Points shape does not match dimension"
        elif discretization_shape is not None:
            shape = discretization_shape
        else:
            raise ValueError(
                "Either discretization_points or discretization_shape must be provided."
            )

        if self.shape is not ...:
            for p_s, s in zip(shape, self.shape):
                assert (
                    p_s == s or s is None
                ), f"Default shape {self.shape} does not match provided points"

        dis_geo = DiscreteGeometry(
            variable=self.variable,
            dim=self.dim,
            shape=shape,
            discretization_points=discretization_points,
        )
        dis_geo.discretization_of = self
        return dis_geo

    def is_discretization_of(self, other_geometry: Geometry) -> bool:
        return False

    def create_mesh(self, max_vertex_distance: float | None = None):
        raise NotImplementedError()

    def set_marker(self, marker, marker_description):
        raise NotImplementedError()

    def get_marker(self, marker):
        if marker in self.markers:
            return self.markers[marker]
        raise KeyError(f"{marker} marker not found.")


class DiscreteGeometry(Geometry):

    def __init__(
        self,
        shape: tuple[int, ...],
        variable: Variable | None = None,
        dim: int | None = None,
        discretization_points: Any | None = None,
    ):
        super().__init__(variable=variable, dim=dim, shape=shape)
        self.discretization_of: Geometry
        self.discretization_points = discretization_points

    def is_discretization_of(self, other_geometry: Geometry) -> bool:
        if self.discretization_of == other_geometry:
            return True
        if self is other_geometry:
            return True
        return False


class AnalyticGeometry(Geometry):
    """
    Base class for geometries defined via the Pioneer backend.
    """

    pass


class Interval(AnalyticGeometry):

    def __init__(self, lower, upper, variable: Variable | None = None):
        self.lower = lower
        self.upper = upper
        super().__init__(variable, 1, (None,))

    def create_mesh(self, max_vertex_distance: float | None = None):
        raise NotImplementedError()

    def set_marker(
        self,
        marker,
    ):
        self.markers[marker] = self.lower
        return super().set_marker(marker, marker_description)


####################################
### Plan:
# - All geometries have a sampling methods (at least grid and random)
# - DiscreteGeo. have also a discrete sampling method (only sampling from discretization points)
# - All geometries can be discretized by create mesh method -> returns MeshGeometry
# - A geometry has a .boundary property (will be created when first called)
# - We can mark CAD-geometries via lambda functions, the functions will
#   be saved internally, and we return the corresponding subdomains for further usage.
#   The saved markers can be used when creating a mesh.
# - Union, etc. is only in CADGeometry
# - Cart. Product not in Geometry
# - Put CAD into qewton.geometry.cad
# - Default geo. take TorchPhysics implementation
# - Start with numpy implementation, switch to nodes etc. maybe later

### Samplers:
# - A sampler has a flag to either use mesh based or direct sampling
# - One can still filter points via rejection sampling
# - Allow products in samplers (passing arguments/outputs between each other)
# - Return Normals flag, to build output ports / And normalvector compute node
# - Has backend and device, moves points accordingly
