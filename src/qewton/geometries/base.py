"""Base classes for geometries: :class:`Geometry`, :class:`DiscreteGeometry`
and :class:`BoundaryGeometry`."""

from __future__ import annotations
from types import EllipsisType
from typing import Any, Generic

from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.config.saving.saving import Serializable
from qewton.backends.base import ComputingBackend, TensorType
from qewton.backends import DEFAULT_DL_BACKEND


def vertex_distance_for(
    max_vertex_distance: float | dict[Variable, float] | None, variable: Variable
) -> float | None:
    """Return the maximum vertex distance that applies to `variable`.

    Args:
        max_vertex_distance: One value for every geometry, or one value per
            geometry variable.
        variable: The variable of the geometry being meshed.

    Returns:
        float or None: The value for `variable`; ``None`` if a dict has no
            entry for it.
    """
    if isinstance(max_vertex_distance, dict):
        return max_vertex_distance.get(variable)
    return max_vertex_distance


class Geometry(Serializable, Generic[TensorType]):
    """Geometric domain used for sampling points, meshing and plotting.

    Args:
        variable: The variable that belongs to this geometry. If ``None``, an
            empty placeholder :class:`~qewton.config.variables.Variable` is
            used.
        dim: The dimension of the geometry. Must match ``variable.dim`` if both
            are given.
        shape: The discrete shape of the geometry. Entries may be ``None`` and
            ``...`` means the shape is unknown.
        backend: The backend used for computations.

    Attributes:
        variable (Variable): The variable that belongs to this geometry.
        dim (int or None): The dimension, taken from `variable` if it is not
            empty, otherwise the `dim` passed at construction or set by
            unification.
        shape (tuple of (int or None) or EllipsisType): The discrete shape of
            the geometry.
        backend (type[ComputingBackend]): The backend used for computations.
        markers (dict): Named subdomains of this geometry.
        boundary (BoundaryGeometry): The boundary of this geometry. Created by
            :meth:`create_boundary` on first access and cached in
            ``boundary_object``.

    Notes:
        Subclasses should implement :meth:`create_boundary`,
        :meth:`sample_random_uniform`, :meth:`sample_grid`, :meth:`contains`,
        :meth:`bounding_box` and ``_get_volume`` where these operations make
        sense; the base implementations raise :class:`NotImplementedError`.
        They may override :meth:`create_mesh`, :meth:`is_discretization_of` and
        :meth:`apply_unified`.
    """

    def __init__(
        self,
        variable: Variable | None = None,
        dim: int | None = None,
        shape: tuple[int | None, ...] | EllipsisType = ...,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.variable = variable if variable is not None else Variable()
        self._dim = dim
        self.shape: tuple[int | None, ...] | EllipsisType = shape
        if variable is not None and dim is not None:
            assert variable.dim == dim
        self.markers = {}
        self._user_volume = None
        self.boundary_object: BoundaryGeometry | None = None
        self.backend = backend

    @property
    def boundary(self) -> BoundaryGeometry:
        if self.boundary_object is None:
            self.boundary_object = self.create_boundary()
        return self.boundary_object

    def create_boundary(self) -> BoundaryGeometry:
        """Create the boundary of this geometry.

        Returns:
            BoundaryGeometry: A new boundary geometry.

        Raises:
            NotImplementedError: If the geometry has no boundary
                implementation.
        """
        raise NotImplementedError()

    @property
    def dim(self) -> int | None:
        return self.variable.dim if not self.variable.is_empty else self._dim

    def unify_with(self, other: Geometry) -> Geometry:
        """Unify this geometry with `other`.

        Placeholder values may appear for `variable`, `dim` and `shape`. If
        one geometry is a discretization of the other, the discretization is
        returned. Otherwise the placeholders are filled from the other
        geometry and the more specific of the two objects (the instance of
        the subclass) is refined in place and returned.

        Args:
            other: The geometry to unify with.

        Returns:
            Geometry: The unified geometry, either `self` or `other`.

        Raises:
            DataConfigMismatchError: If dimensions, variables or shapes are
                incompatible.
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
        """Fill this geometry's placeholders with unified values in place.

        The variable and dimension are only set if they are still empty; the
        shape is replaced if it differs from `unified_shape`.

        Args:
            unified_variable: The unified variable.
            unified_dim: The unified dimension.
            unified_shape: The unified shape.
        """
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
        """Discretize this geometry into a :class:`DiscreteGeometry`.

        The discretization is defined by the discretization points, the
        discretization shape, or both (then they must be consistent).

        Args:
            discretization_points: The points of the discrete geometry, with
                the dimension in the last axis.
            discretization_shape: The shape of the discretization, without the
                dimension axis.

        Returns:
            DiscreteGeometry: The discretized geometry. Its
                ``discretization_of`` is set to this geometry.

        Raises:
            ValueError: If neither `discretization_points` nor
                `discretization_shape` is given.
            AssertionError: If the given points, shape, dimension or this
                geometry's shape are inconsistent.
        """
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

    def is_discretization_of(
        self, other_geometry: Geometry  # pylint: disable=unused-argument
    ) -> bool:
        """Check whether this geometry is a discretization of another one.

        Args:
            other_geometry: The geometry to compare to.

        Returns:
            bool: Whether this geometry discretizes `other_geometry`. Always
                ``False`` for a continuous geometry.
        """
        return False

    def create_mesh(self, max_vertex_distance: float | None = None, device: Device = cpu):
        """Mesh this geometry into a mesh geometry.

        Args:
            max_vertex_distance: The maximum distance between mesh vertices,
                which controls how fine the mesh is.
            device: The device the mesh is created on.

        Returns:
            DiscreteGeometry: The meshed geometry.

        Raises:
            NotImplementedError: If the geometry does not support meshing.
        """
        raise NotImplementedError()

    def set_marker(self, marker, marker_description):
        """Mark a subdomain of this geometry.

        Args:
            marker (hashable): The name of the marker.
            marker_description (Any): The description of the marked subdomain.

        Raises:
            NotImplementedError: Always, since general geometries can not be
                marked.
        """
        # TODO!!!
        raise NotImplementedError(
            "General geometries can not be marked, use "
            "the filter functions in the sampler object instead."
        )

    def get_marker(self, marker):
        """Return the subdomain stored under `marker`.

        Args:
            marker (hashable): The name of the marker.

        Returns:
            Any: The stored marker entry.

        Raises:
            KeyError: If no marker with this name exists.
        """
        # TODO!!!
        if marker in self.markers:
            return self.markers[marker]
        raise KeyError(f"{marker} marker not found.")

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Sample uniformly distributed random points inside the geometry.

        Args:
            n_points: The number of points to sample.
            device: The device to sample on.

        Returns:
            TensorType: The sampled points, shape ``(n_points, dim)``.
        """
        raise NotImplementedError()

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        """Sample grid points inside the geometry.

        Args:
            n_points: The number of points to sample.
            device: The device to sample on.

        Returns:
            TensorType: The sampled points, shape ``(n_points, dim)``.
        """
        raise NotImplementedError()

    def __and__(self, other):
        """Return the intersection of two geometries."""
        raise NotImplementedError()

    def __add__(self, other):
        """Return the union of two geometries."""
        return NotImplementedError()

    def __or__(self, other):
        """Return the union of two geometries."""
        raise self + other

    def __sub__(self, other):
        """Return the difference of two geometries."""
        raise NotImplementedError()

    def __mul__(self, other):
        """Return the Cartesian product of two geometries.

        Returns:
            ProductGeometry: The product of this geometry and `other`, see
                :class:`~qewton.geometries.product.ProductGeometry`.
        """
        from .product import ProductGeometry

        return ProductGeometry(self, other)

    def contains(self, points):
        """Check for every point whether it lies inside the geometry.

        Args:
            points (TensorType): The points to check, shape
                ``(n_points, dim)``.

        Returns:
            TensorType: Boolean tensor of shape ``(n_points, 1)``, ``True``
                where the point lies inside.
        """
        raise NotImplementedError()

    def bounding_box(self):
        """Compute the axis-aligned bounds of the geometry.

        Returns:
            TensorType: Tensor of length ``2 * dim`` of the form
                ``[axis_1_min, axis_1_max, axis_2_min, axis_2_max, ...]``.
        """
        raise NotImplementedError()

    def set_volume(self, volume: float):
        """Set the volume of the geometry by hand.

        Args:
            volume: The volume of the geometry. :meth:`volume` returns this
                value instead of computing one.

        Notes:
            Use this for geometries whose volume can otherwise only be
            approximated, if the exact value is known.
        """
        self._user_volume = volume

    def _get_volume(self):
        raise NotImplementedError()

    def volume(self):
        """Return the volume of the geometry.

        Returns:
            float: The volume set by :meth:`set_volume`, if any, otherwise the
                computed volume.
        """
        if self._user_volume is None:
            return self._get_volume()
        return self._user_volume


class DiscreteGeometry(Geometry[TensorType]):
    """Geometry with a concrete discretization shape and optional points.

    Args:
        shape: The shape of the discretization, e.g. the number of points per
            axis.
        variable: The variable connected to this geometry.
        dim: The dimension of this geometry.
        discretization_points: The discrete points that make up this geometry,
            with the dimension in the last axis.
        backend: The backend used for computations.

    Attributes:
        discretization_of (Geometry or None): The geometry this one
            discretizes, if known.
        discretization_points (TensorType or None): The discrete points of this
            geometry.

    Notes:
        Subclasses may implement
        :meth:`sample_random_uniform_from_discretization` and
        :meth:`sample_grid_from_discretization`.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        variable: Variable | None = None,
        dim: int | None = None,
        discretization_points: TensorType | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(variable=variable, dim=dim, shape=shape, backend=backend)
        self.discretization_of: Geometry | None = None
        self.discretization_points = discretization_points

    def bounding_box(self):
        """Compute the axis-aligned bounds of the geometry.

        Uses the discretization points if present, otherwise the bounding
        box of ``discretization_of``.

        Returns:
            TensorType: Tensor of the form ``[axis_1_min, axis_1_max, ...]``.

        Raises:
            NotImplementedError: If neither points nor a source geometry are
                available.
        """
        if self.discretization_points is not None:
            self.discretization_points = self.backend.build_tensor(
                self.discretization_points
            )
            bounding_box = []
            for i in range(self.discretization_points.shape[-1]):
                min_val = self.backend.math.min(self.discretization_points[..., i])
                max_val = self.backend.math.max(self.discretization_points[..., i])
                bounding_box.append(min_val)
                bounding_box.append(max_val)
            return self.backend.build_tensor(bounding_box)
        if self.discretization_of is not None:
            return self.discretization_of.bounding_box()
        raise NotImplementedError(
            "Bounding box is not implemented for this discrete geometry."
        )

    def is_discretization_of(self, other_geometry: Geometry) -> bool:
        """Check whether this geometry is a discretization of another one.

        Args:
            other_geometry: The geometry to compare to.

        Returns:
            bool: ``True`` if `other_geometry` equals ``discretization_of`` or
                is this geometry itself.
        """
        if self.discretization_of == other_geometry:
            return True
        if self is other_geometry:
            return True
        return False

    def sample_random_uniform_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Sample random points from the discretization points.

        Args:
            n_points: The number of points to sample. If larger than the number
                of discretization points, duplicates are returned.
            device: The device to sample on.

        Returns:
            TensorType: The sampled points, shape ``(n_points, dim)``.
        """
        raise NotImplementedError()

    def sample_grid_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Sample points evenly from the discretization points.

        Every discretization point is returned equally often.

        Args:
            n_points: The number of points to sample. If larger than the number
                of discretization points, duplicates are returned.
            device: The device to sample on.

        Returns:
            TensorType: The sampled points, shape ``(n_points, dim)``.
        """
        raise NotImplementedError()


class BoundaryGeometry(Geometry[TensorType]):
    """Base class for the boundary of a geometry.

    Can be used like any other :class:`Geometry`.

    Args:
        geometry (Geometry): The geometry whose boundary this is. Its variable,
            dimension and backend are taken over.

    Attributes:
        geometry (Geometry): The geometry whose boundary this is.

    Notes:
        Subclasses should implement :meth:`normal`, :meth:`sample_grid` and
        :meth:`sample_random_uniform`, including the `include_normals` option.
    """

    def __init__(self, geometry):
        assert isinstance(geometry, Geometry)
        if geometry.dim is not None:
            dim = geometry.dim
        else:
            dim = None
        super().__init__(variable=geometry.variable, dim=dim, backend=geometry.backend)
        self.geometry = geometry

    def bounding_box(self):
        """Return the bounding box of the underlying geometry.

        Returns:
            TensorType: Tensor of the form ``[axis_1_min, axis_1_max, ...]``.
        """
        return self.geometry.bounding_box()

    def create_boundary(self) -> BoundaryGeometry:
        """Raise, since a boundary has no boundary.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Boundary of a boundary geometry is not defined.")

    def normal(self, points, device: Device | str = cpu) -> TensorType:
        """Compute the outward normal vector at each point.

        Args:
            points (TensorType): The boundary points, shape
                ``(n_points, dim)``.
            device: The device to compute on.

        Returns:
            TensorType: The normal vectors, shape ``(n_points, dim)``.
        """
        raise NotImplementedError()

    def sample_grid(
        self,
        n_points: int,
        device: Device | str = cpu,
        include_normals: bool = False,  # pylint: disable=unused-argument
    ) -> TensorType | tuple[TensorType, TensorType]:
        """Sample grid points on the boundary.

        Args:
            n_points: The number of points to sample.
            device: The device to sample on.
            include_normals: Whether to also return the normal vectors at the
                points.

        Returns:
            TensorType or tuple of TensorType: The points, or
                ``(points, normals)`` if `include_normals` is ``True``.
        """
        return super().sample_grid(n_points)

    def sample_random_uniform(
        self,
        n_points: int,
        device: Device | str = cpu,
        include_normals: bool = False,  # pylint: disable=unused-argument
    ) -> TensorType | tuple[TensorType, TensorType]:
        """Sample uniformly distributed random points on the boundary.

        Args:
            n_points: The number of points to sample.
            device: The device to sample on.
            include_normals: Whether to also return the normal vectors at the
                points.

        Returns:
            TensorType or tuple of TensorType: The points, or
                ``(points, normals)`` if `include_normals` is ``True``.
        """
        return super().sample_random_uniform(n_points)


####################################
### Plan:
# - We can mark CAD-geometries via lambda functions, the functions will
#   be saved internally, and we return the corresponding subdomains for further usage.
#   The saved markers can be used when creating a mesh.
# - Put CAD into qewton.geometry.cad

### Samplers:
# - One can still filter points via rejection sampling
