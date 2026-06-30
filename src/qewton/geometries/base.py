from __future__ import annotations
from types import EllipsisType
from typing import Any, Generic

from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.backends.base import ComputingBackend, TensorType
from qewton.backends import DEFAULT_DL_BACKEND


class Geometry(Generic[TensorType]):
    """Represents a geometric shape for sampling points, computing normal
    vectors, plotting, etc.

    Args:
        variable (Variable | None, optional): The variable that belongs to this
                geometry. Defaults to None.
        dim (int | None, optional): The dimension of the geometry.
            Defaults to None.
        shape (tuple[int  |  None, ...] | EllipsisType, optional): A discrete
            shape. Defaults to ....
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
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
        """Returns the boundary of this geometry as a new Geometry object.

        Returns:
            BoundaryGeometry: The boundary of this geometry.
        """
        if self.boundary_object is None:
            self.boundary_object = self.create_boundary()
        return self.boundary_object

    def create_boundary(self) -> BoundaryGeometry:
        raise NotImplementedError()

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
        """Discretizes this geometry into a `DiscreteGeometry` object.
        The discretization can be defined by providing either the
        discretization points or the shape of the discretization.

        Args:
            discretization_points (Any | None, optional): The points for the
                discrete geometry. Defaults to None.
            discretization_shape (tuple[int, ...] | None, optional): The shape of
                discretization. Defaults to None.

        Raises:
            ValueError: If both or neither of discretization_points and
                discretization_shape are provided, or if the provided information
                is inconsistent.

        Returns:
            DiscreteGeometry: The discretized geometry as a DiscreteGeometry object.
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
        """Check if this geometry is a discretization of the provided one.

        Args:
            other_geometry (Geometry, optional): The geometry to compare to.
                Defaults to unused-argument.

        Returns:
            bool: If this geometry is a discretization of the provided one.
        """
        return False

    def create_mesh(self, max_vertex_distance: float | None = None, device: Device = cpu):
        """Meshes this geometry into a `MeshGeometry` object. The meshing can be
        controlled by providing a maximum vertex distance, which determines the
        fineness of the mesh.

        Args:
            max_vertex_distance (float | None, optional): How fine the mesh
                should be. Defaults to None.

        Raises:
            NotImplementedError: By default meshing is not provided.
        """
        raise NotImplementedError()

    def set_marker(self, marker, marker_description):
        # TODO!!!
        raise NotImplementedError(
            "General geometries can not be marked, use "
            "the filter functions in the sampler object instead."
        )

    def get_marker(self, marker):
        # TODO!!!
        if marker in self.markers:
            return self.markers[marker]
        raise KeyError(f"{marker} marker not found.")

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Samples random uniform points inside this domain.

        Args:
            n_points (int): The number of points that should be sampled.
            device (Device): The device to sample on.

        Returns:
            array: An array of shape (n_points, dim) containing the sampled points.
        """
        raise NotImplementedError()

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        """Samples grid points inside this domain.

        Args:
            n_points (int): The number of points that should be sampled.
            device (Device): The device to sample on.

        Returns:
            array: An array of shape (n_points, dim) containing the sampled points.
        """
        raise NotImplementedError()

    def __and__(self, other):
        """Returns the intersection of two domains"""
        raise NotImplementedError()

    def __add__(self, other):
        """Returns the union of two domains"""
        return NotImplementedError()

    def __or__(self, other):
        """Returns the union of two domains"""
        raise self + other

    def __sub__(self, other):
        """Returns the difference of two domains"""
        raise NotImplementedError()

    def __mul__(self, other):
        """Creates the Cartesian product of two domains."""
        from .product import ProductGeometry

        return ProductGeometry(self, other)

    def contains(self, points):
        """Checks for every point in points if it lays inside the domain.

        Parameters
        ----------
        points : np.Array
            The points that should be checked.

        Returns
        -------
        np.Array
            A boolean array of the shape (len(points), 1) where every entry contains
            true if the point was inside or false if not.
        """
        raise NotImplementedError()

    def bounding_box(self):
        """Computes the bounds of the domain.

        Returns
        -------
        array :
            A np.array with the length of 2*self.dim.
            It has the form [axis_1_min, axis_1_max, axis_2_min, axis_2_max, ...],
            where min and max are the minimum and maximum value that the domain
            reaches in each dimension.
        """
        raise NotImplementedError()

    def set_volume(self, volume: float):
        """Set the volume of the given domain.

        Parameters
        ----------
        volume : number
            The volume of the domain. Can be a function if the volume changes
            depending on other variables.

        Notes
        -----
        For all basic domains the volume (and surface) are implemented.
        But if the given domain has a complex the volume can only be approximated.
        Therefore one can set here a exact expression for the volume, if known.
        """
        self._user_volume = volume

    def _get_volume(self):
        raise NotImplementedError()

    def volume(self):
        """Computes the volume of the current domain.

        Returns
        -------
        volume: float
            Returns the volume of the domain.
        """
        if self._user_volume is None:
            return self._get_volume()
        return self._user_volume


class DiscreteGeometry(Geometry[TensorType]):
    """A discrete geometry represents a geometry that includes a number of
    discretized points and concrete discretization shape.

    Args:
        shape (tuple[int, ...]): The shape of the discretization, e.g. the
            number of points in each dimension. Used for predicting the shape
            of a GeometryAxes.
        variable (Variable | None, optional): The variable connected to this
            geometry. Defaults to None.
        dim (int | None, optional): The dimension of this geometry. Defaults to None.
        discretization_points (Any | None, optional): Discrete points that
            make up this geometry. Defaults to None.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        variable: Variable | None = None,
        dim: int | None = None,
        discretization_points: Any | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(variable=variable, dim=dim, shape=shape, backend=backend)
        self.discretization_of: Geometry | None = None
        self.discretization_points = discretization_points

    def is_discretization_of(self, other_geometry: Geometry) -> bool:
        if self.discretization_of == other_geometry:
            return True
        if self is other_geometry:
            return True
        return False

    def sample_random_uniform_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Samples random points from the discretization_points saved
        in this geometry.

        Args:
            n_points (int): How many points should be sampled from the
                discretization. If n_points is larger than the number of
                discretization points, duplicates will be returned.
            device (Device): The device to sample on.

        Returns:
            array: An array of shape (n_points, dim) containing the sampled points.
        """
        raise NotImplementedError()

    def sample_grid_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Samples points from the discretization_points, the n_points
        will be equally distributed over the number of all points, such
        that every point is returned the same amount.

        Args:
            n_points (int): How many points should be sampled from the
                discretization. If n_points is larger than the number of
                discretization points, duplicates will be returned.
            device (Device): The device to sample on.

        Returns:
            Any: An array of shape (n_points, dim) containing the sampled points.
        """
        raise NotImplementedError()


class BoundaryGeometry(Geometry[TensorType]):
    """The parent class for all built-in boundaries. Can be used just like
    any other Geometry.

    Args:
        geometry (Geometry): The geometry for which this is the boundary of.
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
        return self.geometry.bounding_box()

    def create_boundary(self) -> BoundaryGeometry:
        raise NotImplementedError("Boundary of a boundary geometry is not defined.")

    def normal(self, points, device: Device | str = cpu) -> TensorType:
        """Computes the normal vector at each point in points.

        Args:
            points (array): An array of shape (n_points, dim) containing the points
                at which the normal vector should be computed.
            device (Device): The device to sample on.

        Returns:
            array: An array of shape (n_points, dim) containing the normal vectors
                at the given points.
        """
        raise NotImplementedError()

    def sample_grid(
        self,
        n_points: int,
        device: Device | str = cpu,
        include_normals: bool = False,  # pylint: disable=unused-argument
    ) -> TensorType | tuple[TensorType, TensorType]:
        """Samples a grid along the boundary.

        Args:
            n_points (int): The number of points.
            device (Device): The device to sample on.
            include_normals (bool, optional): Wether also the normal vectors at
                each point should be returned. Defaults to False.

        Returns:
            TensorType | tuple(TensorType, TensorType): The points and normal
                vectors at the points if include_normals is True, otherwise None.
        """
        return super().sample_grid(n_points)

    def sample_random_uniform(
        self,
        n_points: int,
        device: Device | str = cpu,
        include_normals: bool = False,  # pylint: disable=unused-argument
    ) -> TensorType | tuple[TensorType, TensorType]:
        """Samples random uniform points along the boundary.

        Args:
            n_points (int): The number of points.
            device (Device): The device to sample on.
            include_normals (bool, optional): Wether also the normal vectors at
                each point should be returned. Defaults to False.

        Returns:
            TensorType | tuple(TensorType, TensorType): The points and normal
                vectors at the points if include_normals is True, otherwise None.
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
