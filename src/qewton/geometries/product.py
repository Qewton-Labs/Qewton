"""Cartesian product of two geometries."""

from qewton.geometries.base import BoundaryGeometry, Geometry, vertex_distance_for
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.backends.base import TensorType


class ProductGeometry(Geometry[TensorType]):
    """Cartesian product of two geometries.

    Usually created with ``geometry_a * geometry_b``. Points of the product
    are the concatenation of a point of `geometry_a` and a point of
    `geometry_b`.

    Args:
        geometry_a: The first factor.
        geometry_b: The second factor. Must use the same backend as
            `geometry_a` and a different variable.

    Attributes:
        geometry_a (Geometry): The first factor.
        geometry_b (Geometry): The second factor.
        factors (list of Geometry): The non-product geometries this product is
            built from, in order. Nested products are flattened.
    """

    def __init__(self, geometry_a: Geometry, geometry_b: Geometry):
        assert (
            geometry_a.backend == geometry_b.backend
        ), "Both geometries need the same backend"
        assert (
            geometry_a.variable != geometry_b.variable
        ), "Both geometries can not belong to the same variable"
        super().__init__(
            variable=geometry_a.variable * geometry_b.variable,
            dim=geometry_a.dim + geometry_b.dim,  # type: ignore
            backend=geometry_a.backend,
        )
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b

    @property
    def factors(self) -> list[Geometry]:
        factors = []
        for geometry in (self.geometry_a, self.geometry_b):
            if isinstance(geometry, ProductGeometry):
                factors.extend(geometry.factors)
            else:
                factors.append(geometry)
        return factors

    def create_mesh(
        self,
        max_vertex_distance: float | dict[Variable, float] | None = None,
        device: Device = cpu,
    ):
        """Mesh every factor and combine the meshes into a product mesh.

        Args:
            max_vertex_distance: How fine each factor mesh is. A dict sets the
                value per factor variable; factors missing from it get
                ``None``.
            device: The device the mesh is created on.

        Returns:
            ProductMeshGeometry: The product of the factor meshes, see
                :class:`~qewton.geometries.discrete.product_mesh_geometry.ProductMeshGeometry`.
        """
        from qewton.geometries.discrete.product_mesh_geometry import (
            ProductMeshGeometry,
        )

        factor_meshes = [
            factor.create_mesh(
                vertex_distance_for(max_vertex_distance, factor.variable), device
            )
            for factor in self.factors
        ]
        return ProductMeshGeometry(
            factor_meshes,
            variable=self.variable,
            discretization_of=self,
            device=device,
            backend=self.backend,
        )

    def create_boundary(self) -> BoundaryGeometry:
        """Raise, since the boundary of a product is not built directly.

        Raises:
            NotImplementedError: Always. Build a product of the factor
                boundaries instead.
        """
        raise NotImplementedError(
            "Can not build the boundary directly, instead build a product from the"
            " boundary geometries by hand."
        )

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        """Raise, since grids on products are built from product samplers.

        Raises:
            NotImplementedError: Always. Use a product of samplers instead.
        """
        raise NotImplementedError("Create a product of samplers instead to build a grid.")

    def _get_volume(self):
        return self.geometry_a.volume() * self.geometry_b.volume()

    def contains(self, points):
        """Check for every point whether it lies inside both factors.

        Args:
            points (TensorType): The points to check; the first
                ``geometry_a.dim`` coordinates belong to `geometry_a`, the rest
                to `geometry_b`.

        Returns:
            TensorType: Boolean tensor, ``True`` where the point lies inside.
        """
        dim_a = self.geometry_a.dim
        return self.backend.math.logical_and(
            self.geometry_a.contains(points=points[..., :dim_a]),
            self.geometry_b.contains(points=points[..., dim_a:]),
        )

    def bounding_box(self):
        """Return the concatenated bounding boxes of both factors.

        Returns:
            TensorType: Tensor of the form ``[axis_1_min, axis_1_max, ...]``.
        """
        return self.backend.math.concatenate(
            [self.geometry_a.bounding_box(), self.geometry_b.bounding_box()]
        )

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        """Sample random points in both factors and concatenate them.

        Args:
            n_points: The number of points to sample.
            device: The device to sample on.

        Returns:
            TensorType: The sampled points, shape ``(n_points, dim)``.
        """
        points_a = self.geometry_a.sample_random_uniform(n_points, device)
        points_b = self.geometry_b.sample_random_uniform(n_points, device)
        return self.backend.math.concatenate([points_a, points_b], axis=-1)
