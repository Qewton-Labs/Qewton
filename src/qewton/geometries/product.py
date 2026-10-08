from qewton.geometries.base import BoundaryGeometry, Geometry, vertex_distance_for
from qewton.config.variables import Variable
from qewton.config.devices import Device, cpu
from qewton.backends.base import TensorType


class ProductGeometry(Geometry[TensorType]):

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
        """The non-product geometries this product is built from, in order.
        Nested products are flattened."""
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
        """Meshes every factor and combines them into a ProductMeshGeometry.

        Args:
            max_vertex_distance (float | dict[Variable, float] | None, optional):
                How fine each factor mesh should be. A dict sets it per factor
                variable; factors missing from it use None. Defaults to None.
            device (Device, optional): Where the mesh is created.
                Defaults to cpu.
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
        raise NotImplementedError(
            "Can not build the boundary directly, instead build a product from the"
            " boundary geometries by hand."
        )

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        raise NotImplementedError("Create a product of samplers instead to build a grid.")

    def _get_volume(self):
        return self.geometry_a.volume() * self.geometry_b.volume()

    def contains(self, points):
        dim_a = self.geometry_a.dim
        return self.backend.math.logical_and(
            self.geometry_a.contains(points=points[..., :dim_a]),
            self.geometry_b.contains(points=points[..., dim_a:]),
        )

    def bounding_box(self):
        return self.backend.math.concatenate(
            [self.geometry_a.bounding_box(), self.geometry_b.bounding_box()]
        )

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        points_a = self.geometry_a.sample_random_uniform(n_points, device)
        points_b = self.geometry_b.sample_random_uniform(n_points, device)
        return self.backend.math.concatenate([points_a, points_b], axis=-1)
