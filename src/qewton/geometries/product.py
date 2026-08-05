from typing import Any

from qewton.geometries.base import BoundaryGeometry, Geometry, GEOMETRY_REGISTRY
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
        self.backend.math.concatenate(
            [self.geometry_a.bounding_box(), self.geometry_b.bounding_box()]
        )

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        points_a = self.geometry_a.sample_random_uniform(n_points, device)
        points_b = self.geometry_b.sample_random_uniform(n_points, device)
        return self.backend.math.concatenate([points_a, points_b], axis=-1)

    def save(self) -> dict[str, Any]:
        combi_dict = {
            "class": self.__class__.__name__,
            "geometry_a": self.geometry_a.save(),
            "geometry_b": self.geometry_b.save(),
        }
        return combi_dict

    @classmethod
    def load(cls, data: dict[str, Any]):
        geometry_a = GEOMETRY_REGISTRY[data["geometry_a"]["class"]].load(
            data["geometry_a"]
        )
        geometry_b = GEOMETRY_REGISTRY[data["geometry_b"]["class"]].load(
            data["geometry_b"]
        )
        return ProductGeometry(geometry_a, geometry_b)
