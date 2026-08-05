from typing import Any

from qewton.geometries.base import GEOMETRY_REGISTRY
from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.continuous.domain_operations.sampler_helper import (
    _boundary_grid_with_n,
    _boundary_random_with_n,
)
from qewton.backends.base import TensorType
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Bool


class UnionGeometry(ContinuousGeometry[TensorType]):
    """Implements the logical union of two geometries. The union is implemented via
    "on time sampling" and does not have an explicit representation. When points
    are sampled they are filtered using logical operations.

    Args:
        geometry_a (ContinuousGeometry): The first geometry.
        geometry_b (ContinuousGeometry): The second geometry.
        contained (bool): Whether geometry_b is fully contained within geometry_a.
    """

    def __init__(self, geometry_a: ContinuousGeometry, geometry_b: ContinuousGeometry):
        assert geometry_a.variable == geometry_b.variable
        assert geometry_a.shape == geometry_b.shape
        assert geometry_a.backend == geometry_b.backend, "Backends do not match!"
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b
        super().__init__(
            variable=geometry_a.variable,
            shape=geometry_a.shape,
            backend=geometry_a.backend,
        )

    def _get_volume(self):
        volume_a = self.geometry_a.volume()
        volume_b = self.geometry_b.volume()
        return volume_a + volume_b

    def contains(self, points):
        in_a = self.geometry_a.contains(points)
        in_b = self.geometry_b.contains(points)
        return self.backend.math.logical_or(in_a, in_b)

    def bounding_box(self):
        bounds_a = self.geometry_a.bounding_box()
        bounds_b = self.geometry_b.bounding_box()
        bounds = []
        for i in range(self.variable.dim):
            bounds.append(min([bounds_a[2 * i], bounds_b[2 * i]]))
            bounds.append(max([bounds_a[2 * i + 1], bounds_b[2 * i + 1]]))
        return self.backend.build_tensor(bounds)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        # sample n points in both domains
        points_a = self.geometry_a.sample_random_uniform(n_points=n_points, device=device)
        points_b = self.geometry_b.sample_random_uniform(n_points=n_points, device=device)
        # check which points of geometry b are in geometry a
        in_a = self.geometry_a.contains(points_b)
        # approximate volume of this geometry
        volume_approx = self._get_volume()
        volume_a = self.geometry_a.volume()
        volume_ratio = volume_a / volume_approx
        # choose points depending of the proportion of the geometry w.r.t. the
        # whole geometry union
        rand_index = self.backend.random.uniform((n_points, 1), device=device)
        rand_index = self.backend.math.logical_or(in_a, rand_index <= volume_ratio)
        return self.backend.math.where(rand_index, points_a, points_b)

    def sample_grid(self, n_points: int, device: Device = cpu):
        volume_approx = self._get_volume()
        volume_a = self.geometry_a.volume()
        scaled_n = int(self.backend.math.ceil(n_points * volume_a / volume_approx))
        points_a = self.geometry_a.sample_grid(n_points=scaled_n, device=device)
        if n_points > scaled_n:
            return self._sample_in_b(n_points, points_a, device=device)
        return points_a

    def _sample_in_b(self, n, points_a, device: Device = cpu):
        # check how many points from geometry a lay in b, these points will not be used!
        in_b = self.geometry_b.contains(points_a)
        index = self.backend.math.where(self.backend.math.logical_not(in_b))[0]
        scaled_n = n - len(index)
        points_b = self.geometry_b.sample_grid(n_points=scaled_n, device=device)
        return self.backend.math.concatenate([points_a[index,], points_b])

    def create_boundary(self):
        return UnionBoundaryDomain(self)

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
        return UnionGeometry(geometry_a, geometry_b)  # type: ignore


class UnionBoundaryDomain(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry: UnionGeometry):
        assert isinstance(geometry, UnionGeometry)
        assert not isinstance(geometry.geometry_a, ContinuousBoundaryGeometry)
        assert not isinstance(geometry.geometry_b, ContinuousBoundaryGeometry)
        self.overlap_tol = 0.5
        super().__init__(geometry)
        self.geometry: UnionGeometry = geometry  # type: ignore

    def contains(self, points):
        in_a = self.geometry.geometry_a.contains(points)
        in_b = self.geometry.geometry_b.contains(points)
        on_a_bound = self.geometry.geometry_a.boundary.contains(points)
        on_b_bound = self.geometry.geometry_b.boundary.contains(points)
        on_both = self.backend.math.logical_and(on_b_bound, on_a_bound)
        on_a_part = self.backend.math.logical_and(
            on_a_bound, self.backend.math.logical_not(in_b)
        )
        on_b_part = self.backend.math.logical_and(
            on_b_bound, self.backend.math.logical_not(in_a)
        )

        # if on the both lay on both boundaries it could still happen that
        # the boundary is in the inside of the union, this we can only check
        # via a normal test
        overlap_points = self.backend.math.ones_like(on_both, dtype=Bool)
        if self.backend.math.any(on_both):
            index_tensor = on_both.copy().flatten()

            normals_a = self.geometry.geometry_a.boundary.normal(points[index_tensor])
            normals_b = self.geometry.geometry_b.boundary.normal(points[index_tensor])

            inner_product_ok = (
                self.backend.math.sum(normals_a * normals_b, axis=-1, keepdims=True)
                >= self.overlap_tol
            )
            overlap_points[index_tensor] = inner_product_ok

        default_check = self.backend.math.logical_or(
            on_a_part, self.backend.math.logical_or(on_b_part, on_both)
        )
        return self.backend.math.logical_and(default_check, overlap_points)

    def _get_volume(self):
        volume_a = self.geometry.geometry_a.boundary.volume()
        volume_b = self.geometry.geometry_b.boundary.volume()
        return volume_a + volume_b

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        points = _boundary_random_with_n(
            self,
            self.geometry.geometry_a,
            self.geometry.geometry_b,
            n_points,
            device=device,
        )
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        points = _boundary_grid_with_n(
            self,
            self.geometry.geometry_a,
            self.geometry.geometry_b,
            n_points,
            device=device,
        )
        if include_normals:
            normals = self.normal(points)
            return points, normals
        return points

    def normal(self, points, device: Device = cpu):
        a_normals = self.geometry.geometry_a.boundary.normal(points, device=device)
        b_normals = self.geometry.geometry_b.boundary.normal(points, device=device)
        on_a = self.geometry.geometry_a.boundary.contains(points)
        normals = self.backend.math.where(on_a, a_normals, b_normals)
        return normals

    @classmethod
    def load(cls, data: dict[str, Any]):
        return UnionGeometry.load(data).boundary
