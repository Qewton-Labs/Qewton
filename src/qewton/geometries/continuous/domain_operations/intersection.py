# import warnings

from typing import Any

from qewton.geometries.base import GEOMETRY_REGISTRY
from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.continuous.domain_operations.sampler_helper import (
    _boundary_grid_with_n,
    _inside_grid_with_n,
    _inside_random_with_n,
    _boundary_random_with_n,
)
from qewton.backends.base import TensorType
from qewton.config.devices import Device, cpu


class IntersectionGeometry(ContinuousGeometry[TensorType]):
    """Implements the logical intersection of two geometries.
    The intersection is implemented via "on time sampling" and does not
    have an explicit representation. When points are sampled they are
    filtered using logical operations.

    Args:
        geometry_a (ContinuousGeometry): The first geometry.
        geometry_b (ContinuousGeometry): The second geometry.
    """

    def __init__(
        self,
        geometry_a: ContinuousGeometry,
        geometry_b: ContinuousGeometry,
    ):
        assert geometry_a.variable == geometry_b.variable
        assert geometry_a.backend == geometry_b.backend, "Backends do not match!"
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b
        super().__init__(variable=geometry_a.variable, backend=geometry_a.backend)

    def contains(self, points):
        in_a = self.geometry_a.contains(points)
        in_b = self.geometry_b.contains(points)
        return self.backend.math.logical_and(in_a, in_b)

    def bounding_box(self):
        bounds_a = self.geometry_a.bounding_box()
        bounds_b = self.geometry_b.bounding_box()
        bounds = []
        for i in range(self.variable.dim):
            bounds.append(max([bounds_a[2 * i], bounds_b[2 * i]]))
            bounds.append(min([bounds_a[2 * i + 1], bounds_b[2 * i + 1]]))
        return self.backend.build_tensor(bounds)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        return _inside_random_with_n(
            self.geometry_a, self.geometry_b, n_points, invert=False, device=device
        )

    def sample_grid(self, n_points: int, device: Device = cpu):
        return _inside_grid_with_n(
            self.geometry_a, self.geometry_b, n_points, invert=False, device=device
        )

    def _get_volume(self):
        # warnings.warn("""Exact volume of this intersection is not known,
        #     will use the estimate: volume = geometry_a.volume.
        #     If you need the exact volume for sampling,
        #     use geometry.set_volume()""")
        return self.geometry_a.volume()

    def create_boundary(self):
        return IntersectionBoundaryGeometry(self)

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
        return IntersectionGeometry(geometry_a, geometry_b)  # type: ignore


class IntersectionBoundaryGeometry(ContinuousBoundaryGeometry):

    def __init__(self, geometry: IntersectionGeometry):
        assert isinstance(geometry, IntersectionGeometry)
        super().__init__(geometry)
        self.geometry: IntersectionGeometry = geometry  # type: ignore

    def contains(self, points):
        in_a = self.geometry.geometry_a.contains(points)
        in_b = self.geometry.geometry_b.contains(points)
        on_a_bound = self.geometry.geometry_a.create_boundary().contains(points)
        on_b_bound = self.geometry.geometry_b.create_boundary().contains(points)
        on_a_part = self.backend.math.logical_and(on_a_bound, in_b)
        on_b_part = self.backend.math.logical_and(on_b_bound, in_a)
        return self.backend.math.logical_or(on_a_part, on_b_part)

    def _get_volume(self):
        # warnings.warn("""Exact volume of this intersection-boundary is not known,
        #     will use the estimate: volume = boundary_a + boundary_b.
        #     If you need the exact volume for sampling,
        #     use geometry.set_volume()""")
        volume_a = self.geometry.geometry_a.create_boundary().volume()
        volume_b = self.geometry.geometry_b.create_boundary().volume()
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
            normals = self.normal(points, device=device)
            return points, normals

    def normal(self, points, device: Device = cpu):
        on_a = self.geometry.geometry_a.create_boundary().contains(points)
        a_normals = self.geometry.geometry_a.create_boundary().normal(
            points, device=device
        )
        b_normals = self.geometry.geometry_b.create_boundary().normal(
            points, device=device
        )
        normals = self.backend.math.where(on_a, a_normals, b_normals)
        return normals

    @classmethod
    def load(cls, data: dict[str, Any]):
        return IntersectionGeometry.load(data).boundary
