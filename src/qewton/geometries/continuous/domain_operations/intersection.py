import warnings
import numpy as np

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


class IntersectionGeometry(ContinuousGeometry):
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
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b
        super().__init__(variable=geometry_a.variable)

    def contains(self, points):
        in_a = self.geometry_a.contains(points)
        in_b = self.geometry_b.contains(points)
        return np.logical_and(in_a, in_b)

    def bounding_box(self):
        bounds_a = self.geometry_a.bounding_box()
        bounds_b = self.geometry_b.bounding_box()
        bounds = []
        for i in range(self.variable.dim):
            bounds.append(max([bounds_a[2 * i], bounds_b[2 * i]]))
            bounds.append(min([bounds_a[2 * i + 1], bounds_b[2 * i + 1]]))
        return np.array(bounds)

    def sample_random_uniform(self, n_points: int):
        return _inside_random_with_n(
            self.geometry_a,
            self.geometry_b,
            n_points,
            invert=False,
        )

    def sample_grid(self, n_points: int):
        return _inside_grid_with_n(
            self.geometry_a,
            self.geometry_b,
            n_points,
            invert=False,
        )

    def _get_volume(self):
        warnings.warn("""Exact volume of this intersection is not known,
            will use the estimate: volume = geometry_a.volume.
            If you need the exact volume for sampling,
            use geometry.set_volume()""")
        return self.geometry_a.volume()

    def create_boundary(self):
        return IntersectionBoundaryGeometry(self)


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
        on_a_part = np.logical_and(on_a_bound, in_b)
        on_b_part = np.logical_and(on_b_bound, in_a)
        return np.logical_or(on_a_part, on_b_part)

    def _get_volume(self):
        warnings.warn("""Exact volume of this intersection-boundary is not known,
            will use the estimate: volume = boundary_a + boundary_b.
            If you need the exact volume for sampling,
            use geometry.set_volume()""")
        volume_a = self.geometry.geometry_a.create_boundary().volume()
        volume_b = self.geometry.geometry_b.create_boundary().volume()
        return volume_a + volume_b

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        points = _boundary_random_with_n(
            self,
            self.geometry.geometry_a,
            self.geometry.geometry_b,
            n_points,
        )
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        points = _boundary_grid_with_n(
            self,
            self.geometry.geometry_a,
            self.geometry.geometry_b,
            n_points,
        )
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def normal(self, points):
        on_a = self.geometry.geometry_a.create_boundary().contains(points)
        a_normals = self.geometry.geometry_a.create_boundary().normal(points)
        b_normals = self.geometry.geometry_b.create_boundary().normal(points)
        normals = np.where(on_a, a_normals, b_normals)
        return normals
