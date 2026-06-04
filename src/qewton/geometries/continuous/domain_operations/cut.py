import warnings
import numpy as np

from ..base import ContinuousGeometry, ContinuousBoundaryGeometry
from .sampler_helper import (
    _boundary_grid_with_n,
    _inside_grid_with_n,
    _inside_random_with_n,
    _boundary_random_with_n,
)


class CutGeometry(ContinuousGeometry):
    """Implements the logical cut of two geometries.

    Parameters
    ----------
    geometry_a : ContinuousGeometry
        The first geometry.
    geometry_b : ContinuousGeometry
        The second geometry.
    contained : bool
        Whether geometry_b is fully contained within geometry_a.
    """

    def __init__(
        self,
        geometry_a: ContinuousGeometry,
        geometry_b: ContinuousGeometry,
        contained=False,
    ):
        assert geometry_a.variable == geometry_b.variable
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b
        self.contained = contained
        super().__init__(variable=geometry_a.variable)

    def contains(self, points):
        in_a = self.geometry_a.contains(points)
        in_b = self.geometry_b.contains(points)
        return np.logical_and(in_a, np.logical_not(in_b))

    def bounding_box(self):
        return self.geometry_a.bounding_box()

    def sample_random_uniform(self, n_points: int):
        return _inside_random_with_n(
            self.geometry_a,
            self.geometry_b,
            n_points,
            invert=True,
        )

    def sample_grid(self, n_points: int):
        return _inside_grid_with_n(
            self.geometry_a,
            self.geometry_b,
            n_points,
            invert=True,
        )

    def _get_volume(self):
        if not self.contained:
            warnings.warn(
                """Exact volume of this cut is not known, will use the
                estimate: volume = geometry_a.volume.
                If you need the exact volume for sampling,
                use geometry.set_volume()"""
            )
            return self.geometry_a.volume()
        volume_a = self.geometry_a.volume()
        volume_b = self.geometry_b.volume()
        return volume_a - volume_b

    def create_boundary(self):
        return CutBoundaryGeometry(self)


class CutBoundaryGeometry(ContinuousBoundaryGeometry):

    def __init__(self, geometry: CutGeometry):
        assert isinstance(geometry, CutGeometry)
        super().__init__(geometry)
        self.geometry: CutGeometry = geometry  # type: ignore

    def contains(self, points):
        in_a = self.geometry.geometry_a.contains(points)
        in_b = self.geometry.geometry_b.contains(points)
        on_a_bound = self.geometry.geometry_a.boundary.contains(points)
        on_b_bound = self.geometry.geometry_b.boundary.contains(points)
        on_a_part = np.logical_and(on_a_bound, np.logical_not(in_b))
        on_b_part = np.logical_and(on_b_bound, in_a)
        on_b_part = np.logical_and(on_b_part, np.logical_not(on_a_bound))
        return np.logical_or(on_a_part, on_b_part)

    def _get_volume(self):
        if not self.geometry.contained:
            warnings.warn(
                """Exact volume of this boundary is not known, 
                will use the estimate: 
                volume = geometry_a.boundary.volume + geometry_b.boundary.volume.
                If you need the exact volume for sampling,
                use geometry.set_volume()."""
            )
        volume_a = self.geometry.geometry_a.boundary.volume()
        volume_b = self.geometry.geometry_b.boundary.volume()
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
        a_normals = self.geometry.geometry_a.boundary.normal(points)
        b_normals = self.geometry.geometry_b.boundary.normal(points)
        on_a = self.geometry.geometry_a.boundary.contains(points)
        normals = np.where(on_a, a_normals, -b_normals)
        return normals
