import numpy as np

from ..base import ContinuousGeometry, ContinuousBoundaryGeometry
from .sampler_helper import _boundary_grid_with_n, _boundary_random_with_n


class UnionGeometry(ContinuousGeometry):
    """Implements the logical union of two domains.

    Parameters
    ----------
    geometry_a : geometry
        The first geometry.
    geometry_b : geometry
        The second geometry.
    """

    def __init__(self, geometry_a: ContinuousGeometry, geometry_b: ContinuousGeometry):
        assert geometry_a.variable == geometry_b.variable
        assert geometry_a.shape == geometry_b.shape
        self.geometry_a = geometry_a
        self.geometry_b = geometry_b
        super().__init__(variable=geometry_a.variable, shape=geometry_a.shape)

    def _get_volume(self):
        volume_a = self.geometry_a.volume()
        volume_b = self.geometry_b.volume()
        return volume_a + volume_b

    def contains(self, points):
        in_a = self.geometry_a.contains(points)
        in_b = self.geometry_b.contains(points)
        return np.logical_or(in_a, in_b)

    def bounding_box(self):
        bounds_a = self.geometry_a.bounding_box()
        bounds_b = self.geometry_b.bounding_box()
        bounds = []
        for i in range(self.variable.dim):
            bounds.append(min([bounds_a[2 * i], bounds_b[2 * i]]))
            bounds.append(max([bounds_a[2 * i + 1], bounds_b[2 * i + 1]]))
        return np.array(bounds)

    def sample_random_uniform(self, n_points: int):
        # sample n points in both domains
        points_a = self.geometry_a.sample_random_uniform(n_points=n_points)
        points_b = self.geometry_b.sample_random_uniform(n_points=n_points)
        # check which points of geometry b are in geometry a
        in_a = self.geometry_a.contains(points_b)
        # approximate volume of this geometry
        volume_approx = self._get_volume()
        volume_a = self.geometry_a.volume()
        volume_ratio = volume_a / volume_approx
        # choose points depending of the proportion of the geometry w.r.t. the
        # whole geometry union
        rand_index = np.random.rand(n_points, 1)
        rand_index = np.logical_or(in_a, rand_index <= volume_ratio)
        return np.where(rand_index, points_a, points_b)

    def sample_grid(self, n_points: int):
        volume_approx = self._get_volume()
        volume_a = self.geometry_a.volume()
        scaled_n = int(np.ceil(n_points * volume_a / volume_approx))
        points_a = self.geometry_a.sample_grid(n_points=scaled_n)
        if n_points > scaled_n:
            return self._sample_in_b(n_points, points_a)
        return points_a

    def _sample_in_b(self, n, points_a):
        # check how many points from geometry a lay in b, these points will not be used!
        in_b = self.geometry_b.contains(points_a)
        index = np.where(np.logical_not(in_b))[0]
        scaled_n = n - len(index)
        points_b = self.geometry_b.sample_grid(n_points=scaled_n)
        return np.concatenate([points_a[index,], points_b])

    def create_boundary(self):
        return UnionBoundaryDomain(self)


class UnionBoundaryDomain(ContinuousBoundaryGeometry):

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
        on_both: np.ndarray = np.logical_and(on_b_bound, on_a_bound)
        on_a_part = np.logical_and(on_a_bound, np.logical_not(in_b))
        on_b_part = np.logical_and(on_b_bound, np.logical_not(in_a))

        # if on the both lay on both boundaries it could still happen that
        # the boundary is in the inside of the union, this we can only check
        # via a normal test
        overlap_points = np.ones_like(on_both, dtype=np.bool)
        if np.any(on_both):
            index_tensor = on_both.copy().flatten()

            normals_a = self.geometry.geometry_a.boundary.normal(points[index_tensor])
            normals_b = self.geometry.geometry_b.boundary.normal(points[index_tensor])

            inner_product_ok = (
                np.sum(normals_a * normals_b, axis=-1, keepdims=True) >= self.overlap_tol
            )
            overlap_points[index_tensor] = inner_product_ok

        default_check = np.logical_or(on_a_part, np.logical_or(on_b_part, on_both))
        return np.logical_and(default_check, overlap_points)

    def _get_volume(self):
        volume_a = self.geometry.geometry_a.boundary.volume()
        volume_b = self.geometry.geometry_b.boundary.volume()
        return volume_a + volume_b

    def sample_random_uniform(self, n_points: int):
        return _boundary_random_with_n(
            self, self.geometry.geometry_a, self.geometry.geometry_b, n_points
        )

    def sample_grid(self, n_points: int):
        return _boundary_grid_with_n(
            self, self.geometry.geometry_a, self.geometry.geometry_b, n_points
        )

    def normal(self, points):
        a_normals = self.geometry.geometry_a.boundary.normal(points)
        b_normals = self.geometry.geometry_b.boundary.normal(points)
        on_a = self.geometry.geometry_a.boundary.contains(points)
        normals = np.where(on_a, a_normals, b_normals)
        return normals
