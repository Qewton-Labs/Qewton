import numpy as np

from ..base import ContinuousGeometry, ContinuousBoundaryGeometry
from ....config.variables import Variable


class Parallelogram(ContinuousGeometry):
    """Class for parallelograms in 2D.

    Parameters
    ----------
    variable : Variable
        The variable representing the underlying space.
    origin : array_like
        One corner of the parallelogram.
    corner_1 : array_like
        A second corner adjacent to `origin`.
    corner_2 : array_like
        A third corner adjacent to `origin`.
    """

    def __init__(
        self,
        variable: Variable,
        origin: np.ndarray | list[float] | tuple[float, float],
        corner_1: np.ndarray | list[float] | tuple[float, float],
        corner_2: np.ndarray | list[float] | tuple[float, float],
    ):
        assert variable.dim == 2
        self.origin = self._to_vector(origin)
        self.corner_1 = self._to_vector(corner_1)
        self.corner_2 = self._to_vector(corner_2)
        super().__init__(variable=variable)

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        points = points.reshape(-1, 2)
        origin = self.origin
        dir_1 = self.corner_1 - origin
        dir_2 = self.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.solve_barycentric(relative, dir_1, dir_2)
        inside = np.logical_and.reduce(
            (0 <= bary_x, bary_x <= 1, 0 <= bary_y, bary_y <= 1)
        )
        return inside.reshape(-1, 1)

    def bounding_box(self):
        origin = self.origin
        corner_1 = self.corner_1
        corner_2 = self.corner_2
        corner_3 = corner_1 + corner_2 - origin
        corners = np.vstack((origin, corner_1, corner_2, corner_3))
        mins = np.min(corners, axis=0)
        maxs = np.max(corners, axis=0)
        return np.concatenate((mins[0:1], maxs[0:1], mins[1:2], maxs[1:2]))

    def sample_random_uniform(self, n_points: int):
        bary_coords = np.random.rand(n_points, 2)
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + bary_coords[:, :1] * dir_1 + bary_coords[:, 1:] * dir_2
        return points

    def sample_grid(self, n_points: int):
        n_side = int(np.ceil(np.sqrt(n_points)))
        u = np.linspace(0, 1, n_side)
        v = np.linspace(0, 1, n_side)
        uu, vv = np.meshgrid(u, v)
        bary_coords = np.column_stack((uu.ravel(), vv.ravel()))
        points = (
            self.origin
            + bary_coords[:, :1] * (self.corner_1 - self.origin)
            + bary_coords[:, 1:] * (self.corner_2 - self.origin)
        )
        return points[:n_points]

    def solve_barycentric(
        self,
        relative: np.ndarray,
        dir_1: np.ndarray,
        dir_2: np.ndarray,
    ):
        det = dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0]
        if det == 0:
            raise ValueError("Parallelogram corners must not be collinear.")
        bary_x = (dir_2[1] * relative[:, 0] - dir_2[0] * relative[:, 1]) / det
        bary_y = (-dir_1[1] * relative[:, 0] + dir_1[0] * relative[:, 1]) / det
        return bary_x, bary_y

    def _to_vector(self, vector):
        array = np.asarray(vector, dtype=float).reshape(
            2,
        )
        if array.shape != (2,):
            raise ValueError("Parallelogram corners must be 2D vectors.")
        return array

    def _get_volume(self):
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        det = dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0]
        return abs(det)

    def create_boundary(self):
        return ParallelogramBoundary(self)


class ParallelogramBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Parallelogram):
        super().__init__(geometry)
        self.geometry: Parallelogram = geometry  # type: ignore

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.geometry.solve_barycentric(relative, dir_1, dir_2)
        x_close = self._bary_coords_close_to_0_or_1(bary_x, bary_y)
        y_close = self._bary_coords_close_to_0_or_1(bary_y, bary_x)
        return np.logical_or(x_close, y_close).reshape(-1, 1)

    def _get_volume(self):
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        side_length1 = np.linalg.norm(dir_1)
        side_length2 = np.linalg.norm(dir_2)
        return 2 * (side_length1 + side_length2)

    def _bary_coords_close_to_0_or_1(
        self, bary_coord1: np.ndarray, bary_coord2: np.ndarray
    ):
        between_0_1 = np.logical_and(0 <= bary_coord2, bary_coord2 <= 1)
        close_to_0 = np.isclose(bary_coord1, 0.0)
        close_to_1 = np.isclose(bary_coord1, 1.0)
        return np.logical_and(np.logical_or(close_to_0, close_to_1), between_0_1)

    def sample_random_uniform(self, n_points: int):
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        corner_3 = self.geometry.corner_1 + self.geometry.corner_2 - origin
        side_lengths = np.array(
            [
                np.linalg.norm(dir_1),
                np.linalg.norm(dir_2),
                np.linalg.norm(dir_1),
                np.linalg.norm(dir_2),
            ],
            dtype=float,
        )
        total_length = np.sum(side_lengths)
        positions = np.random.rand(n_points) * total_length
        points = np.empty((n_points, 2), dtype=float)
        breaks = np.cumsum(side_lengths)

        mask = positions < breaks[0]
        t = positions[mask] / side_lengths[0]
        points[mask] = origin + t[:, None] * dir_1

        mask = np.logical_and(positions >= breaks[0], positions < breaks[1])
        t = (positions[mask] - breaks[0]) / side_lengths[1]
        points[mask] = self.geometry.corner_1 + t[:, None] * dir_2

        mask = np.logical_and(positions >= breaks[1], positions < breaks[2])
        t = (positions[mask] - breaks[1]) / side_lengths[2]
        points[mask] = corner_3 - t[:, None] * dir_1

        mask = positions >= breaks[2]
        t = (positions[mask] - breaks[2]) / side_lengths[3]
        points[mask] = self.geometry.corner_2 - t[:, None] * dir_2

        return points

    def sample_grid(self, n_points: int):
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        corner_3 = self.geometry.corner_1 + self.geometry.corner_2 - origin
        side_lengths = np.array(
            [
                np.linalg.norm(dir_1),
                np.linalg.norm(dir_2),
                np.linalg.norm(dir_1),
                np.linalg.norm(dir_2),
            ],
            dtype=float,
        )
        total_length = np.sum(side_lengths)
        positions = np.linspace(0, total_length, n_points + 1)[:-1]
        points = np.empty((n_points, 2), dtype=float)
        breaks = np.cumsum(side_lengths)

        mask = positions < breaks[0]
        t = positions[mask] / side_lengths[0]
        points[mask] = origin + t[:, None] * dir_1

        mask = np.logical_and(positions >= breaks[0], positions < breaks[1])
        t = (positions[mask] - breaks[0]) / side_lengths[1]
        points[mask] = self.geometry.corner_1 + t[:, None] * dir_2

        mask = np.logical_and(positions >= breaks[1], positions < breaks[2])
        t = (positions[mask] - breaks[1]) / side_lengths[2]
        points[mask] = corner_3 - t[:, None] * dir_1

        mask = positions >= breaks[2]
        t = (positions[mask] - breaks[2]) / side_lengths[3]
        points[mask] = self.geometry.corner_2 - t[:, None] * dir_2

        return points

    def normal(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.geometry.solve_barycentric(relative, dir_1, dir_2)
        normal_dir_1 = self._get_normal_direction(dir_1)
        normal_dir_2 = -self._get_normal_direction(dir_2)
        normals = np.zeros_like(points)
        x_close_0_or_1 = np.logical_or(np.isclose(bary_x, 0.0), np.isclose(bary_x, 1.0))
        y_close_0_or_1 = np.logical_or(np.isclose(bary_y, 0.0), np.isclose(bary_y, 1.0))
        normals[x_close_0_or_1] += normal_dir_2
        normals[y_close_0_or_1] += normal_dir_1
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / norms

    def _get_normal_direction(self, direction: np.ndarray):
        normal = np.array([-direction[1], direction[0]], dtype=float)
        return normal / np.linalg.norm(normal)
