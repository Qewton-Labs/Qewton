import numpy as np

from ..base import ContinuousGeometry, ContinuousBoundaryGeometry
from ....config.variables import Variable


class Circle(ContinuousGeometry):
    """Class for circles."""

    def __init__(
        self,
        variable: Variable,
        center: np.ndarray | list[float] | tuple[float, float],
        radius: float,
    ):
        assert variable.dim == 2
        assert len(center) == 2
        if isinstance(center, (list, tuple)):
            center = np.array(center)
        self.center: np.ndarray = center
        self.radius = radius
        super().__init__(variable=variable)

    def contains(self, points):
        norm = np.linalg.norm(points - self.center, axis=1).reshape(-1, 1)
        return norm <= self.radius

    def bounding_box(self):
        bounds = []
        for i in range(2):
            bounds.append(self.center[i] - self.radius)
            bounds.append(self.center[i] + self.radius)
        return np.array(bounds)

    def sample_random_uniform(self, n_points: int):
        r = self.radius * np.sqrt(np.random.rand(n_points, 1))
        phi = 2 * np.pi * np.random.rand(n_points, 1)
        points = np.concat([r * np.cos(phi), r * np.sin(phi)], axis=-1)
        points += self.center[None, :]
        return points

    def sample_grid(self, n_points: int):
        grid = self._equidistant_points_in_circle(n_points)
        points = self.radius * grid
        points += self.center[None, :]
        return points

    def _equidistant_points_in_circle(self, n_points: int):
        # use a sunflower seed arrangement:
        # https://demonstrations.wolfram.com/SunflowerSeedArrangements/
        gr = (np.sqrt(5) + 1) / 2.0  # golden ratio
        points = np.arange(1, n_points + 1)
        phi = (2 * np.pi / gr) * points
        radius = np.sqrt(points - 0.5) / np.sqrt(n_points + 0.5)
        points = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
        return points

    def _get_volume(self):
        volume = np.pi * self.radius**2
        return volume

    def create_boundary(self):
        return CircleBoundary(self)


class CircleBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry):
        assert isinstance(geometry, Circle)
        super().__init__(geometry)
        self.geometry: Circle = geometry  # type: ignore

    def contains(self, points):
        norm = np.linalg.norm(points - self.geometry.center, axis=1).reshape(-1, 1)
        return np.isclose(norm, self.geometry.radius)

    def sample_random_uniform(self, n_points: int):
        phi = 2 * np.pi * np.random.rand(n_points, 1).reshape(-1, 1)
        points = np.concat(
            [self.geometry.radius * np.cos(phi), self.geometry.radius * np.sin(phi)],
            axis=-1,
        )
        points += self.geometry.center[None, :]
        return points

    def sample_grid(self, n_points: int):
        phi = np.linspace(0, 2 * np.pi, n_points + 1)[:-1].reshape(-1, 1)
        points = np.concat(
            [self.geometry.radius * np.cos(phi), self.geometry.radius * np.sin(phi)],
            axis=-1,
        )
        points += self.geometry.center[None, :]
        return points

    def normal(self, points):
        normal = points - self.geometry.center[None, :]
        return (normal / self.geometry.radius).reshape(-1, 2)

    def _get_volume(self):
        volume = 2 * np.pi * self.geometry.radius
        return volume
