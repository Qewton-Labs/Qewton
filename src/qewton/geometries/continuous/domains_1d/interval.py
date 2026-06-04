import numpy as np

from qewton.geometries.continuous.base import ContinuousGeometry, ContinuousBoundaryGeometry
from qewton.geometries.discrete.mesh_domain import MeshGeometry, Mesh
from qewton.config.variables import Variable


class Interval(ContinuousGeometry):
    """Interval class"""

    def __init__(self, variable: Variable, lower_bound, upper_bound):
        assert variable.dim == 1
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        super().__init__(variable=variable)

    def contains(self, points):
        bigger_then_low = points >= self.lower_bound
        smaller_then_up = points <= self.upper_bound
        return np.logical_and(bigger_then_low, smaller_then_up).reshape(-1, 1)

    def create_mesh(self, max_vertex_distance: float | None = None):
        size = self.upper_bound - self.lower_bound
        if max_vertex_distance is None:
            max_vertex_distance = size
        vertices = np.linspace(
            self.lower_bound,
            self.upper_bound,
            1 + max(1, np.ceil(size / max_vertex_distance)),
        )
        cells = [[i, i + 1] for i in range(len(vertices) - 1)]
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=cells),
            discretization_of=self,
        )

    def sample_random_uniform(self, n_points: int):
        points = np.random.rand(n_points, 1)
        points *= self.upper_bound - self.lower_bound
        points += self.lower_bound
        return points

    def sample_grid(self, n_points: int):
        points = np.linspace(self.lower_bound, self.upper_bound, n_points + 2)[1:-1, None]
        return points

    def bounding_box(self):
        return np.array([self.lower_bound, self.upper_bound])

    def _get_volume(self):
        return self.upper_bound - self.lower_bound

    def create_boundary(self):
        return IntervalBoundary(self)

    @property
    def boundary_left(self):
        """Returns only the left boundary value, useful for the definition
        of initial conditions.
        """
        return IntervalSingleBoundaryPoint(self, side=self.lower_bound)

    @property
    def boundary_right(self):
        """Returns only the left boundary value, useful for the definition
        of end conditions.
        """
        return IntervalSingleBoundaryPoint(self, side=self.upper_bound, normal_vec=1)


class IntervalBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry):
        assert isinstance(geometry, Interval)
        super().__init__(geometry)
        self.geometry: Interval = geometry  # type: ignore

    def contains(self, points):
        close_left = np.isclose(points, self.geometry.lower_bound)
        close_right = np.isclose(points, self.geometry.upper_bound)
        return np.logical_or(close_left, close_right).reshape(-1, 1)

    def create_mesh(
        self, max_vertex_distance: float | None = None  # pylint: disable=unused-argument
    ):
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(
                vertices=[self.geometry.lower_bound, self.geometry.upper_bound],
                cells=[],
            ),
            discretization_of=self,
        )

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        rand_side = np.random.rand(n_points, 1)
        random_boundary_index = rand_side < 0.5
        points = np.where(
            random_boundary_index, self.geometry.lower_bound, self.geometry.upper_bound
        )
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        lb = self.geometry.lower_bound
        ub = self.geometry.upper_bound
        b_index = np.array([0, 1], dtype=bool).repeat(int(n_points / 2.0) + 1)
        points = np.where(b_index[:n_points], lb, ub)
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def normal(self, points):
        close_left = np.isclose(points, self.geometry.lower_bound)
        return np.where(close_left, -1, 1).reshape(-1, 1)

    def _get_volume(self):
        return 2


class IntervalSingleBoundaryPoint(ContinuousBoundaryGeometry):

    def __init__(self, geometry, side, normal_vec=-1):
        assert isinstance(geometry, Interval)
        super().__init__(geometry)
        self.side = side
        self.normal_vec = normal_vec

    def contains(self, points):
        inside = np.isclose(points, self.side)
        return inside.reshape(-1, 1)

    def create_mesh(
        self, max_vertex_distance: float | None = None  # pylint: disable=unused-argument
    ):
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(
                vertices=[self.side],
                cells=[],
            ),
            discretization_of=self,
        )

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        points = self.side * np.ones((n_points, 1))
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        return self.sample_random_uniform(
            n_points=n_points, include_normals=include_normals
        )

    def normal(self, points):
        return self.normal_vec * np.ones((len(points), 1))

    def _get_volume(self):
        return 1
