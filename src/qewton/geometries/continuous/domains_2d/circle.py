import numpy as np

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.discrete.mesh_domain import MeshGeometry, Mesh
from qewton.config.variables import Variable


class Circle(ContinuousGeometry):
    """Class for circles represented by center and radius.

    Args:
        variable (Variable): The variable associated with the circle, must be 2D.
        center (np.ndarray | list[float] | tuple[float, float]):
            The center of the circle.
        radius (float): The radius of the circle.
    """

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

    def create_mesh(self, max_vertex_distance: float | None = None) -> MeshGeometry:
        if max_vertex_distance is None:
            power_n = 4
            n = 16
        else:
            power_n = int(np.ceil(np.log2(2 * np.pi * self.radius / max_vertex_distance)))
            n = int(max(4, 2**power_n))
        vertices = []
        triangles = []
        while_counter = 0
        while n >= 4:
            current_radius = ((power_n - while_counter) / power_n) ** 1.3 * self.radius
            for i in range(n):
                angle = 2 * np.pi * i / n
                vertices.append(
                    [current_radius * np.cos(angle), current_radius * np.sin(angle)]
                )
            # connect inner ring to outside:
            #   O---O---O
            #   |  / \  |
            #   | /   \ |
            #   |/     \|
            #   o-------o
            v_c = len(vertices) - 1
            if while_counter > 0:
                for i in range(n - 1):
                    triangles.append([v_c - i, v_c - 2 * i - n, v_c - 2 * i - n - 1])
                    triangles.append([v_c - i, v_c - 2 * i - n - 1, v_c - 2 * i - n - 2])
                    triangles.append([v_c - i, v_c - i - 1, v_c - 2 * i - n - 2])
                # Last point is a bit more tricky since have connect to the period jump
                triangles.append([v_c - n + 1, v_c - 3 * n + 2, v_c - 3 * n + 1])
                triangles.append([v_c - n + 1, v_c - n, v_c - 3 * n + 1])
                triangles.append([v_c, v_c - n + 1, v_c - n])
            # Reduce number of points in next layer
            n //= 2
            while_counter += 1

        v_count = len(vertices) - 1
        triangles.append([v_count, v_count - 1, v_count - 2])
        triangles.append([v_count, v_count - 2, v_count - 3])

        triangles = np.asarray(triangles)
        vertices = np.asarray(vertices)

        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=triangles),
            discretization_of=self,
        )

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

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        phi = 2 * np.pi * np.random.rand(n_points, 1).reshape(-1, 1)
        points = np.concat(
            [self.geometry.radius * np.cos(phi), self.geometry.radius * np.sin(phi)],
            axis=-1,
        )
        points += self.geometry.center[None, :]
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        phi = np.linspace(0, 2 * np.pi, n_points + 1)[:-1].reshape(-1, 1)
        points = np.concat(
            [self.geometry.radius * np.cos(phi), self.geometry.radius * np.sin(phi)],
            axis=-1,
        )
        points += self.geometry.center[None, :]
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def normal(self, points):
        normal = points - self.geometry.center[None, :]
        return (normal / self.geometry.radius).reshape(-1, 2)

    def _get_volume(self):
        volume = 2 * np.pi * self.geometry.radius
        return volume
