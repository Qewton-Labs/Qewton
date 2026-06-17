import math

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.discrete.mesh_domain import MeshGeometry, Mesh
from qewton.config.variables import Variable
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu


class Circle(ContinuousGeometry[TensorType]):
    """Class for circles represented by center and radius.

    Args:
        variable (Variable): The variable associated with the circle, must be 2D.
        center (np.ndarray | list[float] | tuple[float, float]):
            The center of the circle.
        radius (float): The radius of the circle.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        center: list[float] | tuple[float, float],
        radius: float,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 2
        assert len(center) == 2
        self.center = backend.build_tensor(center)
        self.radius = radius
        super().__init__(variable=variable, backend=backend)

    def create_mesh(self, max_vertex_distance: float | None = None) -> MeshGeometry:
        if max_vertex_distance is None:
            power_n = 4
            n = 16
        else:
            power_n = int(
                math.ceil(math.log2(2 * math.pi * self.radius / max_vertex_distance))
            )
            n = int(max(4, 2**power_n))
        vertices = []
        triangles = []
        while_counter = 0
        while n >= 4:
            current_radius = ((power_n - while_counter) / power_n) ** 1.3 * self.radius
            for i in range(n):
                angle = 2 * math.pi * i / n
                vertices.append(
                    [
                        current_radius * math.cos(angle),
                        current_radius * math.sin(angle),
                    ]
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

        triangles = self.backend.build_tensor(triangles)
        vertices = self.backend.build_tensor(vertices)

        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=triangles),
            discretization_of=self,
            backend=self.backend,
        )

    def contains(self, points):
        norm = self.backend.linalg.norm(points - self.center, ord=2, axis=1).reshape(
            -1, 1
        )
        return norm <= self.radius

    def bounding_box(self):
        bounds = []
        for i in range(2):
            bounds.append(self.center[i] - self.radius)  # type: ignore
            bounds.append(self.center[i] + self.radius)  # type: ignore
        return self.backend.build_tensor(bounds)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        rand_radius = self.backend.random.uniform((n_points, 1), device=device)
        r = self.radius * self.backend.math.sqrt(rand_radius)
        phi = 2 * math.pi * self.backend.random.uniform((n_points, 1), device=device)
        x_coord = r * self.backend.math.cos(phi)
        y_coord = r * self.backend.math.sin(phi)
        points = self.backend.math.concatenate([x_coord, y_coord], axis=-1)
        points += self.backend.to(self.center[None, :], device)  # type: ignore
        return points

    def sample_grid(self, n_points: int, device: Device = cpu):
        grid = self._equidistant_points_in_circle(n_points, device=device)
        points = self.radius * grid
        points += self.backend.to(self.center[None, :], device)  # type: ignore
        return points

    def _equidistant_points_in_circle(self, n_points: int, device: Device = cpu):
        # use a sunflower seed arrangement:
        # https://demonstrations.wolfram.com/SunflowerSeedArrangements/
        gr = (math.sqrt(5) + 1) / 2.0  # golden ratio
        points = self.backend.math.arange(1, n_points + 1, device=device)
        points = self.backend.math.unsqueeze(points, axis=-1)
        phi = (2 * math.pi / gr) * points
        radius = self.backend.math.sqrt(points - 0.5) / math.sqrt(n_points + 0.5)
        x_coord = radius * self.backend.math.cos(phi)
        y_coord = radius * self.backend.math.sin(phi)
        return self.backend.math.concatenate([x_coord, y_coord], axis=-1)

    def _get_volume(self):
        volume = math.pi * self.radius**2
        return volume

    def create_boundary(self):
        return CircleBoundary(self)


class CircleBoundary(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry):
        assert isinstance(geometry, Circle)
        super().__init__(geometry)
        self.geometry: Circle = geometry  # type: ignore

    def contains(self, points):
        norm = self.backend.linalg.norm(
            points - self.geometry.center, ord=2, axis=1
        ).reshape(-1, 1)
        return self.backend.math.isclose(
            norm, self.backend.build_tensor(self.geometry.radius, dtype=norm.dtype)
        )

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        phi = (
            2
            * math.pi
            * self.backend.random.uniform((n_points, 1), device=device).reshape(-1, 1)
        )
        x_coord = self.geometry.radius * self.backend.math.cos(phi)
        y_coord = self.geometry.radius * self.backend.math.sin(phi)
        points = self.backend.math.concatenate([x_coord, y_coord], axis=-1)
        points += self.backend.to(self.geometry.center[None, :], device)
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        phi = self.backend.math.linspace(0, 2 * math.pi, n_points + 1, device=device)[
            :-1
        ].reshape(-1, 1)
        points = self.backend.math.concatenate(
            [
                self.geometry.radius * self.backend.math.cos(phi),
                self.geometry.radius * self.backend.math.sin(phi),
            ],
            axis=-1,
        )
        points += self.backend.to(self.geometry.center[None, :], device)
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points, device: Device = cpu):
        normal = points - self.backend.to(self.geometry.center[None, :], device)
        return (normal / self.geometry.radius).reshape(-1, 2)

    def _get_volume(self):
        volume = 2 * math.pi * self.geometry.radius
        return volume
