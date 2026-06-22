import math
from typing import Any, Callable

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.geometries.discrete.mesh_domain import MeshGeometry, Mesh
from qewton.config.variables import Variable
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Float32


class Parallelogram(ContinuousGeometry[TensorType]):
    """Class for parallelograms in 2D.

    Args:
        variable (Variable): The variable representing the underlying space.
        origin (TensorType | list[float] | tuple[float, float]):
            One corner of the parallelogram.
        corner_1 (TensorType | list[float] | tuple[float, float]):
            A second corner adjacent to `origin`.
        corner_2 (TensorType | list[float] | tuple[float, float]):
            A third corner adjacent to `origin`.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        origin: Any | list[float] | tuple[float, float],
        corner_1: Any | list[float] | tuple[float, float],
        corner_2: Any | list[float] | tuple[float, float],
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 2
        super().__init__(variable=variable, backend=backend)
        self.origin = self._to_vector(origin)
        self.corner_1 = self._to_vector(corner_1)
        self.corner_2 = self._to_vector(corner_2)

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 2)
        origin = self.origin
        dir_1 = self.corner_1 - origin
        dir_2 = self.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.solve_barycentric(relative, dir_1, dir_2)

        c1 = self.backend.math.greater_equal(bary_x, 0.0)
        c2 = self.backend.math.less_equal(bary_x, 1.0)
        c3 = self.backend.math.greater_equal(bary_y, 0.0)
        c4 = self.backend.math.less_equal(bary_y, 1.0)

        inside = self.backend.math.logical_and(
            self.backend.math.logical_and(c1, c2), self.backend.math.logical_and(c3, c4)
        )
        return self.backend.math.reshape(inside, (-1, 1))

    def bounding_box(self):
        origin = self.origin
        corner_1 = self.corner_1
        corner_2 = self.corner_2
        corner_3 = corner_1 + corner_2 - origin
        corners = self.backend.math.vstack([origin, corner_1, corner_2, corner_3])
        mins = self.backend.math.min(corners, axis=0)
        maxs = self.backend.math.max(corners, axis=0)
        return self.backend.math.concatenate(
            [mins[0:1], maxs[0:1], mins[1:2], maxs[1:2]], axis=0
        )

    def create_mesh(self, max_vertex_distance: float | None = None) -> MeshGeometry:
        # edge vectors
        e1 = self.corner_1 - self.origin
        e2 = self.corner_2 - self.origin

        # choose subdivision count
        if max_vertex_distance is None:
            nx = ny = 1
        else:
            nx = max(
                1,
                int(
                    math.ceil(
                        float(self.backend.linalg.norm(e1, order=2)) / max_vertex_distance
                    )
                ),
            )
            ny = max(
                1,
                int(
                    math.ceil(
                        float(self.backend.linalg.norm(e2, order=2)) / max_vertex_distance
                    )
                ),
            )
        # unit-square vertices
        u = self.backend.math.linspace(0.0, 1.0, num=nx + 1)
        v = self.backend.math.linspace(0.0, 1.0, num=ny + 1)

        U, V = self.backend.math.meshgrid(u, v, indexing="ij")

        # affine map
        vertices = self.backend.math.reshape(
            self.origin + U[..., None] * e1 + V[..., None] * e2, (-1, 2)
        )

        # triangulation
        triangles = []

        def idx(i, j):
            return i * (ny + 1) + j

        for i in range(nx):
            for j in range(ny):
                a = idx(i, j)
                b = idx(i + 1, j)
                c = idx(i + 1, j + 1)
                d = idx(i, j + 1)

                triangles.append([a, b, c])
                triangles.append([a, c, d])

        triangles = self.backend.build_tensor(triangles)

        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=triangles),
            discretization_of=self,
        )

    def move_to_device(self, new_device: Device):
        self.origin = self.backend.to(self.origin, device=new_device)
        self.corner_1 = self.backend.to(self.corner_1, device=new_device)
        self.corner_2 = self.backend.to(self.corner_2, device=new_device)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        self.move_to_device(device)
        bary_coords = self.backend.random.uniform((n_points, 2), device=device)
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + bary_coords[:, :1] * dir_1 + bary_coords[:, 1:] * dir_2
        return points

    def sample_grid(self, n_points: int, device: Device = cpu):
        self.move_to_device(device)
        n_side = int(math.ceil(math.sqrt(n_points)))
        u = self.backend.math.linspace(0.0, 1.0, num=n_side, device=device)
        v = self.backend.math.linspace(0.0, 1.0, num=n_side, device=device)
        uu, vv = self.backend.math.meshgrid(u, v)
        bary_coords = self.backend.math.concatenate(
            [
                self.backend.math.reshape(uu, (-1, 1)),
                self.backend.math.reshape(vv, (-1, 1)),
            ],
            axis=-1,
        )

        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + bary_coords[:, :1] * dir_1 + bary_coords[:, 1:] * dir_2
        return points[:n_points]

    def solve_barycentric(
        self,
        relative: TensorType,
        dir_1: TensorType,
        dir_2: TensorType,
    ):
        det = dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0]
        if det == 0:
            raise ValueError("Parallelogram corners must not be collinear.")
        bary_x = (dir_2[1] * relative[:, 0] - dir_2[0] * relative[:, 1]) / det
        bary_y = (-dir_1[1] * relative[:, 0] + dir_1[0] * relative[:, 1]) / det
        return bary_x, bary_y

    def _to_vector(self, vector):
        array = self.backend.build_tensor(vector, dtype=Float32).reshape(  # type: ignore
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


class ParallelogramBoundary(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry: Parallelogram):
        super().__init__(geometry)
        self.geometry: Parallelogram = geometry  # type: ignore

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 2)
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.geometry.solve_barycentric(relative, dir_1, dir_2)
        x_close = self._bary_coords_close_to_0_or_1(
            bary_x, bary_y, self.backend.math.isclose
        )
        y_close = self._bary_coords_close_to_0_or_1(
            bary_y, bary_x, self.backend.math.isclose
        )
        return self.backend.math.reshape(
            self.backend.math.logical_or(x_close, y_close), (-1, 1)
        )

    def _get_volume(self):
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        side_length1 = self.backend.linalg.norm(dir_1, order=2)
        side_length2 = self.backend.linalg.norm(dir_2, order=2)
        return 2 * (side_length1 + side_length2)

    def _bary_coords_close_to_0_or_1(
        self, bary_coord1: TensorType, bary_coord2: TensorType, isclose_func: Callable
    ):
        between_0_1 = self.backend.math.logical_and(
            self.backend.math.greater_equal(bary_coord2, 0.0),
            self.backend.math.less_equal(bary_coord2, 1.0),
        )
        close_to_0 = isclose_func(bary_coord1, self.backend.build_tensor(0.0))
        close_to_1 = isclose_func(bary_coord1, self.backend.build_tensor(1.0))
        return self.backend.math.logical_and(
            self.backend.math.logical_or(close_to_0, close_to_1), between_0_1
        )

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        self.geometry.move_to_device(device)
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        corner_3 = self.geometry.corner_1 + self.geometry.corner_2 - origin
        side_lengths = self.backend.math.stack(
            [
                self.backend.linalg.norm(dir_1, order=2),
                self.backend.linalg.norm(dir_2, order=2),
                self.backend.linalg.norm(dir_1, order=2),
                self.backend.linalg.norm(dir_2, order=2),
            ],
        )
        total_length = self.backend.math.sum(side_lengths)
        positions = self.backend.random.uniform((n_points,), device=device) * total_length
        breaks = self.backend.math.cumsum(side_lengths)

        # Segment 0: origin to corner_1
        mask0 = self.backend.math.less(positions, breaks[0])
        t0 = positions / side_lengths[0]
        p0 = origin + self.backend.math.unsqueeze(t0, -1) * dir_1

        # Segment 1: corner_1 to corner_3
        mask1 = self.backend.math.logical_and(
            self.backend.math.greater_equal(positions, breaks[0]),
            self.backend.math.less(positions, breaks[1]),
        )
        t1 = (positions - breaks[0]) / side_lengths[1]
        p1 = self.geometry.corner_1 + self.backend.math.unsqueeze(t1, -1) * dir_2

        # Segment 2: corner_3 to corner_2
        mask2 = self.backend.math.logical_and(
            self.backend.math.greater_equal(positions, breaks[1]),
            self.backend.math.less(positions, breaks[2]),
        )
        t2 = (positions - breaks[1]) / side_lengths[2]
        p2 = corner_3 - self.backend.math.unsqueeze(t2, -1) * dir_1

        # Segment 3: corner_2 to origin
        mask3 = self.backend.math.greater_equal(positions, breaks[2])
        t3 = (positions - breaks[2]) / side_lengths[3]
        p3 = self.geometry.corner_2 - self.backend.math.unsqueeze(t3, -1) * dir_2

        points = self.backend.math.zeros((n_points, 2), device=device)
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask0, -1), p0, points
        )
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask1, -1), p1, points
        )
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask2, -1), p2, points
        )
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask3, -1), p3, points
        )

        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        self.geometry.move_to_device(device)
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        corner_3 = self.geometry.corner_1 + self.geometry.corner_2 - origin
        side_lengths = self.backend.math.stack(
            [
                self.backend.linalg.norm(dir_1, order=2),
                self.backend.linalg.norm(dir_2, order=2),
                self.backend.linalg.norm(dir_1, order=2),
                self.backend.linalg.norm(dir_2, order=2),
            ],
        )
        total_length = self.backend.math.sum(side_lengths)
        positions = self.backend.math.linspace(
            0.0, total_length, num=n_points + 1, device=device
        )[:-1]
        breaks = self.backend.math.cumsum(side_lengths)

        # Segment 0
        mask0 = self.backend.math.less(positions, breaks[0])
        t0 = positions / side_lengths[0]
        p0 = origin + self.backend.math.unsqueeze(t0, -1) * dir_1

        # Segment 1
        mask1 = self.backend.math.logical_and(
            self.backend.math.greater_equal(positions, breaks[0]),
            self.backend.math.less(positions, breaks[1]),
        )
        t1 = (positions - breaks[0]) / side_lengths[1]
        p1 = self.geometry.corner_1 + self.backend.math.unsqueeze(t1, -1) * dir_2

        # Segment 2
        mask2 = self.backend.math.logical_and(
            self.backend.math.greater_equal(positions, breaks[1]),
            self.backend.math.less(positions, breaks[2]),
        )
        t2 = (positions - breaks[1]) / side_lengths[2]
        p2 = corner_3 - self.backend.math.unsqueeze(t2, -1) * dir_1

        # Segment 3
        mask3 = self.backend.math.greater_equal(positions, breaks[2])
        t3 = (positions - breaks[2]) / side_lengths[3]
        p3 = self.geometry.corner_2 - self.backend.math.unsqueeze(t3, -1) * dir_2

        points = self.backend.math.zeros((n_points, 2), device=device)
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask0, -1), p0, points
        )
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask1, -1), p1, points
        )
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask2, -1), p2, points
        )
        points = self.backend.math.where(
            self.backend.math.unsqueeze(mask3, -1), p3, points
        )

        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points, device: Device = cpu):
        self.geometry.move_to_device(device)
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 2)
        origin = self.geometry.origin
        dir_1 = self.geometry.corner_1 - origin
        dir_2 = self.geometry.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.geometry.solve_barycentric(relative, dir_1, dir_2)

        normal_dir_1 = self._get_normal_direction(dir_1, device=device)
        normal_dir_2 = -self._get_normal_direction(dir_2, device=device)

        normals = self.backend.math.zeros_like(points, device=device)
        for bary, n_dir in zip([bary_x, bary_y], [normal_dir_2, normal_dir_1]):
            for i in range(2):
                at_boundary = self.backend.math.isclose(
                    bary, self.backend.build_tensor(i)
                )
                normals = self.backend.math.where(
                    at_boundary[:, None], normals + (2 * i - 1) * n_dir, normals
                )

        norms = self.backend.linalg.norm(normals, order=2, axis=1, keepdims=True)
        return normals / norms

    def _get_normal_direction(self, direction: TensorType, device: Device = cpu):
        normal = self.backend.build_tensor([-direction[1], direction[0]], dtype=Float32)
        normal = self.backend.to(normal, device=device)
        return normal / self.backend.linalg.norm(normal, order=2)
