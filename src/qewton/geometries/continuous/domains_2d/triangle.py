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
from qewton.config.dtypes import Float32, Int32


class Triangle(ContinuousGeometry[TensorType]):
    """Class for triangles.

    Args:
        variable (Variable): The variable representing the underlying 2D space,
            must be 2D.
        origin (TensorType | list[float] | tuple[float, float]):
            The origin of the triangle (first corner).
        corner_1 (TensorType | list[float] | tuple[float, float]):
            The second corner of the triangle.
        corner_2 (TensorType | list[float] | tuple[float, float]):
            The third corner of the triangle.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        origin: TensorType | list[float] | tuple[float, float],
        corner_1: TensorType | list[float] | tuple[float, float],
        corner_2: TensorType | list[float] | tuple[float, float],
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 2
        super().__init__(variable=variable, backend=backend)
        self.origin = self._to_vector(origin)
        self.corner_1 = self._to_vector(corner_1)
        self.corner_2 = self._to_vector(corner_2)

    def _update_device(self, new_device):
        self.origin = self.backend.to(self.origin, device=new_device)
        self.corner_1 = self.backend.to(self.corner_1, device=new_device)
        self.corner_2 = self.backend.to(self.corner_2, device=new_device)

    def create_mesh(self, max_vertex_distance: float | None = None, device: Device = cpu):
        self._update_device(device)
        max_length = self.backend.math.max(
            [
                self.backend.linalg.norm(self.corner_1 - self.origin, order=2),
                self.backend.linalg.norm(self.corner_2 - self.origin, order=2),
                self.backend.linalg.norm(self.corner_1 - self.corner_2, order=2),
            ]
        )
        max_length = self.backend.cast_dtype(max_length, dtype=Float32)
        if max_vertex_distance is None:
            max_vertex_distance = max_length
        nx = int(self.backend.math.ceil(max_length / max_vertex_distance))

        vertices = []
        triangles = []
        for i in range(nx + 1):
            u = i / nx
            for j in range(nx + 1 - i):
                v = j / nx
                w = 1 - u - v
                vertices.append([w * self.origin + v * self.corner_1 + u * self.corner_2])

                if i > 0:
                    v_count = len(vertices) - 1
                    triangles.append(
                        [v_count, v_count - (nx + 1 - i), v_count - (nx + 1 - i) - 1]
                    )
                    if j > 0:
                        triangles.append(
                            [v_count, v_count - 1, v_count - (nx + 1 - i) - 1]
                        )
        vertices = self.backend.build_tensor(vertices, dtype=Float32).reshape(-1, 2)  # type: ignore
        triangles = self.backend.build_tensor(triangles, dtype=Int32).reshape(-1, 3)  # type: ignore
        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=triangles),
            discretization_of=self,
            backend=self.backend,
        )

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32)
        points = points.reshape(-1, 2)  # type: ignore
        origin = self.origin
        dir_1 = self.corner_1 - origin
        dir_2 = self.corner_2 - origin
        relative = points - origin
        bary_x, bary_y = self.solve_barycentric(relative, dir_1, dir_2)
        x_y_ok = self.backend.math.logical_and(bary_x >= 0.0, bary_y >= 0.0)
        inside = self.backend.math.logical_and(x_y_ok, bary_x + bary_y <= 1.0)
        return inside.reshape(-1, 1)

    def bounding_box(self):
        corners = self.backend.math.vstack([self.origin, self.corner_1, self.corner_2])
        mins = self.backend.math.min(corners, axis=0)
        maxs = self.backend.math.max(corners, axis=0)
        return self.backend.math.concatenate((mins[0:1], maxs[0:1], mins[1:2], maxs[1:2]))

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        # use standard method to sample uniformly in triangle
        self._update_device(device)
        r1 = self.backend.random.uniform((n_points,), device=device)
        r2 = self.backend.random.uniform((n_points,), device=device)
        sqrt_r1 = self.backend.math.sqrt(r1)
        b = sqrt_r1 * (1 - r2)
        c = sqrt_r1 * r2
        # point = a*A + b*B + c*C  -> convert to origin + b*(B-A) + c*(C-A)
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + (b[:, None] * dir_1) + (c[:, None] * dir_2)
        return points

    def sample_grid(self, n_points: int, device: Device = cpu):
        # build a barycentric grid and keep first n_points
        self._update_device(device)
        n_side = int(math.ceil(math.sqrt(2 * n_points)))
        u = self.backend.math.linspace(0, 1, n_side, device=device)
        v = self.backend.math.linspace(0, 1, n_side, device=device)
        uu, vv = self.backend.math.meshgrid(u, v)
        bary = self.backend.math.stack([uu.ravel(), vv.ravel()], axis=-1)
        mask = bary.sum(axis=1) <= 1
        bary = bary[mask]
        if len(bary) < n_points:
            # pad with random points if grid too coarse
            extra = n_points - len(bary)
            bary_rand = self.backend.random.uniform((extra, 2), device=device)
            sum_ge_1 = bary_rand.sum(axis=1) >= 1
            bary_rand[sum_ge_1] = 1 - bary_rand[sum_ge_1]
            bary = self.backend.math.vstack([bary, bary_rand])
        random_choice = self.backend.random.permutation(len(bary), device=device)
        bary = bary[random_choice[:n_points]]
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        points = self.origin + bary[:, :1] * dir_1 + bary[:, 1:] * dir_2
        return points

    def solve_barycentric(
        self, relative: TensorType, dir_1: TensorType, dir_2: TensorType
    ):
        # solve [dir_1 dir_2] [x; y] = relative.T for many points
        relative = self.backend.build_tensor(relative, dtype=Float32).reshape(-1, 2)
        det = dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0]
        if det == 0:
            raise ValueError("Degenerate triangle: direction vectors are collinear")
        bary_x = (dir_2[1] * relative[:, 0] - dir_2[0] * relative[:, 1]) / det
        bary_y = (-dir_1[1] * relative[:, 0] + dir_1[0] * relative[:, 1]) / det
        return bary_x, bary_y

    def _to_vector(self, vector):
        vec = self.backend.build_tensor(vector, dtype=Float32)
        if vec.shape != (2,):
            raise ValueError("Triangle corner must be a 2D vector")
        return vec

    def _get_volume(self):
        # area of triangle
        dir_1 = self.corner_1 - self.origin
        dir_2 = self.corner_2 - self.origin
        area = 0.5 * abs(dir_1[0] * dir_2[1] - dir_1[1] * dir_2[0])
        return area

    def create_boundary(self):
        return TriangleBoundary(self)


class TriangleBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Triangle):
        assert isinstance(geometry, Triangle)
        super().__init__(geometry)
        self.geometry: Triangle = geometry  # type: ignore

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 2)
        dir_1 = self.geometry.corner_1 - self.geometry.origin
        dir_2 = self.geometry.corner_2 - self.geometry.origin
        relative = points - self.geometry.origin
        bary_x, bary_y = self.geometry.solve_barycentric(relative, dir_1, dir_2)

        on_diagonal = self.backend.math.isclose(bary_x + bary_y, 1)
        on_bottom = self.backend.math.logical_and(
            self.backend.math.isclose(bary_x, 0), (bary_y >= 0) & (bary_y <= 1)
        )
        on_side = self.backend.math.logical_and(
            self.backend.math.isclose(bary_y, 0), (bary_x >= 0) & (bary_x <= 1)
        )
        on_edge = self.backend.math.logical_or(on_diagonal, on_bottom)
        on_edge = self.backend.math.logical_or(on_edge, on_side)
        return on_edge.reshape(-1, 1)

    def _get_volume(self):
        _, _, side_length = self._compute_edge_length()
        return side_length

    def _compute_edge_length(self):
        edges = [
            (self.geometry.origin, self.geometry.corner_1),
            (self.geometry.corner_1, self.geometry.corner_2),
            (self.geometry.corner_2, self.geometry.origin),
        ]
        lengths = self.backend.build_tensor(
            [self.backend.linalg.norm(b - a, order=2) for a, b in edges]
        )
        total = self.backend.math.sum(lengths)
        return edges, lengths, total

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        # sample uniformly along triangle edges proportional to edge length
        self.geometry._update_device(device)
        edges, lengths, total = self._compute_edge_length()
        probs = lengths / total
        choices = self.backend.random.choice(3, shape=(n_points,), p=probs, device=device)
        points = self.backend.math.zeros((n_points, 2), device=device)
        for i in range(3):
            idx = self.backend.math.where(choices == i)[0]
            if idx.size == 0:
                continue
            a, b = edges[i]
            t = self.backend.random.uniform((len(idx), 1), device=device)
            points[idx] = a + t * (b - a)
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        # distribute roughly equally across edges
        self.geometry._update_device(device)
        edges, lengths, total = self._compute_edge_length()
        counts = self.backend.math.floor(n_points * (lengths / total))
        counts = self.backend.cast_dtype(counts, Int32)
        # ensure total count
        rem = n_points - counts.sum()
        for i in range(rem):
            counts[i % 3] += 1
        points = []
        for (a, b), c in zip(edges, counts):
            if c == 0:
                continue
            if c <= 1:
                t = self.backend.to(self.backend.build_tensor([[0.5]]), device=device)
            else:
                t = self.backend.math.linspace(0, 1, c + 2, device=device)[1:-1][:, None]
            pts = a + t * (b - a)
            points.append(pts)
        if len(points) == 0:
            return self.backend.math.zeros((0, 2), device=device)
        points = self.backend.math.vstack(points)
        if len(points) > n_points:
            points = points[:n_points]
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points, device: Device = cpu):
        self.geometry._update_device(device)
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 2)
        edges = [
            (self.geometry.origin, self.geometry.corner_1),
            (self.geometry.corner_1, self.geometry.corner_2),
            (self.geometry.corner_2, self.geometry.origin),
        ]
        mids = [0.5 * (a + b) for a, b in edges]
        edge_vecs = [b - a for a, b in edges]
        # normals (perp) and ensure outward by checking centroid
        centroid = (
            self.geometry.origin + self.geometry.corner_1 + self.geometry.corner_2
        ) / 3.0
        normals = []
        for vec, mid in zip(edge_vecs, mids):
            n = self.backend.build_tensor([vec[1], -vec[0]])
            n = self.backend.to(n, device=device)
            # if the normal points towards centroid, flip it to point outward
            if self.backend.math.dot(n, centroid - mid) > 0:
                n = -n
            n = n / self.backend.linalg.norm(n, order=2)
            normals.append(n)
        normals = self.backend.math.stack(normals, axis=0)

        # compute barycentric coords to decide which edge each point belongs to
        dir_1 = self.geometry.corner_1 - self.geometry.origin
        dir_2 = self.geometry.corner_2 - self.geometry.origin
        bary_x, bary_y = self.geometry.solve_barycentric(
            points - self.geometry.origin, dir_1, dir_2
        )
        out = self.backend.math.zeros_like(points)
        # edge mapping: x==0 -> edge CA (index 2), y==0 -> edge AB (index 0),
        # x+y==1 -> edge BC (index 1)
        lower_bound = self.backend.build_tensor(0.0, dtype=Float32)
        upper_bound = self.backend.build_tensor(1.0, dtype=Float32)
        idx_x0 = self.backend.math.isclose(bary_x, lower_bound)
        idx_y0 = self.backend.math.isclose(bary_y, lower_bound)
        idx_sum1 = self.backend.math.isclose(bary_x + bary_y, upper_bound)
        # Add normals (corners get sum of normals) then normalize
        out[idx_x0] += normals[2]
        out[idx_y0] += normals[0]
        out[idx_sum1] += normals[1]
        # handle corner cases where multiple conditions true
        norms = self.backend.linalg.norm(out, order=2, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
