import math

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.config.variables import Variable
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Float32, Bool
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.mesh import Mesh


class Box(ContinuousGeometry[TensorType]):
    """Class for three-dimensional boxes.

    Args:
        variable (Variable): The variable associated with the box, must be 3D.
        origin (TensorType | list[float] | tuple[float, float, float]):
            The origin of the box (one corner).
        width (float): The width of the box.
        height (float): The height of the box.
        depth (float): The depth of the box.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        origin: TensorType | list[float] | tuple[float, float, float],
        width: float,
        height: float,
        depth: float,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 3
        super().__init__(variable=variable, backend=backend)
        self.origin = self._to_vector(origin)
        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)

    def contains(self, points):
        p_device = points.device if hasattr(points, "device") else cpu
        self.origin = self.backend.to(self.origin, device=p_device)
        points = self.backend.build_tensor(points, dtype=Float32)
        points = points.reshape(-1, 3)
        relative = points - self.origin
        inside = self.backend.math.ones((len(points), 1), dtype=bool, device=p_device)
        scale_list = [self.width, self.height, self.depth]
        for i in range(3):
            in_current = self.backend.math.logical_and(
                0 <= relative[:, i : i + 1], relative[:, i : i + 1] <= scale_list[i]
            )
            inside = self.backend.math.logical_and(in_current, inside)
        return inside.reshape(-1, 1)

    def bounding_box(self):
        bounds = []
        scale_list = [self.width, self.height, self.depth]
        for i in range(3):
            bounds.append(self.origin[i])
            bounds.append(self.origin[i] + scale_list[i])
        return self.backend.build_tensor(bounds)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        points = self.backend.random.uniform((n_points, 3), device=device)
        self.origin = self.backend.to(self.origin, device=device)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        points[:, 2] *= self.depth
        points += self.origin
        return points

    def sample_grid(self, n_points: int, device: Device = cpu):
        # Scale the number of points w.r.t. the shape of the box
        volume = self.width * self.height * self.depth
        n_scale = (n_points / volume) ** (1.0 / 3.0)
        n_x = max(1, int(self.width * n_scale))
        n_y = max(1, int(self.height * n_scale))
        n_z = max(1, int(self.depth * n_scale))

        x = self.backend.math.linspace(0, 1, n_x + 2, device=device)[1:-1]
        y = self.backend.math.linspace(0, 1, n_y + 2, device=device)[1:-1]
        z = self.backend.math.linspace(0, 1, n_z + 2, device=device)[1:-1]

        xx, yy, zz = self.backend.math.meshgrid(x, y, z, indexing="ij")
        grid = self.backend.math.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Scale and shift the grid
        self.origin = self.backend.to(self.origin, device=device)
        grid[:, 0] *= self.width
        grid[:, 1] *= self.height
        grid[:, 2] *= self.depth
        grid += self.origin

        # Ensure we have exactly n_points by adding random samples if needed
        if len(grid) < n_points:
            n_random = n_points - len(grid)
            random_points = self.backend.random.uniform((n_random, 3), device=device)
            random_points[:, 0] *= self.width
            random_points[:, 1] *= self.height
            random_points[:, 2] *= self.depth
            random_points += self.origin
            grid = self.backend.math.vstack([grid, random_points])

        return grid[:n_points]

    def _get_volume(self):
        return self.width * self.height * self.depth

    def _to_vector(self, vector):
        vector = self.backend.build_tensor(vector, dtype=Float32)
        return vector

    def create_boundary(self):
        return BoxBoundary(self)

    def create_mesh(
        self, max_vertex_distance: float | None = None, device: Device = cpu
    ) -> MeshGeometry:
        self.origin = self.backend.to(self.origin, device=device)

        # choose subdivision count
        if max_vertex_distance is None:
            nx = ny = nz = 1
        else:
            nx = max(1, int(math.ceil(self.width / max_vertex_distance)))
            ny = max(1, int(math.ceil(self.height / max_vertex_distance)))
            nz = max(1, int(math.ceil(self.depth / max_vertex_distance)))
        # unit-square vertices
        u = self.backend.math.linspace(0.0, 1.0, num=nx + 1)
        v = self.backend.math.linspace(0.0, 1.0, num=ny + 1)
        w = self.backend.math.linspace(0.0, 1.0, num=nz + 1)

        U, V, W = self.backend.math.meshgrid(u, v, w, indexing="ij")

        # affine map
        vertices = self.backend.math.reshape(
            self.origin
            + U[..., None] * self.backend.build_tensor([self.width, 0, 0], dtype=Float32)
            + V[..., None] * self.backend.build_tensor([0, self.height, 0], dtype=Float32)
            + W[..., None] * self.backend.build_tensor([0, 0, self.depth], dtype=Float32),
            (-1, 3),
        )

        # triangulation
        tetrahedra = []

        def idx(i, j, k):
            return (i * (ny + 1) + j) * (nz + 1) + k

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    a = idx(i, j, k)
                    b = idx(i + 1, j, k)
                    c = idx(i + 1, j + 1, k)
                    d = idx(i, j + 1, k)
                    a1 = idx(i, j, k + 1)
                    b1 = idx(i + 1, j, k + 1)
                    c1 = idx(i + 1, j + 1, k + 1)
                    d1 = idx(i, j + 1, k + 1)
                    tetrahedra.append([a1, b1, c1, b])
                    tetrahedra.append([a1, b, a, c1])
                    tetrahedra.append([a, b, c1, c])
                    tetrahedra.append([a, d1, c, d])
                    tetrahedra.append([a, d1, c1, a1])
                    tetrahedra.append([a, d1, c, c1])

        tetrahedra = self.backend.build_tensor(tetrahedra)

        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=vertices, cells=tetrahedra),
            discretization_of=self,
        )


class BoxBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Box):
        assert isinstance(geometry, Box)
        super().__init__(geometry)
        self.geometry: Box = geometry  # type: ignore

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32)
        points = points.reshape(-1, 3)
        p_device = points.device if hasattr(points, "device") else cpu
        self.geometry.origin = self.backend.to(self.geometry.origin, device=p_device)
        relative = points - self.geometry.origin
        scale_list = [self.geometry.width, self.geometry.height, self.geometry.depth]

        on_boundary = self.backend.math.zeros(
            (len(points), 1), dtype=self.backend.dtypes[Bool], device=p_device
        )
        inside = self.backend.math.ones(
            (len(points), 1), dtype=self.backend.dtypes[Bool], device=p_device
        )
        for i in range(3):
            # Check if on boundary at min or max in this direction
            lower_bound = self.backend.build_tensor(0.0, dtype=Float32)
            upper_bound = self.backend.build_tensor(scale_list[i], dtype=Float32)
            close_0 = self.backend.math.isclose(relative[:, i : i + 1], lower_bound)
            close_size = self.backend.math.isclose(relative[:, i : i + 1], upper_bound)
            on_boundary |= close_0 | close_size
            # Also check if points are inside the box (one dim on boundary, others inside)
            check_in = self.backend.math.logical_and(
                0 <= relative[:, i : i + 1], relative[:, i : i + 1] <= scale_list[i]
            )
            inside = self.backend.math.logical_and(inside, check_in)
        on_boundary = self.backend.math.logical_and(inside, on_boundary)
        return on_boundary.reshape(-1, 1)

    def _get_volume(self):
        w = self.geometry.width
        h = self.geometry.height
        d = self.geometry.depth
        area_xy = w * h
        area_xz = w * d
        area_yz = h * d
        return 2 * (area_xy + area_xz + area_yz)

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        w = self.geometry.width
        h = self.geometry.height
        d = self.geometry.depth

        # Surface areas of each pair of faces
        area_xy = w * h
        area_xz = w * d
        area_yz = h * d
        total_area = area_xy + area_xz + area_yz

        # Scale number of points to each direction's surface area
        area_list = [area_yz, area_xz, area_xy]
        scale_list = [w, h, d]

        points = self.backend.math.zeros((n_points, 3), device=device)
        current_n = 0

        # Sample on each pair of faces
        for i in range(3):
            if i < 2:
                n_scale = int(n_points * area_list[i] / total_area)
            else:  # last direction gets remaining points
                n_scale = n_points - current_n

            # Sample n_scale points on this pair of faces
            for k in range(2):
                n_half = n_scale // 2 if k == 0 else n_scale - n_scale // 2
                if current_n + n_half > n_points:
                    n_half = n_points - current_n

                pts = self.backend.random.uniform((n_half, 3))
                # Set current direction to boundary (0 or scale_list[i])
                pts[:, i] = k * scale_list[i]
                # Scale other two dimensions
                i_mod_1 = (i + 1) % 3
                i_mod_2 = (i + 2) % 3
                pts[:, i_mod_1] *= scale_list[i_mod_1]
                pts[:, i_mod_2] *= scale_list[i_mod_2]

                points[current_n : current_n + n_half] = pts
                current_n += n_half

        self.geometry.origin = self.backend.to(self.geometry.origin, device=device)
        points += self.geometry.origin
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        w = self.geometry.width
        h = self.geometry.height
        d = self.geometry.depth

        # Surface areas
        area_xy = w * h
        area_xz = w * d
        area_yz = h * d
        total_area = area_xy + area_xz + area_yz

        area_list = [area_yz, area_xz, area_xy]
        scale_list = [w, h, d]
        difference_list = [
            [h, d],
            [d, w],
            [w, h],
        ]

        points = self.backend.math.zeros((n_points, 3), device=device)
        current_n = 0

        # Sample on each direction's faces
        for i in range(3):
            if i < 2:
                n_scale = int(n_points * area_list[i] / total_area)
            else:  # last direction gets remaining points
                n_scale = n_points - current_n

            # Compute grid dimensions for this face
            n_1 = max(
                1,
                int(
                    math.sqrt(
                        n_scale / 2.0 * difference_list[i][0] / difference_list[i][1]
                    )
                ),
            )
            n_2 = max(
                1,
                int(
                    math.sqrt(
                        n_scale / 2.0 * difference_list[i][1] / difference_list[i][0]
                    )
                ),
            )

            grid_1 = self.backend.math.linspace(0, 1, n_1 + 1, device=device)
            grid_2 = self.backend.math.linspace(0, 1, n_2 + 1, device=device)

            # Sample on two faces (min and max in this direction)
            for k in range(2):
                if k == 0:
                    g1, g2 = self.backend.math.meshgrid(
                        grid_1[:-1], grid_2[:-1], indexing="ij"
                    )
                    grid = self.backend.math.stack([g1.ravel(), g2.ravel()], axis=1)
                    grid = self.backend.math.stack(
                        [self.backend.math.zeros(len(grid), device=device), grid], axis=0
                    )
                else:
                    g1, g2 = self.backend.math.meshgrid(
                        grid_1[1:], grid_2[1:], indexing="ij"
                    )
                    grid = self.backend.math.stack([g1.ravel(), g2.ravel()], axis=1)
                    # Add random points if needed to reach n_scale
                    n_prod = len(grid)
                    n_difference = n_scale - (2 * n_1 * n_2 + n_prod)
                    if n_difference > 0:
                        random_pts = self.backend.random.uniform(
                            (n_difference, 2), device=device
                        )
                        grid = self.backend.math.vstack([grid, random_pts])
                    grid = self.backend.math.stack(
                        [
                            scale_list[i]
                            * self.backend.math.ones(len(grid), device=device),
                            grid,
                        ],
                        axis=0,
                    )

                # Scale by the other two dimensions
                i_mod_1 = (i + 1) % 3
                i_mod_2 = (i + 2) % 3
                grid[:, 1] *= scale_list[i_mod_1]
                grid[:, 2] *= scale_list[i_mod_2]

                # Permute to correct order
                grid = grid[:, permute_list[i]]

                n_to_add = min(len(grid), n_points - current_n)
                points[current_n : current_n + n_to_add] = grid[:n_to_add]
                current_n += n_to_add

                if current_n >= n_points:
                    break

            if current_n >= n_points:
                break

        self.geometry.origin = self.backend.to(self.geometry.origin, device=device)
        points += self.geometry.origin

        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points, device: Device = cpu):
        points = self.backend.build_tensor(points, dtype=Float32)
        self.geometry.origin = self.backend.to(self.geometry.origin, device=device)
        points = points.reshape(-1, 3)
        relative = points - self.geometry.origin
        scale_list = [self.geometry.width, self.geometry.height, self.geometry.depth]

        normals = self.backend.math.zeros_like(relative)
        for i in range(3):
            lower_bound = self.backend.build_tensor(0.0, dtype=Float32)
            upper_bound = self.backend.build_tensor(scale_list[i], dtype=Float32)
            close_0 = self.backend.math.isclose(relative[:, i], lower_bound)
            close_size = self.backend.math.isclose(relative[:, i], upper_bound)
            normals[close_0, i] = -1.0
            normals[close_size, i] = 1.0

        # Normalize (scale normal vectors if they're in a corner)
        norms = self.backend.linalg.norm(normals, order=2, axis=1, keepdims=True)
        norms = self.backend.math.where(norms == 0, 1.0, norms)  # Avoid division by zero
        return normals / norms
