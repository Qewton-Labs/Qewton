import numpy as np

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.config.variables import Variable


class Box(ContinuousGeometry):
    """Class for three-dimensional boxes.

    Args:
        variable (Variable): The variable associated with the box, must be 3D.
        origin (np.ndarray | list[float] | tuple[float, float, float]):
            The origin of the box (one corner).
        width (float): The width of the box.
        height (float): The height of the box.
        depth (float): The depth of the box.

    """

    def __init__(
        self,
        variable: Variable,
        origin: np.ndarray | list[float] | tuple[float, float, float],
        width: float,
        height: float,
        depth: float,
    ):
        assert variable.dim == 3
        self.origin = self._to_vector(origin)
        self.width = float(width)
        self.height = float(height)
        self.depth = float(depth)
        super().__init__(variable=variable)

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        points = points.reshape(-1, 3)
        relative = points - self.origin
        inside = np.ones((len(points), 1), dtype=bool)
        scale_list = [self.width, self.height, self.depth]
        for i in range(3):
            in_current = np.logical_and(
                0 <= relative[:, i : i + 1], relative[:, i : i + 1] <= scale_list[i]
            )
            inside = np.logical_and(in_current, inside)
        return inside.reshape(-1, 1)

    def bounding_box(self):
        bounds = []
        for i in range(3):
            bounds.append(self.origin[i])
        scale_list = [self.width, self.height, self.depth]
        for i in range(3):
            bounds.append(self.origin[i] + scale_list[i])
        return np.array(bounds)

    def sample_random_uniform(self, n_points: int):
        points = np.random.rand(n_points, 3)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        points[:, 2] *= self.depth
        points += self.origin
        return points

    def sample_grid(self, n_points: int):
        # Scale the number of points w.r.t. the shape of the box
        volume = self.width * self.height * self.depth
        n_scale = (n_points / volume) ** (1.0 / 3.0)
        n_x = max(1, int(self.width * n_scale))
        n_y = max(1, int(self.height * n_scale))
        n_z = max(1, int(self.depth * n_scale))

        x = np.linspace(0, 1, n_x + 2)[1:-1]
        y = np.linspace(0, 1, n_y + 2)[1:-1]
        z = np.linspace(0, 1, n_z + 2)[1:-1]

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Scale and shift the grid
        grid[:, 0] *= self.width
        grid[:, 1] *= self.height
        grid[:, 2] *= self.depth
        grid += self.origin

        # Ensure we have exactly n_points by adding random samples if needed
        if len(grid) < n_points:
            n_random = n_points - len(grid)
            random_points = np.random.rand(n_random, 3)
            random_points[:, 0] *= self.width
            random_points[:, 1] *= self.height
            random_points[:, 2] *= self.depth
            random_points += self.origin
            grid = np.vstack([grid, random_points])

        return grid[:n_points]

    def _get_volume(self):
        return self.width * self.height * self.depth

    def _to_vector(self, vector):
        if isinstance(vector, (list, tuple)):
            vector = np.array(vector, dtype=float)
        else:
            vector = np.asarray(vector, dtype=float)
        return vector

    def create_boundary(self):
        return BoxBoundary(self)


class BoxBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Box):
        assert isinstance(geometry, Box)
        super().__init__(geometry)
        self.geometry: Box = geometry  # type: ignore

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        points = points.reshape(-1, 3)
        relative = points - self.geometry.origin
        scale_list = [self.geometry.width, self.geometry.height, self.geometry.depth]

        on_boundary = np.zeros((len(points), 1), dtype=bool)
        for i in range(3):
            # Check if on boundary at min or max in this direction
            close_0 = np.isclose(relative[:, i : i + 1], 0.0)
            close_size = np.isclose(relative[:, i : i + 1], scale_list[i])
            on_boundary |= close_0 | close_size

        # Also check if points are inside the box (one dim on boundary, others inside)
        in_current = np.logical_and.reduce(
            (
                0 <= relative[:, 0:1],
                relative[:, 0:1] <= scale_list[0],
                0 <= relative[:, 1:2],
                relative[:, 1:2] <= scale_list[1],
                0 <= relative[:, 2:3],
                relative[:, 2:3] <= scale_list[2],
            )
        )

        on_boundary &= in_current
        return on_boundary.reshape(-1, 1)

    def _get_volume(self):
        w = self.geometry.width
        h = self.geometry.height
        d = self.geometry.depth
        area_xy = w * h
        area_xz = w * d
        area_yz = h * d
        return 2 * (area_xy + area_xz + area_yz)

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
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

        points = np.zeros((n_points, 3))
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

                pts = np.random.rand(n_half, 3)
                # Set current direction to boundary (0 or scale_list[i])
                pts[:, i] = k * scale_list[i]
                # Scale other two dimensions
                i_mod_1 = (i + 1) % 3
                i_mod_2 = (i + 2) % 3
                pts[:, i_mod_1] *= scale_list[i_mod_1]
                pts[:, i_mod_2] *= scale_list[i_mod_2]

                points[current_n : current_n + n_half] = pts
                current_n += n_half

        points += self.geometry.origin
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
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
        permute_list = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
        difference_list = [
            [h, d],
            [d, w],
            [w, h],
        ]

        points = np.zeros((n_points, 3))
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
                    np.sqrt(n_scale / 2.0 * difference_list[i][0] / difference_list[i][1])
                ),
            )
            n_2 = max(
                1,
                int(
                    np.sqrt(n_scale / 2.0 * difference_list[i][1] / difference_list[i][0])
                ),
            )

            grid_1 = np.linspace(0, 1, n_1 + 1)
            grid_2 = np.linspace(0, 1, n_2 + 1)

            # Sample on two faces (min and max in this direction)
            for k in range(2):
                if k == 0:
                    g1, g2 = np.meshgrid(grid_1[:-1], grid_2[:-1], indexing="ij")
                    grid = np.stack([g1.ravel(), g2.ravel()], axis=1)
                    grid = np.column_stack([np.zeros(len(grid)), grid])
                else:
                    g1, g2 = np.meshgrid(grid_1[1:], grid_2[1:], indexing="ij")
                    grid = np.stack([g1.ravel(), g2.ravel()], axis=1)
                    # Add random points if needed to reach n_scale
                    n_prod = len(grid)
                    n_difference = n_scale - (2 * n_1 * n_2 + n_prod)
                    if n_difference > 0:
                        random_pts = np.random.rand(n_difference, 2)
                        grid = np.vstack([grid, random_pts])
                    grid = np.column_stack([scale_list[i] * np.ones(len(grid)), grid])

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

        points += self.geometry.origin
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def normal(self, points):
        points = np.asarray(points, dtype=float)
        points = points.reshape(-1, 3)
        relative = points - self.geometry.origin
        scale_list = [self.geometry.width, self.geometry.height, self.geometry.depth]

        normals = np.zeros_like(relative)
        for i in range(3):
            close_0 = np.isclose(relative[:, i], 0.0)
            close_size = np.isclose(relative[:, i], scale_list[i])
            normals[close_0, i] = -1.0
            normals[close_size, i] = 1.0

        # Normalize (scale normal vectors if they're in a corner)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        return normals / norms
