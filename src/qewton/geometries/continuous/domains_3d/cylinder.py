import numpy as np

from ..base import ContinuousGeometry, ContinuousBoundaryGeometry
from ....config.variables import Variable


class Cylinder(ContinuousGeometry):
    """Class for cylinders (NumPy-only, simplified).

    The cylinder is axis-aligned along the z-direction.

    Parameters
    ----------
    variable : Variable
        The variable representing the underlying 3D space.
    center : array_like
        The center of the cylinder, e.g. [0, 0, 0].
    radius : float
        The radius of the cylinder in the x-y plane.
    height : float
        The height of the cylinder along the z-axis.
    """

    def __init__(
        self,
        variable: Variable,
        center: np.ndarray | list[float] | tuple[float, float, float],
        radius: float,
        height: float,
    ):
        assert variable.dim == 3
        if isinstance(center, (list, tuple)):
            center = np.array(center, dtype=float)
        self.center: np.ndarray = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.height = float(height)
        super().__init__(variable=variable)

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        # Distance from axis in x-y plane
        radial_dist = np.linalg.norm(points[:, :2] - self.center[:2], axis=1).reshape(
            -1, 1
        )
        # Height check
        z_min = self.center[2] - self.height / 2.0
        z_max = self.center[2] + self.height / 2.0
        height_check = np.logical_and(points[:, 2:3] >= z_min, points[:, 2:3] <= z_max)
        return np.logical_and(radial_dist <= self.radius, height_check)

    def bounding_box(self):
        bounds = []
        for i in range(2):
            bounds.append(self.center[i] - self.radius)
            bounds.append(self.center[i] + self.radius)
        z_min = self.center[2] - self.height / 2.0
        z_max = self.center[2] + self.height / 2.0
        bounds.append(z_min)
        bounds.append(z_max)
        return np.array(bounds)

    def sample_random_uniform(self, n_points: int):
        # Sample radius with proper weighting (r^2 in 2D)
        r = self.radius * np.sqrt(np.random.rand(n_points, 1))
        # Sample angle
        phi = 2 * np.pi * np.random.rand(n_points, 1)
        # Sample height uniformly
        z = np.random.uniform(
            self.center[2] - self.height / 2.0,
            self.center[2] + self.height / 2.0,
            (n_points, 1),
        )
        # Convert to Cartesian coordinates
        x = r * np.cos(phi) + self.center[0]
        y = r * np.sin(phi) + self.center[1]
        points = np.concatenate([x, y, z], axis=1)
        return points

    def sample_grid(self, n_points: int):
        # Use a single radial grid and replicate it evenly along the cylinder height.
        z_min = self.center[2] - self.height / 2.0
        z_max = self.center[2] + self.height / 2.0

        n_xy = max(1, int(np.ceil(np.sqrt(n_points))))
        n_layers = max(1, int(np.ceil(n_points / n_xy)))

        base_circle = self._sample_grid_circle(n_xy, z_min)
        xy = base_circle[:, :2]

        z_vals = np.linspace(z_min, z_max, n_layers)
        points = []
        for z in z_vals:
            layer = np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=1)
            points.append(layer)

        points = np.vstack(points)
        return points[:n_points]

    def _sample_grid_circle(self, n_points: int, z_val: float):
        """Sample points on a circle using sunflower arrangement."""
        grid = self._equidistant_points_in_circle(n_points)
        xy = self.radius * grid
        z = np.full((n_points, 1), z_val)
        xy += self.center[:2][None, :]
        points = np.concatenate([xy, z], axis=1)
        return points

    def _equidistant_points_in_circle(self, n_points: int):
        """Sunflower seed arrangement in a circle."""
        gr = (np.sqrt(5) + 1) / 2.0  # golden ratio
        points = np.arange(1, n_points + 1)
        phi = (2 * np.pi / gr) * points
        radius = np.sqrt(points - 0.5) / np.sqrt(n_points + 0.5)
        points = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
        return points

    def _get_volume(self):
        volume = np.pi * self.radius**2 * self.height
        return volume

    def create_boundary(self):
        return CylinderBoundary(self)


class CylinderBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Cylinder):
        assert isinstance(geometry, Cylinder)
        super().__init__(geometry)
        self.geometry: Cylinder = geometry  # type: ignore

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        radial_dist = np.linalg.norm(
            points[:, :2] - self.geometry.center[:2], axis=1
        ).reshape(-1, 1)
        z_vals = points[:, 2:3]

        z_min = self.geometry.center[2] - self.geometry.height / 2.0
        z_max = self.geometry.center[2] + self.geometry.height / 2.0

        # On side surface: distance is close to radius and z is within range
        on_side = np.logical_and(
            np.isclose(radial_dist, self.geometry.radius),
            np.logical_and(z_vals >= z_min, z_vals <= z_max),
        )

        # On bottom cap: z is close to z_min and radial distance <= radius
        on_bottom = np.logical_and(
            np.isclose(z_vals, z_min), radial_dist <= self.geometry.radius
        )

        # On top cap: z is close to z_max and radial distance <= radius
        on_top = np.logical_and(
            np.isclose(z_vals, z_max), radial_dist <= self.geometry.radius
        )

        return np.logical_or(np.logical_or(on_side, on_bottom), on_top)

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        """Sample uniformly from all surfaces (side + bottom + top)."""
        # Surface areas
        side_area = 2 * np.pi * self.geometry.radius * self.geometry.height
        cap_area = np.pi * self.geometry.radius**2
        total_area = side_area + 2 * cap_area

        n_side = max(1, int(n_points * side_area / total_area))
        n_cap = max(1, int((n_points - n_side) / 2))

        points_list = []

        # Sample side surface
        if n_side > 0:
            side_points = self._sample_random_side(n_side)
            points_list.append(side_points)

        # Sample bottom cap
        if n_cap > 0:
            bottom_points = self._sample_random_circle(
                n_cap, self.geometry.center[2] - self.geometry.height / 2.0
            )
            points_list.append(bottom_points)

        # Sample top cap
        remaining = n_points - n_side - n_cap
        if remaining > 0:
            top_points = self._sample_random_circle(
                remaining, self.geometry.center[2] + self.geometry.height / 2.0
            )
            points_list.append(top_points)

        points = np.vstack(points_list)
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        """Sample on a grid from all surfaces."""
        # Surface areas
        side_area = 2 * np.pi * self.geometry.radius * self.geometry.height
        cap_area = np.pi * self.geometry.radius**2
        total_area = side_area + 2 * cap_area

        n_side = max(1, int(n_points * side_area / total_area))
        n_cap = max(1, int((n_points - n_side) / 2))

        points_list = []

        # Sample side surface
        if n_side > 0:
            side_points = self._sample_grid_side(n_side)
            points_list.append(side_points)

        # Sample bottom cap
        if n_cap > 0:
            bottom_points = self._sample_grid_circle(
                n_cap, self.geometry.center[2] - self.geometry.height / 2.0
            )
            points_list.append(bottom_points)

        # Sample top cap
        remaining = n_points - n_side - n_cap
        if remaining > 0:
            top_points = self._sample_grid_circle(
                remaining, self.geometry.center[2] + self.geometry.height / 2.0
            )
            points_list.append(top_points)

        points = np.vstack(points_list)
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def _sample_random_circle(self, n_points: int, z_val: float):
        """Sample random points on a circle."""
        r = self.geometry.radius * np.sqrt(np.random.rand(n_points, 1))
        phi = 2 * np.pi * np.random.rand(n_points, 1)
        x = r * np.cos(phi) + self.geometry.center[0]
        y = r * np.sin(phi) + self.geometry.center[1]
        z = np.full((n_points, 1), z_val)
        return np.concatenate([x, y, z], axis=1)

    def _sample_grid_circle(self, n_points: int, z_val: float):
        """Sample grid points on a circle using sunflower arrangement."""
        grid = self._equidistant_points_in_circle(n_points)
        xy = self.geometry.radius * grid
        z = np.full((n_points, 1), z_val)
        xy += self.geometry.center[:2][None, :]
        points = np.concatenate([xy, z], axis=1)
        return points

    def _sample_random_side(self, n_points: int):
        """Sample random points on the cylindrical side surface."""
        phi = 2 * np.pi * np.random.rand(n_points, 1)
        z = np.random.uniform(
            self.geometry.center[2] - self.geometry.height / 2.0,
            self.geometry.center[2] + self.geometry.height / 2.0,
            (n_points, 1),
        )
        x = self.geometry.radius * np.cos(phi) + self.geometry.center[0]
        y = self.geometry.radius * np.sin(phi) + self.geometry.center[1]
        return np.concatenate([x, y, z], axis=1)

    def _sample_grid_side(self, n_points: int):
        """Sample grid points on the cylindrical side surface."""
        n_phi = max(1, int(np.sqrt(n_points)))
        n_z = max(1, (n_points + n_phi - 1) // n_phi)

        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        z_vals = np.linspace(
            self.geometry.center[2] - self.geometry.height / 2.0,
            self.geometry.center[2] + self.geometry.height / 2.0,
            n_z,
        )

        phi_grid, z_grid = np.meshgrid(phi, z_vals)
        phi_flat = phi_grid.flatten()[:n_points]
        z_flat = z_grid.flatten()[:n_points]

        x = self.geometry.radius * np.cos(phi_flat) + self.geometry.center[0]
        y = self.geometry.radius * np.sin(phi_flat) + self.geometry.center[1]
        z = z_flat

        points = np.column_stack([x, y, z])
        return points

    def _equidistant_points_in_circle(self, n_points: int):
        """Sunflower seed arrangement in a circle."""
        gr = (np.sqrt(5) + 1) / 2.0  # golden ratio
        points = np.arange(1, n_points + 1)
        phi = (2 * np.pi / gr) * points
        radius = np.sqrt(points - 0.5) / np.sqrt(n_points + 0.5)
        points = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
        return points

    def normal(self, points):
        """Compute outward normal vectors at boundary points."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        normals = np.zeros_like(points)

        z_min = self.geometry.center[2] - self.geometry.height / 2.0
        z_max = self.geometry.center[2] + self.geometry.height / 2.0

        # On bottom cap
        bottom_mask = np.isclose(points[:, 2], z_min)
        normals[bottom_mask] = np.array([0, 0, -1])

        # On top cap
        top_mask = np.isclose(points[:, 2], z_max)
        normals[top_mask] = np.array([0, 0, 1])

        # On side surface: radial direction
        side_mask = ~(bottom_mask | top_mask)
        radial = points[side_mask, :2] - self.geometry.center[:2]
        radial_dist = np.linalg.norm(radial, axis=1, keepdims=True)
        normals[side_mask, :2] = radial / radial_dist

        return normals

    def _get_volume(self):
        """Surface area of the cylinder."""
        side_area = 2 * np.pi * self.geometry.radius * self.geometry.height
        cap_area = 2 * np.pi * self.geometry.radius**2
        return side_area + cap_area
