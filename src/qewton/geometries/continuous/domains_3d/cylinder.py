import math

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.config.variables import Variable
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Float32
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.discrete.mesh_geometry import MeshGeometry, Mesh


class Cylinder(ContinuousGeometry[TensorType]):
    """Class for cylinders.

    The cylinder is axis-aligned along the upwards direction.

    Args:
        variable (Variable): The variable associated with the cylinder, must be 3D.
        center (TensorType) | list[float] | tuple[float, float, float]):
            The center of the cylinder, e.g. [0, 0, 0].
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        center: TensorType | list[float] | tuple[float, float, float],
        radius: float,
        height: float,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 3
        super().__init__(variable=variable, backend=backend)
        self.center: TensorType = self.backend.build_tensor(center, dtype=Float32)
        self.radius = self.backend.build_tensor(radius, dtype=Float32)
        self.height = self.backend.build_tensor(height, dtype=Float32)

    def _move_params(self, device: Device):
        self.center = self.backend.to(self.center, device=device)
        self.radius = self.backend.to(self.radius, device=device)
        self.height = self.backend.to(self.height, device=device)

    def contains(self, points):
        p_device = points.device if hasattr(points, "device") else cpu
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 3)
        self._move_params(p_device)
        # Distance from axis in x-y plane
        radial_dist = self.backend.linalg.norm(
            points[:, :2] - self.center[:2], axis=1, order=2
        ).reshape(-1, 1)
        # Height check
        z_min = self.center[2] - self.height / 2.0
        z_max = self.center[2] + self.height / 2.0
        height_check = self.backend.math.logical_and(
            points[:, 2:3] >= z_min, points[:, 2:3] <= z_max
        )
        return self.backend.math.logical_and(radial_dist <= self.radius, height_check)

    def bounding_box(self):
        bounds = []
        for i in range(2):
            bounds.append(self.center[i] - self.radius)
            bounds.append(self.center[i] + self.radius)
        z_min = self.center[2] - self.height / 2.0
        z_max = self.center[2] + self.height / 2.0
        bounds.append(z_min)
        bounds.append(z_max)
        return self.backend.build_tensor(bounds)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        self._move_params(device)
        # Sample radius with proper weighting (r^2 in 2D)
        r = self.radius * self.backend.math.sqrt(
            self.backend.random.uniform((n_points, 1), device=device)
        )
        # Sample angle
        phi = 2 * math.pi * self.backend.random.uniform((n_points, 1), device=device)
        # Sample height uniformly
        z = self.backend.random.uniform(
            shape=(n_points, 1),
            low=self.center[2] - self.height / 2.0,
            high=self.center[2] + self.height / 2.0,
            device=device,
        )
        # Convert to Cartesian coordinates
        x = r * self.backend.math.cos(phi) + self.center[0]
        y = r * self.backend.math.sin(phi) + self.center[1]
        points = self.backend.math.concatenate([x, y, z], axis=1)
        return points

    def sample_grid(self, n_points: int, device: Device = cpu):
        self._move_params(device)
        # Use a single radial grid and replicate it evenly along the cylinder height.
        z_min = self.center[2] - self.height / 2.0
        z_max = self.center[2] + self.height / 2.0

        n_xy = max(1, int(math.ceil(math.sqrt(n_points))))
        n_layers = max(1, int(math.ceil(n_points / n_xy)))

        base_circle = self._sample_grid_circle(n_xy, z_min, device=device)
        xy = base_circle[:, :2]

        z_vals = self.backend.math.linspace(z_min, z_max, n_layers, device=device)
        points = []
        for z in z_vals:
            layer = self.backend.math.concatenate(
                [xy, self.backend.math.full((xy.shape[0], 1), z, device=device)], axis=1
            )
            points.append(layer)

        points = self.backend.math.concatenate(points, axis=0)
        return points[:n_points]

    def _sample_grid_circle(self, n_points: int, z_val: float, device: Device):
        """Sample points on a circle using sunflower arrangement."""
        grid = self._equidistant_points_in_circle(n_points, device=device)
        xy = self.radius * grid
        z = self.backend.math.full((n_points, 1), z_val, device=device)
        xy += self.center[:2][None, :]
        points = self.backend.math.concatenate([xy, z], axis=1)
        return points

    def _equidistant_points_in_circle(self, n_points: int, device: Device):
        """Sunflower seed arrangement in a circle."""
        gr = (math.sqrt(5) + 1) / 2.0  # golden ratio
        points = self.backend.math.arange(1, n_points + 1, device=device)
        points = self.backend.math.unsqueeze(points, -1)
        phi = (2 * math.pi / gr) * points
        radius = self.backend.math.sqrt(points - 0.5) / math.sqrt(n_points + 0.5)
        points = self.backend.math.concatenate(
            [radius * self.backend.math.cos(phi), radius * self.backend.math.sin(phi)],
            axis=-1,
        )
        return points

    def _get_volume(self):
        volume = math.pi * self.radius**2 * self.height
        return volume

    def create_boundary(self):
        return CylinderBoundary(self)

    def create_mesh(
        self, max_vertex_distance: float | None = None, device: Device = cpu
    ) -> MeshGeometry:
        vertices, triangles = Circle.triangulate_circle(
            max_vertex_distance, radius=self.radius, backend=self.backend
        )
        zeros = self.backend.math.zeros((len(vertices), 1), device=device)
        vertices = self.backend.math.concatenate([vertices, zeros], axis=1)
        all_vertices = []
        tetrahedra = []
        nz = math.ceil(self.height / max_vertex_distance) + 1
        for k in range(nz):
            vertices_copy = self.backend.math.copy(vertices)
            # triangles_copy = self.backend.math.copy(triangles)
            vertices_copy[:, 2] = self.height * (k / (nz - 1))
            all_vertices.append(vertices_copy)
            if k == nz - 1:
                continue
            # Build tetraheder
            v_count = len(vertices)
            for tri in triangles:
                a, b, c = tri + v_count * k
                a1, b1, c1 = tri + v_count * (k + 1)
                tetrahedra.append([a1, b1, c1, b])
                tetrahedra.append([a1, b, a, c1])
                tetrahedra.append([a, b, c1, c])
        all_vertices = self.backend.math.concatenate(all_vertices, axis=0)
        tetrahedra = self.backend.build_tensor(tetrahedra)

        return MeshGeometry(
            variable=self.variable,
            mesh=Mesh(vertices=all_vertices, cells=tetrahedra),
            discretization_of=self,
        )


class CylinderBoundary(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry: Cylinder):
        assert isinstance(geometry, Cylinder)
        super().__init__(geometry)
        self.geometry: Cylinder = geometry  # type: ignore

    def contains(self, points):
        p_device = points.device if hasattr(points, "device") else cpu
        self.geometry._move_params(p_device)
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 3)
        radial_dist = self.backend.linalg.norm(
            points[:, :2] - self.geometry.center[:2], axis=1, order=2
        ).reshape(-1, 1)
        z_vals = points[:, 2:3]

        z_min = self.geometry.center[2] - self.geometry.height / 2.0
        z_max = self.geometry.center[2] + self.geometry.height / 2.0

        # On side surface: distance is close to radius and z is within range
        on_side = self.backend.math.logical_and(
            self.backend.math.isclose(radial_dist, self.geometry.radius),
            self.backend.math.logical_and(z_vals >= z_min, z_vals <= z_max),
        )

        # On bottom cap: z is close to z_min and radial distance <= radius
        on_bottom = self.backend.math.logical_and(
            self.backend.math.isclose(z_vals, z_min), radial_dist <= self.geometry.radius
        )

        # On top cap: z is close to z_max and radial distance <= radius
        on_top = self.backend.math.logical_and(
            self.backend.math.isclose(z_vals, z_max), radial_dist <= self.geometry.radius
        )

        return self.backend.math.logical_or(
            self.backend.math.logical_or(on_side, on_bottom), on_top
        )

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        """Sample uniformly from all surfaces (side + bottom + top)."""
        self.geometry._move_params(device)
        # Surface areas
        side_area = 2 * math.pi * self.geometry.radius * self.geometry.height
        cap_area = math.pi * self.geometry.radius**2
        total_area = side_area + 2 * cap_area

        n_side = max(1, int(n_points * side_area / total_area))
        n_cap = max(1, int((n_points - n_side) / 2))

        points_list = []

        # Sample side surface
        if n_side > 0:
            side_points = self._sample_random_side(n_side, device=device)
            points_list.append(side_points)

        # Sample bottom cap
        if n_cap > 0:
            bottom_points = self._sample_random_circle(
                n_cap, self.geometry.center[2] - self.geometry.height / 2.0, device=device
            )
            points_list.append(bottom_points)

        # Sample top cap
        remaining = n_points - n_side - n_cap
        if remaining > 0:
            top_points = self._sample_random_circle(
                remaining,
                self.geometry.center[2] + self.geometry.height / 2.0,
                device=device,
            )
            points_list.append(top_points)

        points = self.backend.math.concatenate(points_list, axis=0)
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        """Sample on a grid from all surfaces."""
        self.geometry._move_params(device)
        # Surface areas
        side_area = 2 * math.pi * self.geometry.radius * self.geometry.height
        cap_area = math.pi * self.geometry.radius**2
        total_area = side_area + 2 * cap_area

        n_side = max(1, int(n_points * side_area / total_area))
        n_cap = max(1, int((n_points - n_side) / 2))

        points_list = []

        # Sample side surface
        if n_side > 0:
            side_points = self._sample_grid_side(n_side, device=device)
            points_list.append(side_points)

        # Sample bottom cap
        if n_cap > 0:
            bottom_points = self._sample_grid_circle(
                n_cap, self.geometry.center[2] - self.geometry.height / 2.0, device=device
            )
            points_list.append(bottom_points)

        # Sample top cap
        remaining = n_points - n_side - n_cap
        if remaining > 0:
            top_points = self._sample_grid_circle(
                remaining,
                self.geometry.center[2] + self.geometry.height / 2.0,
                device=device,
            )
            points_list.append(top_points)

        points = self.backend.math.concatenate(points_list, axis=0)
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def _sample_random_circle(self, n_points: int, z_val: float, device: Device):
        """Sample random points on a circle."""
        r = self.geometry.radius * self.backend.math.sqrt(
            self.backend.random.uniform((n_points, 1), device=device)
        )
        phi = 2 * math.pi * self.backend.random.uniform((n_points, 1), device=device)
        x = r * self.backend.math.cos(phi) + self.geometry.center[0]
        y = r * self.backend.math.sin(phi) + self.geometry.center[1]
        z = self.backend.math.full((n_points, 1), z_val, device=device)
        return self.backend.math.concatenate([x, y, z], axis=1)

    def _sample_grid_circle(self, n_points: int, z_val: float, device: Device):
        """Sample grid points on a circle using sunflower arrangement."""
        grid = self.geometry._equidistant_points_in_circle(n_points, device=device)
        xy = self.geometry.radius * grid
        z = self.backend.math.full((n_points, 1), z_val)
        xy += self.geometry.center[:2][None, :]
        points = self.backend.math.concatenate([xy, z], axis=1)
        return points

    def _sample_random_side(self, n_points: int, device: Device):
        """Sample random points on the cylindrical side surface."""
        phi = 2 * math.pi * self.backend.random.uniform((n_points, 1), device=device)
        z = self.backend.random.uniform(
            shape=(n_points, 1),
            low=self.geometry.center[2] - self.geometry.height / 2.0,
            high=self.geometry.center[2] + self.geometry.height / 2.0,
            device=device,
        )
        x = self.geometry.radius * self.backend.math.cos(phi) + self.geometry.center[0]
        y = self.geometry.radius * self.backend.math.sin(phi) + self.geometry.center[1]
        return self.backend.math.concatenate([x, y, z], axis=1)

    def _sample_grid_side(self, n_points: int, device: Device):
        """Sample grid points on the cylindrical side surface."""
        n_phi = max(1, int(math.sqrt(n_points)))
        n_z = max(1, (n_points + n_phi - 1) // n_phi)

        phi = self.backend.math.linspace(
            0, 2 * math.pi, n_phi, endpoint=False, device=device
        )
        z_vals = self.backend.math.linspace(
            self.geometry.center[2] - self.geometry.height / 2.0,
            self.geometry.center[2] + self.geometry.height / 2.0,
            n_z,
            device=device,
        )

        phi_grid, z_grid = self.backend.math.meshgrid(phi, z_vals)
        phi_flat = phi_grid.flatten()[:n_points]
        z_flat = z_grid.flatten()[:n_points]

        x = (
            self.geometry.radius * self.backend.math.cos(phi_flat)
            + self.geometry.center[0]
        ).reshape(-1, 1)
        y = (
            self.geometry.radius * self.backend.math.sin(phi_flat)
            + self.geometry.center[1]
        ).reshape(-1, 1)
        z = self.backend.math.unsqueeze(z_flat, axis=-1)

        points = self.backend.math.concatenate([x, y, z], axis=-1)
        return points

    def normal(self, points, device: Device = cpu):
        """Compute outward normal vectors at boundary points."""
        self.geometry._move_params(device)
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 3)
        normals = self.backend.math.zeros_like(points, device=device)

        z_min = self.geometry.center[2] - self.geometry.height / 2.0
        z_max = self.geometry.center[2] + self.geometry.height / 2.0

        # On bottom cap
        bottom_mask = self.backend.math.isclose(points[:, 2], z_min)
        bottom_normal = self.backend.build_tensor([0, 0, -1], dtype=Float32)
        normals[bottom_mask] = self.backend.to(bottom_normal, device=device)

        # On top cap
        top_mask = self.backend.math.isclose(points[:, 2], z_max)
        top_normal = self.backend.build_tensor([0, 0, 1], dtype=Float32)
        normals[top_mask] = self.backend.to(top_normal, device=device)

        # On side surface: radial direction
        side_mask = ~(bottom_mask | top_mask)
        radial = points[side_mask, :2] - self.geometry.center[:2]
        radial_dist = self.backend.linalg.norm(radial, order=2, axis=1, keepdims=True)
        normals[side_mask, :2] = radial / radial_dist

        return normals

    def _get_volume(self):
        """Surface area of the cylinder."""
        side_area = 2 * math.pi * self.geometry.radius * self.geometry.height
        cap_area = 2 * math.pi * self.geometry.radius**2
        return side_area + cap_area
