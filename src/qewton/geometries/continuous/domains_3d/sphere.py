import math
from typing import Any

from qewton.geometries.continuous.base import (
    ContinuousGeometry,
    ContinuousBoundaryGeometry,
)
from qewton.config.variables import Variable
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Float32


class Sphere(ContinuousGeometry[TensorType]):
    """Class for spheres.

    Args:
        variable (Variable): The variable associated with the sphere, must be 3D.
        center (TensorType | list[float] | tuple[float, float, float]):
            The center of the sphere, e.g. [0, 0, 0].
        radius (float): The radius of the sphere.
        backend (type[ComputingBackend[TensorType]], optional): What backend the node
            should use for computations, etc. Defaults to the deep learning
            backend (DEFAULT_DL_BACKEND).
    """

    def __init__(
        self,
        variable: Variable,
        center: TensorType | list[float] | tuple[float, float, float],
        radius: float,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert variable.dim == 3
        super().__init__(variable=variable, backend=backend)
        self.center: TensorType = self.backend.build_tensor(center, dtype=Float32)
        self.radius = self.backend.build_tensor(radius, dtype=Float32)

    def _move_center(self, device: Device):
        self.center = self.backend.to(self.center, device=device)
        self.radius = self.backend.to(self.radius, device=device)

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 3)
        p_device = points.device if hasattr(points, "device") else cpu
        self._move_center(p_device)
        norm = self.backend.linalg.norm(points - self.center, order=2, axis=1).reshape(
            -1, 1
        )
        return norm <= self.radius

    def bounding_box(self):
        bounds = []
        for i in range(3):
            bounds.append(self.center[i] - self.radius)
            bounds.append(self.center[i] + self.radius)
        return self.backend.build_tensor(bounds)

    def sample_random_uniform(self, n_points: int, device: Device = cpu):
        # Sample directions from normal distribution and scale radii with cube-root
        vec = self.backend.random.normal(shape=(n_points, 3), device=device)
        vec /= self.backend.linalg.norm(vec, order=2, axis=1, keepdims=True)
        r = (
            self.backend.random.uniform((n_points, 1), device=device) ** (1.0 / 3.0)
            * self.radius
        )
        self._move_center(device)
        points = vec * r + self.center
        return points

    def sample_grid(self, n_points: int, device: Device = cpu):
        # Create a grid inside the bounding box, keep points inside sphere.
        scaled_n = int(math.ceil((n_points * 6 / math.pi) ** (1.0 / 3.0)))
        if scaled_n < 2:
            return self.sample_random_uniform(n_points, device=device)
        axis = self.backend.math.linspace(
            -1.0 * self.radius, self.radius, scaled_n, device=device
        )
        X, Y, Z = self.backend.math.meshgrid(axis, axis, axis, indexing="xy")
        pts = self.backend.math.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        inside = self.backend.linalg.norm(pts, order=2, axis=1) <= self.radius
        pts_inside = pts[inside]
        if len(pts_inside) >= n_points:
            selected = pts_inside[:n_points]
        else:
            # append random points until n_points
            needed = n_points - len(pts_inside)
            rand_pts = self.sample_random_uniform(needed, device=device)
            selected = self.backend.math.vstack([pts_inside, rand_pts])
        # translate to center
        self._move_center(device)
        selected += self.center
        return selected

    def _get_volume(self):
        return (4.0 / 3.0) * math.pi * self.radius**3

    def _to_vector(self, vector):
        v = self.backend.build_tensor(vector, dtype=Float32)
        return v

    def create_boundary(self):
        return SphereBoundary(self)

    def save(self) -> dict[str, Any]:
        general_save = super().save()
        general_save["center"] = self.center
        general_save["radius"] = self.radius
        return general_save


class SphereBoundary(ContinuousBoundaryGeometry[TensorType]):

    def __init__(self, geometry: Sphere):
        assert isinstance(geometry, Sphere)
        super().__init__(geometry)
        self.geometry: Sphere = geometry  # type: ignore

    def contains(self, points):
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 3)
        p_device = points.device if hasattr(points, "device") else cpu
        self.geometry._move_center(p_device)
        norm = self.backend.linalg.norm(
            points - self.geometry.center, order=2, axis=1
        ).reshape(-1, 1)
        return self.backend.math.isclose(norm, self.geometry.radius)

    def _get_volume(self):
        # Surface area
        return 4.0 * math.pi * self.geometry.radius**2

    def sample_random_uniform(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        # sample directions via normal distribution then normalize
        self.geometry._move_center(device)
        vec = self.backend.random.normal(shape=(n_points, 3), device=device)
        vec /= self.backend.linalg.norm(vec, order=2, axis=1, keepdims=True)
        points = vec * self.geometry.radius + self.geometry.center
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device = cpu, include_normals: bool = False
    ):
        # Fibonacci sphere
        self.geometry._move_center(device)
        phi = math.pi * (3.0 - math.sqrt(5.0))
        i = self.backend.math.arange(0, n_points)
        y = 1 - (i / (n_points - 1)) * 2
        radius_xy = self.backend.math.sqrt(1 - y * y)
        theta = phi * i
        x = radius_xy * self.backend.math.cos(theta)
        z = radius_xy * self.backend.math.sin(theta)
        points = self.backend.math.stack([x, y, z], axis=-1)
        points *= self.geometry.radius
        points += self.geometry.center
        if include_normals:
            normals = self.normal(points, device=device)
            return points, normals
        return points

    def normal(self, points: TensorType, device: Device = cpu):
        points = self.backend.build_tensor(points, dtype=Float32).reshape(-1, 3)
        self.geometry._move_center(device)
        normal = points - self.geometry.center
        normal /= self.backend.linalg.norm(normal, order=2, axis=1, keepdims=True)
        return normal
