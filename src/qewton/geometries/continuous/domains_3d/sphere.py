import numpy as np

from qewton.geometries.continuous.base import ContinuousGeometry, ContinuousBoundaryGeometry
from qewton.config.variables import Variable


class Sphere(ContinuousGeometry):
    """Class for spheres (NumPy-only, simplified).

    Parameters
    ----------
    variable : Variable
        The variable representing the underlying 3D space.
    center : array_like
        The center of the sphere, e.g. [0, 0, 0].
    radius : float
        The radius of the sphere.
    """

    def __init__(
        self,
        variable: Variable,
        center: np.ndarray | list[float] | tuple[float, float, float],
        radius: float,
    ):
        assert variable.dim == 3
        if isinstance(center, (list, tuple)):
            center = np.array(center, dtype=float)
        self.center: np.ndarray = np.asarray(center, dtype=float)
        self.radius = float(radius)
        super().__init__(variable=variable)

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        norm = np.linalg.norm(points - self.center, axis=1).reshape(-1, 1)
        return norm <= self.radius

    def bounding_box(self):
        bounds = []
        for i in range(3):
            bounds.append(self.center[i] - self.radius)
            bounds.append(self.center[i] + self.radius)
        return np.array(bounds)

    def sample_random_uniform(self, n_points: int):
        # Sample directions from normal distribution and scale radii with cube-root
        vec = np.random.normal(size=(n_points, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        r = np.random.rand(n_points, 1) ** (1.0 / 3.0) * self.radius
        points = vec * r + self.center
        return points

    def sample_grid(self, n_points: int):
        # Create a grid inside the bounding box, keep points inside sphere.
        if n_points <= 0:
            return np.empty((0, 3))
        scaled_n = int(np.ceil((n_points * 6 / np.pi) ** (1.0 / 3.0)))
        if scaled_n < 2:
            return self.sample_random_uniform(n_points)
        axis = np.linspace(-self.radius, self.radius, scaled_n)
        X, Y, Z = np.meshgrid(axis, axis, axis, indexing="xy")
        pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        inside = np.linalg.norm(pts, axis=1) <= self.radius
        pts_inside = pts[inside]
        if len(pts_inside) >= n_points:
            selected = pts_inside[:n_points]
        else:
            # append random points until n_points
            needed = n_points - len(pts_inside)
            rand_pts = self.sample_random_uniform(needed)
            selected = np.vstack((pts_inside, rand_pts))
        # translate to center
        selected += self.center
        return selected

    def _get_volume(self):
        return (4.0 / 3.0) * np.pi * self.radius**3

    def _to_vector(self, vector):
        v = np.asarray(vector, dtype=float)
        return v

    def create_boundary(self):
        return SphereBoundary(self)


class SphereBoundary(ContinuousBoundaryGeometry):

    def __init__(self, geometry: Sphere):
        assert isinstance(geometry, Sphere)
        super().__init__(geometry)
        self.geometry: Sphere = geometry  # type: ignore

    def contains(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        norm = np.linalg.norm(points - self.geometry.center, axis=1).reshape(-1, 1)
        return np.isclose(norm, self.geometry.radius)

    def _get_volume(self):
        # Surface area
        return 4.0 * np.pi * self.geometry.radius**2

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        # sample directions via normal distribution then normalize
        vec = np.random.normal(size=(n_points, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        points = vec * self.geometry.radius + self.geometry.center
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        if n_points <= 0:
            return np.empty((0, 3))
        # Fibonacci sphere
        phi = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(0, n_points)
        y = 1 - (i / (n_points - 1)) * 2
        radius_xy = np.sqrt(1 - y * y)
        theta = phi * i
        x = radius_xy * np.cos(theta)
        z = radius_xy * np.sin(theta)
        points = np.column_stack((x, y, z))
        points *= self.geometry.radius
        points += self.geometry.center
        normals = None
        if include_normals:
            normals = self.normal(points)
        return points, normals

    def normal(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        normal = points - self.geometry.center
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        return normal
