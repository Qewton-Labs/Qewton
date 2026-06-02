from __future__ import annotations
import numpy as np

from ..base import Geometry, DiscreteGeometry, BoundaryGeometry
from .mesh import Mesh


class MeshGeometry(DiscreteGeometry):

    def __init__(self, variable, mesh: Mesh, discretization_of: Geometry | None = None):
        assert (
            len(mesh.vertices[0]) == variable.dim
        ), "Dimension of variable must match dimension of mesh vertices."
        self.mesh = mesh
        super().__init__(variable=variable, shape=mesh.vertices.shape)
        if discretization_of is not None:
            self.discretization_of = discretization_of
        # For checking points inside the mesh:
        self.inv_A: np.ndarray | None = None  # Inverse matrix for bary. coords.
        self.v0: np.ndarray | None = None  # Origin of each simplex
        self.bbox_min: np.ndarray | None = None  # bounding box of each simplex
        self.bbox_max: np.ndarray | None = None
        self.contains_tol = 1.0e-5

    def __and__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def __sub__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def create_mesh(self, max_vertex_distance: float | None = None) -> MeshGeometry:
        return self

    def bounding_box(self):
        bounding_box = []
        for i in range(self.variable.dim):
            min_val = np.min(self.mesh.vertices[:, i])
            max_val = np.max(self.mesh.vertices[:, i])
            bounding_box.append(min_val)
            bounding_box.append(max_val)
        return np.array(bounding_box)

    def _get_volume(self):
        cell_volumes = self.mesh.compute_cell_volumes()
        return np.sum(cell_volumes)

    def create_boundary(self) -> MeshBoundaryGeometry:
        return MeshBoundaryGeometry(self)

    def sample_random_uniform_from_discretization(self, n_points: int) -> np.ndarray:
        return self.mesh.sample_random_from_vertices(n_points=n_points)[0]

    def sample_grid_from_discretization(self, n_points: int) -> np.ndarray:
        return self.mesh.sample_grid_from_vertices(n_points=n_points)[0]

    def sample_random_uniform(self, n_points: int) -> np.ndarray:
        return self.mesh.sample_random_inside(n_points=n_points)[0]

    def sample_grid(self, n_points: int) -> np.ndarray:
        mins = self.bounding_box()[::2]
        maxs = self.bounding_box()[1::2]

        dim = len(mins)

        bbox_lengths = maxs - mins
        bbox_volume = np.prod(bbox_lengths)
        # Sample first in bounding box
        scaled_points = int(np.ceil(bbox_volume / self.volume() * n_points))

        # Resolution per axis proportional to side length
        points_per_axis = np.maximum(
            2,
            np.round(
                scaled_points ** (1 / dim) * bbox_lengths / (bbox_volume ** (1 / dim))
            ).astype(int),
        )

        axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(mins, maxs, points_per_axis)]

        meshgrid = np.meshgrid(*axes, indexing="ij")

        grid_points = np.stack(
            [g.ravel() for g in meshgrid],
            axis=-1,
        )
        # Check what points are inside
        grid_points = grid_points[self.contains(grid_points)]

        # Add random points
        missing_n = n_points - len(grid_points)
        if missing_n > 0:
            random_points = self.sample_random_uniform(missing_n)
            grid_points = np.concatenate(
                [grid_points, random_points],
                axis=0,
            )
        elif missing_n < 0:  # or remove some if we have to many
            idx = np.random.permutation(np.arange(len(grid_points)))[:n_points]
            grid_points = grid_points[idx]
        return grid_points

    def contains(self, points):
        if self.inv_A is None or self.v0 is None:
            vertices = self.mesh.vertices[self.mesh.cells]
            self.bbox_min = vertices.min(axis=1)
            self.bbox_max = vertices.max(axis=1)
            self.v0 = vertices[:, 0]
            mat_A = vertices[:, 1:] - self.v0[:, None]
            self.inv_A = np.linalg.inv(mat_A)

        if len(points) < len(self.mesh.cells):
            return self._contains_point_based_search(points)
        return self._contains_cell_based_search(points)

    def _contains_point_based_search(self, points):
        def point_in_simplex(p, v0, inv_A, tol=1e-12):
            u = inv_A @ (p - v0)
            l0 = 1.0 - u.sum()
            return l0 >= -tol and np.all(u >= -tol)

        point_inside = np.zeros(len(points), dtype=bool)
        for i, p in enumerate(points):
            candidates = np.where(
                np.all(p >= self.bbox_min, axis=1) & np.all(p <= self.bbox_max, axis=1)
            )[0]

            for cell in candidates:
                if point_in_simplex(
                    p, self.v0[cell], self.inv_A[cell], self.contains_tol  # type: ignore
                ):
                    point_inside[i] = True
                    break

        return point_inside

    def _contains_cell_based_search(self, points):
        point_inside = np.zeros(len(points), dtype=bool)

        for cell in range(len(self.mesh.cells)):
            # bbox filter
            mask = (
                ~point_inside
                & np.all(points >= self.bbox_min[cell], axis=1)  # type: ignore
                & np.all(points <= self.bbox_max[cell], axis=1)  # type: ignore
            )

            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            # barycentric test
            u = (points[idx] - self.v0[cell]) @ self.inv_A[cell]  # type: ignore
            l0 = 1.0 - np.sum(u, axis=1)
            bary_mask = np.logical_and(
                l0 >= -self.contains_tol, np.all(u >= -self.contains_tol, axis=1)
            )
            point_inside[idx[bary_mask]] = True
        return point_inside


class MeshBoundaryGeometry(BoundaryGeometry):

    def __init__(self, geometry: MeshGeometry):
        assert isinstance(geometry, MeshGeometry)
        super().__init__(geometry)
        self.mesh = geometry.mesh.get_boundary_mesh()
        self.geometry: MeshGeometry = geometry  # type: ignore

        self.face_bbox_min: np.ndarray | None = None  # bounding box of each face
        self.face_bbox_max: np.ndarray | None = None
        self.v0: np.ndarray | None = None  # Origin of each face simplex

    def bounding_box(self):
        return self.geometry.bounding_box()

    def contains(self, points):
        if self.face_bbox_min is None or self.face_bbox_max is None:
            vertices = self.mesh.vertices[self.mesh.cells]
            self.v0 = vertices[:, 0]
            self.face_bbox_min = vertices.min(axis=1) - self.geometry.contains_tol
            self.face_bbox_max = vertices.max(axis=1) + self.geometry.contains_tol
        if len(points) < len(self.mesh.cells):
            return self._contains_point_based_search(points)[0]
        return self._contains_cell_based_search(points)[0]

    def create_mesh(
        self, max_vertex_distance: float | None = None
    ) -> MeshBoundaryGeometry:
        return self

    def _contains_point_based_search(self, points):
        point_inside = np.zeros(len(points), dtype=bool)
        cell_idx = np.zeros(len(points), dtype=int)
        for i, p in enumerate(points):
            # Check if points are close to face
            candidates = np.where(
                np.all(p >= self.face_bbox_min, axis=1)
                & np.all(p <= self.face_bbox_max, axis=1)
            )[0]
            # Do concrete distance check via normal computation
            for cell in candidates:
                dist = np.dot(
                    p - self.v0[cell],  # type: ignore
                    self.geometry.mesh.boundary_normals[cell],
                )
                if abs(dist) <= self.geometry.contains_tol:
                    point_inside[i] = True
                    cell_idx[i] = cell
                    break
        return point_inside, cell_idx

    def _contains_cell_based_search(self, points):
        point_inside = np.zeros(len(points), dtype=bool)
        cell_idx = np.zeros(len(points), dtype=int)
        for cell in range(len(self.mesh.cells)):
            mask = (
                ~point_inside
                & np.all(points >= self.face_bbox_min[cell], axis=1)  # type: ignore
                & np.all(points <= self.face_bbox_max[cell], axis=1)  # type: ignore
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue

            distance = np.dot(
                (points[idx] - self.v0[cell]),  # type: ignore
                self.geometry.mesh.boundary_normals[cell],
            )
            bary_mask = np.abs(distance) <= self.geometry.contains_tol
            point_inside[idx[bary_mask]] = True
            cell_idx[idx[bary_mask]] = cell
        return point_inside, cell_idx

    def __and__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def __sub__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def _get_volume(self):
        cell_volumes = self.mesh.compute_cell_volumes()
        return np.sum(cell_volumes)

    def sample_random_uniform_from_discretization(
        self, n_points: int, include_normals: bool = False
    ):
        # TODO: Add normal computations
        points, idx = self.mesh.sample_random_from_vertices(n_points=n_points)
        normals = None
        if include_normals:
            pass
        return points, normals

    def sample_grid_from_discretization(
        self, n_points: int, include_normals: bool = False
    ):
        points, idx = self.mesh.sample_grid_from_vertices(n_points=n_points)
        normals = None
        if include_normals:
            pass
        return points, normals

    def sample_random_uniform(self, n_points: int, include_normals: bool = False):
        points, idx = self.mesh.sample_random_inside(n_points=n_points)
        normals = None
        if include_normals:
            normals = self.geometry.mesh.boundary_normals[idx]
        return points, normals

    def sample_grid(self, n_points: int, include_normals: bool = False):
        # TODO: Make this a better grid sampling
        points, idx = self.mesh.sample_grid_from_vertices(n_points=n_points)
        normals = None
        if include_normals:
            pass
        return points, normals

    def normal(self, points):
        if self.face_bbox_min is None or self.face_bbox_max is None:
            vertices = self.mesh.vertices[self.mesh.cells]
            self.v0 = vertices[:, 0]
            self.face_bbox_min = vertices.min(axis=1) - self.geometry.contains_tol
            self.face_bbox_max = vertices.max(axis=1) + self.geometry.contains_tol
        if len(points) < len(self.mesh.cells):
            point_found, cell_idx = self._contains_point_based_search(points)
        else:
            point_found, cell_idx = self._contains_cell_based_search(points)[0]
        normals = np.zeros_like(points)
        normals[point_found] = self.geometry.mesh.boundary_normals[cell_idx[point_found]]
        return normals
