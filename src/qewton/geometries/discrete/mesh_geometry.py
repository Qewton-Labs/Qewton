from __future__ import annotations
import math

from qewton.config.variables import Variable
from qewton.geometries.base import Geometry, DiscreteGeometry, BoundaryGeometry
from qewton.geometries.discrete.mesh import Mesh
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Bool, Int32


class MeshGeometry(DiscreteGeometry[TensorType]):
    """A geometry object representing a simplex mesh.

    Args:
        variable (Variable): The variable connected to this geometry.
        mesh (Mesh): The underlying mesh object.
        discretization_of (Geometry | None, optional): The geometry this
            mesh is a discretization of. Defaults to None.
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        variable: Variable,
        mesh: Mesh,
        discretization_of: Geometry | None = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        assert (
            len(mesh.vertices[0]) == variable.dim
        ), "Dimension of variable must match dimension of mesh vertices."
        self.mesh = mesh
        super().__init__(
            variable=variable, shape=(mesh.vertices.shape[0],), backend=backend
        )
        if discretization_of is not None:
            self.discretization_of = discretization_of
        # For checking points inside the mesh:
        self.inv_A: TensorType | None = None  # Inverse matrix for bary. coords.
        self.v0: TensorType | None = None  # Origin of each simplex
        self.bbox_min: TensorType | None = None  # bounding box of each simplex
        self.bbox_max: TensorType | None = None
        self.contains_tol = 1.0e-5

    @classmethod
    def load_mesh(
        cls,
        variable: Variable,
        file_path: str,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> MeshGeometry:
        """Loads a *volume* mesh from a path.

        Args:
            variable (Variable): The variable connected to this geometry.
            file_path (str): The path to the mesh file. This uses the meshio library
                load the mesh. Supported formats are .msh/.vtk/.vtu/.xdmf/.inp and
                more, see the meshio documentation.
            backend (type[ComputingBackend[TensorType]], optional):
                Defaults to DEFAULT_DL_BACKEND.

        Returns:
            MeshGeometry: _description_
        """
        return cls(variable=variable, mesh=Mesh.load_mesh(file_path, backend=backend))

    def __and__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def __sub__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def create_mesh(
        self, max_vertex_distance: float | None = None, device: Device = cpu
    ) -> MeshGeometry:
        return self  # TODO: max_vertex_distance is currently ignored

    def bounding_box(self):
        bounding_box = []
        for i in range(self.variable.dim):
            min_val = self.backend.math.min(self.mesh.vertices[:, i])
            max_val = self.backend.math.max(self.mesh.vertices[:, i])
            bounding_box.append(min_val)
            bounding_box.append(max_val)
        return self.backend.build_tensor(bounding_box)

    def _get_volume(self):
        cell_volumes = self.mesh.compute_cell_volumes()
        return self.backend.math.sum(cell_volumes)

    def get_marker(self, marker):
        return self.get_submesh(marker=marker)

    def mesh_info(self):
        """Print out some general information about the mesh."""
        print("Number of vertices:", self.mesh.vertex_count)
        print("Number of cells:", len(self.mesh.cells))
        if len(self.mesh.marker_labels) > 0:
            print("The mesh has the markers:", self.mesh.marker_labels)
        elif self.mesh.cell_markers is not None:
            print(
                "The mesh has the markers:",
                self.backend.math.unique(self.mesh.cell_markers),
            )
        else:
            print("No markers are known in the mesh.")

    @property
    def boundary(self) -> MeshBoundaryGeometry:
        return super().boundary  # type: ignore

    def create_boundary(self) -> MeshBoundaryGeometry:
        return MeshBoundaryGeometry(self)

    def sample_random_uniform_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        return self.mesh.sample_random_from_vertices(n_points=n_points, device=device)[0]

    def sample_grid_from_discretization(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        return self.mesh.sample_grid_from_vertices(n_points=n_points, device=device)[0]

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu
    ) -> TensorType:
        return self.mesh.sample_random_inside(n_points=n_points, device=device)[0]

    def sample_grid(self, n_points: int, device: Device | str = cpu) -> TensorType:
        mins = self.bounding_box()[::2]
        maxs = self.bounding_box()[1::2]

        dim = len(mins)

        bbox_lengths = maxs - mins
        bbox_volume = self.backend.math.prod(bbox_lengths)
        # Sample first in bounding box
        scaled_points = int(math.ceil(bbox_volume / self.volume() * n_points))

        # Resolution per axis proportional to side length
        points_per_axis = self.backend.math.clip(
            self.backend.math.floor(
                scaled_points ** (1 / dim) * bbox_lengths / (bbox_volume ** (1 / dim))
            ),
            2,
            1.0e8,
        )
        points_per_axis = self.backend.cast_dtype(points_per_axis, Int32)

        axes = [
            self.backend.math.linspace(lo, hi, n, device=device)
            for lo, hi, n in zip(mins, maxs, points_per_axis)
        ]

        meshgrid = self.backend.math.meshgrid(*axes, indexing="ij")

        grid_points = self.backend.math.stack(
            [g.ravel() for g in meshgrid],
            axis=-1,
        )
        # Check what points are inside
        grid_points = grid_points[self.contains(grid_points)]

        # Add random points
        missing_n = n_points - len(grid_points)
        if missing_n > 0:
            random_points = self.sample_random_uniform(missing_n, device=device)
            grid_points = self.backend.math.concatenate(
                [grid_points, random_points],
                axis=0,
            )
        elif missing_n < 0:  # or remove some if we have to many
            idx = self.backend.random.permutation(
                self.backend.math.arange(len(grid_points)), device=device
            )[:n_points]
            grid_points = grid_points[idx]
        return grid_points

    def contains(self, points):
        _, _, found = self.locate(points)
        return found

    def interpolate_to(self, points, values):
        """Interpolates a per-vertex field onto arbitrary target points, via
        linear (barycentric) interpolation inside whichever simplex contains
        each point. Generic over the points, not plane-specific - a plane
        slice is just one choice of `points`, a regular 3D resampling grid or
        a scattered comparison set are others equally valid.

        Args:
            points: Target points, shape (n_points, dim).
            values: Per-vertex field, shape (n_vertices,) or
                (n_vertices, *feature_shape).

        Returns:
            Interpolated values, shape (n_points,) or (n_points, *feature_shape)
            matching `values`'s trailing shape. Points outside the mesh are
            NaN rather than extrapolated, so callers can render/ignore them as
            missing (e.g. Scale.observe() already uses nanmin/nanmax).
        """
        cell_idx, weights, found = self.locate(points)
        vertex_idx = self.mesh.cells[cell_idx]  # (n_points, dim + 1)
        vertex_values = values[vertex_idx]  # (n_points, dim + 1, *feature_shape)

        w = weights
        while w.ndim < vertex_values.ndim:
            w = w[..., None]
        interpolated = self.backend.math.sum(w * vertex_values, axis=1)

        not_found = ~found
        while not_found.ndim < interpolated.ndim:
            not_found = not_found[..., None]
        nan = self.backend.build_tensor(float("nan"))
        return self.backend.math.where(not_found, nan, interpolated)

    def locate(self, points):
        """For each point, finds a containing simplex and its barycentric
        weights - the same search `contains()` does, but keeping the cell
        index and weights it computes along the way instead of collapsing
        them to a boolean. Used by interpolation onto arbitrary target points
        (e.g. a plane slice), where those weights double as the interpolation
        weights - no separate lookup needed.

        Args:
            points: Points to locate, shape (n_points, dim).

        Returns:
            cell_idx: int array (n_points,) - index into self.mesh.cells of a
                simplex containing each point. Arbitrary (0) where not found.
            weights: float array (n_points, dim + 1) - barycentric weights
                against self.mesh.cells[cell_idx[i]]'s vertices, in vertex
                order (v0 first). Rows for points not found are all zero.
            found: bool array (n_points,) - whether a containing simplex was
                found for each point.
        """
        self._build_barycentric_cache()

        if len(points) < len(self.mesh.cells):
            return self._locate_point_based(points)
        return self._locate_cell_based(points)

    def _build_barycentric_cache(self):
        if self.inv_A is None or self.v0 is None:
            vertices = self.mesh.vertices[self.mesh.cells]
            self.bbox_min = self.backend.math.min(vertices, axis=1)
            self.bbox_max = self.backend.math.max(vertices, axis=1)
            self.v0 = vertices[:, 0]
            mat_A = vertices[:, 1:] - self.v0[:, None]  # type: ignore
            self.inv_A = self.backend.linalg.inv(mat_A)

    def _locate_point_based(self, points):
        n_points = len(points)
        dim = self.variable.dim
        cell_idx = self.backend.math.zeros(n_points, dtype=self.backend.dtypes[Int32])
        weights = self.backend.math.zeros((n_points, dim + 1))  # type: ignore
        found = self.backend.math.zeros(n_points, dtype=self.backend.dtypes[Bool])

        for i, p in enumerate(points):
            candidates = self.backend.math.where(
                self.backend.math.all(p >= self.bbox_min, axis=1)
                & self.backend.math.all(p <= self.bbox_max, axis=1)
            )[0]
            for cell in candidates:
                u = (p - self.v0[cell]) @ self.inv_A[cell]  # type: ignore
                l0 = 1.0 - u.sum()
                if l0 >= -self.contains_tol and self.backend.math.all(
                    u >= -self.contains_tol
                ):
                    found[i] = True
                    cell_idx[i] = cell
                    weights[i, 0] = l0
                    weights[i, 1:] = u
                    break

        return cell_idx, weights, found

    def _locate_cell_based(self, points):
        n_points = len(points)
        dim = self.variable.dim
        cell_idx = self.backend.math.zeros(n_points, dtype=self.backend.dtypes[Int32])
        weights = self.backend.math.zeros((n_points, dim + 1))  # type: ignore
        found = self.backend.math.zeros(n_points, dtype=self.backend.dtypes[Bool])

        for cell in range(len(self.mesh.cells)):
            # bbox filter
            mask = (
                ~found
                & self.backend.math.all(
                    points >= self.bbox_min[cell], axis=1  # type: ignore
                )
                & self.backend.math.all(
                    points <= self.bbox_max[cell], axis=1  # type: ignore
                )
            )

            idx = self.backend.math.where(mask)[0]

            if len(idx) == 0:
                continue

            # barycentric test
            u = (points[idx] - self.v0[cell]) @ self.inv_A[cell]  # type: ignore
            l0 = 1.0 - self.backend.math.sum(u, axis=1)
            bary_mask = self.backend.math.logical_and(
                l0 >= -self.contains_tol,
                self.backend.math.all(u >= -self.contains_tol, axis=1),
            )
            good = idx[bary_mask]
            found[good] = True
            cell_idx[good] = cell
            weights[good, 0] = l0[bary_mask]
            weights[good, 1:] = u[bary_mask]
        return cell_idx, weights, found

    def get_submesh(self, marker: int | str) -> MeshGeometry:
        """Returns a submesh of the main mesh that only contains the provided
        marker.

        Args:
            marker (int | str): The marker of the submesh

        Returns:
            MeshGeometry: A new mesh geometry that only consists of the
                mesh with the marker.
        """
        return MeshGeometry(
            variable=self.variable,
            mesh=self.mesh.get_submesh(marker),
            discretization_of=self.discretization_of,
            backend=self.backend,
        )


class MeshBoundaryGeometry(BoundaryGeometry[TensorType]):

    def __init__(self, geometry: MeshGeometry):
        assert isinstance(geometry, MeshGeometry)
        super().__init__(geometry)
        self.mesh = geometry.mesh.get_boundary_mesh()
        self.geometry: MeshGeometry = geometry  # type: ignore

        self.face_bbox_min: TensorType | None = None  # bounding box of each face
        self.face_bbox_max: TensorType | None = None
        self.v0: TensorType | None = None  # Origin of each face simplex

    def bounding_box(self):
        return self.geometry.bounding_box()

    def contains(self, points):
        self._build_face_bbox()
        if len(points) < len(self.mesh.cells):
            return self._contains_point_based_search(points)[0]
        return self._contains_cell_based_search(points)[0]

    def create_mesh(
        self, max_vertex_distance: float | None = None, device: Device = cpu
    ) -> MeshBoundaryGeometry:
        return self

    def _contains_point_based_search(self, points):
        point_inside = self.backend.math.zeros(
            len(points), dtype=self.backend.dtypes[Bool]
        )
        cell_idx = self.backend.math.zeros(len(points), dtype=self.backend.dtypes[Int32])
        for i, p in enumerate(points):
            # Check if points are close to face
            candidates = self.backend.math.where(
                self.backend.math.all(p >= self.face_bbox_min, axis=1)
                & self.backend.math.all(p <= self.face_bbox_max, axis=1)
            )
            # Do concrete distance check via normal computation
            for cell in candidates[0]:
                dist = self.backend.math.dot(
                    p - self.v0[cell],  # type: ignore
                    self.geometry.mesh.boundary_normals[cell],
                )
                if abs(dist) <= self.geometry.contains_tol:
                    point_inside[i] = True
                    cell_idx[i] = cell
                    break
        return point_inside, cell_idx

    def _contains_cell_based_search(self, points):
        point_inside = self.backend.math.zeros(
            len(points), dtype=self.backend.dtypes[Bool]
        )
        cell_idx = self.backend.math.zeros(len(points), dtype=self.backend.dtypes[Int32])
        for cell in range(len(self.mesh.cells)):
            mask = (
                ~point_inside
                & self.backend.math.all(
                    points >= self.face_bbox_min[cell], axis=1  # type: ignore
                )
                & self.backend.math.all(
                    points <= self.face_bbox_max[cell], axis=1  # type: ignore
                )
            )
            idx = self.backend.math.where(mask)[0]
            if len(idx) == 0:
                continue

            distance = self.backend.math.dot(
                (points[idx] - self.v0[cell]),  # type: ignore
                self.geometry.mesh.boundary_normals[cell],
            )
            bary_mask = self.backend.math.abs(distance) <= self.geometry.contains_tol
            point_inside[idx[bary_mask]] = True
            cell_idx[idx[bary_mask]] = cell
        return point_inside, cell_idx

    def __and__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def __sub__(self, other):
        raise NotImplementedError("Mesh combinations are not supported yet.")

    def _get_volume(self):
        cell_volumes = self.mesh.compute_cell_volumes()
        return self.backend.math.sum(cell_volumes)

    def sample_random_uniform_from_discretization(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        points, idx = self.mesh.sample_random_from_vertices(
            n_points=n_points, device=device
        )
        if include_normals:
            self._move_normals(device=device)
            normals = self.geometry.mesh.boundary_normals_at_vertex[idx]
            return points, normals
        return points

    def _move_normals(self, device: Device | str):
        self.geometry.mesh.boundary_normals_at_vertex = self.backend.to(
            self.geometry.mesh.boundary_normals_at_vertex, device=device
        )
        self.geometry.mesh.boundary_normals = self.backend.to(
            self.geometry.mesh.boundary_normals, device=device
        )

    def sample_grid_from_discretization(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        points, idx = self.mesh.sample_grid_from_vertices(
            n_points=n_points, device=device
        )
        if include_normals:
            self._move_normals(device=device)
            normals = self.geometry.mesh.boundary_normals_at_vertex[idx]
            return points, normals
        return points

    def sample_random_uniform(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        points, idx = self.mesh.sample_random_inside(n_points=n_points, device=device)
        if include_normals:
            self._move_normals(device=device)
            normals = self.geometry.mesh.boundary_normals[idx]
            return points, normals
        return points

    def sample_grid(
        self, n_points: int, device: Device | str = cpu, include_normals: bool = False
    ):
        # Allocate points based on area:
        n_areas, local_n = self._compute_local_distribution(n_points)

        # Build grid over all faces
        all_points = []
        face_idx = []
        face_vertices = self.mesh.vertices[self.mesh.cells]
        if face_vertices.shape[1] == 3:
            grid_fn = self._face_grid
        elif face_vertices.shape[1] == 2:
            grid_fn = self._line_grid
        else:
            raise NotImplementedError(
                f"No boundary sampling implemented for dimension {face_vertices.shape[1]}"
            )

        for area_counter in range(n_areas):
            new_points = grid_fn(local_n[area_counter], face_vertices[area_counter])
            all_points.extend(new_points)
            face_idx.extend([area_counter] * len(new_points))

        points = self.backend.math.vstack(all_points)
        face_idx = self.backend.build_tensor(face_idx, dtype=self.backend.dtypes[Int32])
        # Check how many points we have and either cut them or add some more
        random_normals = None
        if len(points) > n_points:
            idx = self.backend.random.permutation(
                self.backend.math.arange(len(points)), device=device
            )[:n_points]
            points = points[idx]
            face_idx = face_idx[idx]
        elif len(points) < n_points:
            missing_n = n_points - len(points)
            random_points, random_normals = self.sample_random_uniform(
                missing_n, include_normals=include_normals, device=device
            )
            points = self.backend.math.concatenate([points, random_points], axis=0)

        if include_normals:
            self._move_normals(device=device)
            normals = self.geometry.mesh.boundary_normals[face_idx]
            if random_normals is not None:
                normals = self.backend.math.concatenate([normals, random_normals], axis=0)
            return points, normals
        return points

    def _compute_local_distribution(self, n_points):
        total_area = self.volume()
        local_area: TensorType = self.mesh.cell_volumes  # type: ignore
        n_areas = len(local_area)
        local_n = self.backend.math.maximum(
            1, self.backend.math.round(local_area / total_area * n_points)  # type: ignore
        ).astype(int)
        # check if we have enough points or not:
        diff = n_points - self.backend.math.sum(local_n)
        if diff != 0:
            # Fix number by adding or removing points, starting from the biggest area
            idx = self.backend.math.argsort(local_area)[::-1]
            for i in range(abs(diff)):
                local_n[idx[i % n_areas]] += self.backend.math.sign(diff)
        return n_areas, local_n

    def _face_grid(self, n_points: int, face_vertices):
        n_i = int(math.ceil(math.sqrt(2 * n_points)))
        all_points = []
        for i in range(n_i):
            for j in range(n_i - i):
                u = (i + 1) / (n_i + 2)  # dont include 1 and 0
                v = (j + 1) / (n_i + 2)
                w = 1 - u - v
                point = u * face_vertices[0] + v * face_vertices[1] + w * face_vertices[2]
                all_points.append(point)
        return all_points

    def _line_grid(self, n_points: int, face_vertices):
        all_points = []
        for i in range(n_points):
            u = (i + 1) / (n_points + 2)
            v = 1 - u
            point = u * face_vertices[0] + v * face_vertices[1]
            all_points.append(point)
        return all_points

    def _build_face_bbox(self):
        if self.face_bbox_min is None or self.face_bbox_max is None:
            vertices = self.mesh.vertices[self.mesh.cells]
            self.v0 = vertices[:, 0]
            self.face_bbox_min = (
                self.backend.math.min(vertices, axis=1) - self.geometry.contains_tol
            )
            self.face_bbox_max = (
                self.backend.math.max(vertices, axis=1) + self.geometry.contains_tol
            )

    def normal(self, points, device: Device | str = cpu):
        self._build_face_bbox()
        if len(points) < len(self.mesh.cells):
            point_found, cell_idx = self._contains_point_based_search(points)
        else:
            point_found, cell_idx = self._contains_cell_based_search(points)[0]
        normals = self.backend.math.zeros_like(points, device=device)
        normals[point_found] = self.geometry.mesh.boundary_normals[cell_idx[point_found]]
        return normals

    def get_submesh(self, marker: int | str) -> MeshBoundaryGeometry:
        sub_mesh_geo = MeshBoundaryGeometry(self.geometry)
        sub_mesh_geo.mesh = self.mesh.get_submesh(marker)
        return sub_mesh_geo
