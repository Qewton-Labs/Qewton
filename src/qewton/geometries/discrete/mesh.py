from __future__ import annotations
from typing import Generic
import math

# import numpy as np

from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Int32, Float32


class Mesh(Generic[TensorType]):

    def __init__(
        self,
        vertices: list[list[float]] | TensorType,
        cells: list[list[int]] | TensorType,
        cell_markers: list[int] | None | TensorType = None,
        faces: list[list[int]] | None | TensorType = None,
        face_markers: list[int] | None | TensorType = None,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.backend = backend
        self.vertices = backend.build_tensor(vertices, dtype=Float32)
        self.cells = backend.build_tensor(cells, dtype=Int32)
        self.cell_markers = (
            backend.build_tensor(cell_markers, dtype=Int32)
            if cell_markers is not None
            else None
        )
        self.faces = (
            backend.build_tensor(faces, dtype=Int32) if faces is not None else None
        )
        self.face_markers = (
            backend.build_tensor(face_markers, dtype=Int32)
            if face_markers is not None
            else None
        )

        # Data for normals and volumes that are only computed once
        # TODO: Update all of this
        self.cell_volumes: TensorType | None = None
        self.cell_probability_weights: TensorType | None = None
        self.boundary_normals: TensorType = self.backend.math.empty(
            (0, self.vertices.shape[1])
        )
        self.boundary_normals_at_vertex: TensorType = self.backend.math.empty(
            (0, self.vertices.shape[1])
        )

        self._find_boundary_facets()

        # TODO: Add names <-> marker connection

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)  # type: ignore

    def _find_boundary_facets(self):
        if len(self.cells.shape) <= 1:
            return
        # Find boundary faces:
        n = self.cells.shape[1]
        # Collect all faces of each cell
        facets = self.backend.math.concatenate(
            [
                self.backend.math.sort(
                    self.backend.math.delete(self.cells, i, axis=1), axis=1
                )
                for i in range(n)
            ],
            axis=0,
        )
        # Also remember where they are from at what is the missing vertex
        cell_ids = self.backend.math.concatenate(
            [self.backend.math.arange(len(self.cells)) for _ in range(n)]
        )
        missing_vertex_ids = self.backend.math.concatenate(
            [self.backend.math.full(len(self.cells), i) for i in range(n)]
        )
        # Find the elements that only appear once -> boundary face
        unique_facets, first_idx, counts = self.backend.math.unique(
            facets,
            axis=0,
            return_index=True,
            return_counts=True,
        )

        boundary_mask = counts == 1
        self.boundary_faces = unique_facets[boundary_mask]
        boundary_rows = first_idx[boundary_mask]

        if len(self.boundary_faces) == 0:
            return
        # Check if orientation is outwards:
        # For this compute the normals and compare with the opposite
        # vertex that does not belong to the face
        boundary_cell_ids = cell_ids[boundary_rows]
        boundary_missing_vertex_ids = missing_vertex_ids[boundary_rows]

        self._compute_normals(boundary_cell_ids, boundary_missing_vertex_ids)

    def _compute_normals(self, boundary_cell_ids, boundary_missing_vertex_ids):
        b_vertex = self.vertices[self.boundary_faces]
        opposite_v = self.vertices[
            self.cells[
                boundary_cell_ids,
                boundary_missing_vertex_ids,
            ]
        ]
        if self.boundary_faces.shape[1] == 3:
            normals = self.backend.math.cross(
                b_vertex[:, 1] - b_vertex[:, 0], b_vertex[:, 2] - b_vertex[:, 0]
            )
            normals /= self.backend.linalg.norm(normals, order=2, axis=1, keepdims=True)
        elif self.boundary_faces.shape[1] == 2:
            normals = b_vertex[:, 1] - b_vertex[:, 0]
            normals_save = self.backend.math.copy(normals[:, 0])
            normals[:, 0] = normals[:, 1]
            normals[:, 1] = -normals_save
            normals /= self.backend.linalg.norm(normals, order=2, axis=1, keepdims=True)
        else:  # 1d case:
            normals = self.backend.build_tensor([[-1.0], [1.0]])
        # Fix sign of the normal vectors:
        flip = (
            self.backend.math.sum(
                normals * (opposite_v - b_vertex.mean(axis=1)),
                axis=1,
            )
            > 0
        )
        normals[flip] *= -1
        self.boundary_normals = normals
        # Compute also the normals at the vertices (take the average of the
        # adjacent faces)
        if self.boundary_faces.shape[1] >= 2:
            vertex_ids = self.boundary_faces.ravel()
            num_vertices = vertex_ids.max() + 1
            self.boundary_normals_at_vertex = self.backend.math.zeros(
                (num_vertices, self.boundary_normals.shape[1])
            )
            for f, verts in enumerate(self.boundary_faces):
                self.boundary_normals_at_vertex[verts[0]] += normals[f]
                self.boundary_normals_at_vertex[verts[1]] += normals[f]

            normal_norm = self.backend.linalg.norm(
                self.boundary_normals_at_vertex, axis=1, order=2, keepdims=True
            )
            normal_norm[self.backend.math.where(normal_norm < 1.0e-9)[0]] += 1.0e-9

            self.boundary_normals_at_vertex /= normal_norm

    @classmethod
    def load_mesh(
        cls,
        file_path,
        marker_key: str | None = None,
        default_cell_tags: int = -1,
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> Mesh:
        try:
            import meshio  # pylint: disable=import-outside-toplevel # type: ignore
        except ImportError as e:
            raise ImportError(
                "For loading meshes the library meshio is required. Install it via pip"
                "or load the mesh manually to pass the mesh information."
            ) from e

        msh = meshio.read(file_path)
        # Read all cell data:
        priority = ["tetra", "triangle", "line", "vertex"]
        p_key, key_idx = "", 0
        for key_idx, p_key in enumerate(priority):
            if p_key in msh.cells_dict:
                break
        # Check for markers of the cells and facets.
        cells, cell_markers, faces, face_markers = cls._read_markers_from_file(
            msh, marker_key, default_cell_tags, p_key, priority[key_idx + 1], backend
        )
        return cls(
            vertices=msh.points,
            cells=cells,
            cell_markers=cell_markers,
            faces=faces,
            face_markers=face_markers,
            backend=backend,
        )

    @classmethod
    def _read_markers_from_file(
        cls,
        msh,
        marker_key,
        default_cell_tags,
        cell_key,
        face_key,
        backend: type[ComputingBackend[TensorType]],
    ):
        # First try to find a key if not provided by the user
        if marker_key is None:
            candidates = [
                "gmsh:physical",
                "material",
                "Material",
                "region",
                "cell_tags",
            ]
            for c in candidates:
                if c in msh.cell_data:
                    marker_key = c
                    break
        # build all cells:
        cells = []
        for block in msh.cells:
            if block.type == cell_key:
                cells.extend(block.data)
        faces = []
        cell_markers = default_cell_tags * backend.math.ones(
            (len(cells),), dtype=backend.dtypes[Int32]
        )
        face_markers = []
        # Try to read out the markers from the mesh:
        if marker_key in msh.cell_data:
            marker_counter = 0
            for i, block in enumerate(msh.cells):
                # Cell data has markers for each block
                markers = backend.build_tensor(
                    msh.cell_data[marker_key][i], dtype=backend.dtypes[Int32]
                )
                # Markers of main dimension:
                if block.type == cell_key:
                    assert not any(
                        m == default_cell_tags for m in markers
                    ), f"Default marker {default_cell_tags} tag found in the mesh \
                        data. This can lead to unexpected behavior."
                    cell_markers[marker_counter : marker_counter + len(markers)] = markers
                    marker_counter += len(markers)
                # Facets marker:
                elif block.type == face_key:
                    faces.extend(block.data)
                    face_markers.extend(markers)
        elif marker_key is not None:
            print(f"Could not find cell marker information. Mesh contains \
                {msh.cell_dict.keys()} which does not have the key {marker_key}.")
        return cells, cell_markers, faces, face_markers

    def compute_cell_volumes(self) -> TensorType:
        if self.cell_volumes is not None:
            return self.cell_volumes
        dim = self.cells.shape[1] - 1
        cell_corners = self.vertices[self.cells]
        jacobian = self.backend.math.transpose(
            cell_corners[:, 1:, :] - cell_corners[:, :1, :], (0, 2, 1)
        )
        # jacobian = (cell_corners[:, 1:, :] - cell_corners[:, :1, :]).transpose(0, 2, 1)
        gram_matrix = self.backend.math.einsum("...ij,...ik->...jk", jacobian, jacobian)
        det = self.backend.linalg.det(gram_matrix)
        self.cell_volumes = self.backend.math.sqrt(
            self.backend.math.clip(det, 0.0, 1.0e10)
        ) / math.factorial(dim)
        return self.cell_volumes  # type: ignore

    def compute_cell_probability_weights(self) -> TensorType:
        if self.cell_probability_weights is not None:
            return self.cell_probability_weights
        cell_volumes = self.compute_cell_volumes()
        self.cell_probability_weights = cell_volumes / self.backend.math.sum(cell_volumes)
        return self.cell_probability_weights  # type: ignore

    def get_boundary_mesh(self) -> Mesh:
        boundary_indices = self.boundary_faces.flatten()
        boundary_indices = self.backend.math.unique(boundary_indices)
        boundary_vertices = self.vertices[boundary_indices]

        # Map faces to new vertex ordering:
        inverse_map = self.backend.math.full(
            self.vertices.shape[0], -1, dtype=self.backend.dtypes[Int32]
        )
        inverse_map[boundary_indices] = self.backend.math.arange(
            len(boundary_indices), dtype=self.backend.dtypes[Int32]
        )
        remapped_faces = inverse_map[self.boundary_faces]

        # Transfer face mapping
        # Check if a face was completely at the boundary and was marked
        # -> transfer this marking to a cell marking
        mapped_face_markers = None
        if self.faces is not None and self.face_markers is not None:
            default_marker = self.backend.math.min(self.face_markers) - 1

            marked_faces_mapping = inverse_map[self.faces]
            # if we have a -1, the face is inside and we dont keep it
            mask = self.backend.math.all(marked_faces_mapping != -1, axis=1)
            mapped_faces = self.faces[mask]
            mapped_face_markers = self.face_markers[mask]

            face_to_marker = {
                tuple(sorted(int(v) for v in face)): marker
                for face, marker in zip(mapped_faces, mapped_face_markers)
            }

            mapped_face_markers = self.backend.build_tensor(
                [
                    face_to_marker.get(
                        tuple(sorted(int(v) for v in face)), default_marker
                    )
                    for face in self.boundary_faces
                ]
            )

        return Mesh(
            vertices=boundary_vertices,
            cells=remapped_faces,
            cell_markers=mapped_face_markers,
            backend=self.backend,
        )

    def get_submesh(self, marker: int) -> Mesh:
        if self.cell_markers is None:
            raise ValueError("No markers in mesh available.")
        mask = self.cell_markers == marker
        if not self.backend.math.any(mask):
            raise ValueError(f"Marker {marker} not found in mesh. \
                Available markers: {self.backend.math.unique(self.cell_markers)}")

        new_cells = self.cells[mask]
        remaining_indices = self.backend.math.unique(new_cells.flatten())
        remaining_vertices = self.vertices[remaining_indices]

        inverse_map = self.backend.math.full(
            self.vertices.shape[0], -1, dtype=self.backend.dtypes[Int32]
        )
        inverse_map[remaining_indices] = self.backend.math.arange(
            len(remaining_indices), dtype=self.backend.dtypes[Int32]
        )
        new_cells = inverse_map[new_cells]

        return Mesh(
            vertices=remaining_vertices,
            cells=new_cells,
            cell_markers=self.cell_markers[mask],
            backend=self.backend,
        )

    def sample_random_from_vertices(
        self, n_points: int, device: Device | str = cpu
    ) -> tuple[TensorType, TensorType]:
        idx = self.backend.random.choice(
            self.backend.math.arange(self.vertex_count),
            shape=n_points,
            replace=(n_points > self.vertex_count),
            device=device,
        )
        self.vertices = self.backend.to(self.vertices, device=device)
        return self.vertices[idx], idx

    def sample_grid_from_vertices(
        self, n_points: int, device: Device | str = cpu
    ) -> tuple[TensorType, TensorType]:
        self.vertices = self.backend.to(self.vertices, device=device)
        # TODO: Could be done distance based
        if n_points <= self.vertex_count:
            idx = self.backend.random.permutation(self.vertex_count, device=device)[
                :n_points
            ]
            return self.vertices[idx], idx
        # Else, more points wanted then exist in the mesh itself
        vertex_count = self.vertex_count

        reps = n_points // vertex_count
        rem = n_points % vertex_count

        base = self.backend.math.tile(self.vertices, (reps, 1))
        base_idx = self.backend.math.tile(
            self.backend.math.arange(vertex_count, device=device), (reps,)
        )

        extra_idx = self.backend.random.permutation(vertex_count, device=device)[:rem]
        extra = self.vertices[extra_idx]

        return self.backend.math.vstack([base, extra]), self.backend.math.concatenate(
            [base_idx, extra_idx]
        )

    def sample_random_inside(
        self, n_points: int, device: Device | str = cpu
    ) -> tuple[TensorType, TensorType]:
        chosen_cells = self.backend.random.choice(
            len(self.cells),
            shape=n_points,
            p=self.compute_cell_probability_weights(),
            device=device,
        )

        self.vertices = self.backend.to(self.vertices, device=device)
        self.cells = self.backend.to(self.cells, device=device)

        simplices = self.cells[chosen_cells]
        verts = self.vertices[simplices]

        # (Dirichlet sampling via exponential function)
        w = self.backend.random.exponential(shape=verts.shape[:2], device=device)
        w /= w.sum(axis=1, keepdims=True)

        # Convex combination
        return self.backend.math.einsum("ni,nid->nd", w, verts), chosen_cells
