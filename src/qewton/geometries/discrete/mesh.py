from __future__ import annotations
import math

import numpy as np


class Mesh:

    def __init__(
        self,
        vertices: list[list[float]] | np.ndarray,
        cells: list[list[int]] | np.ndarray,
        cell_markers: list[int] | None | np.ndarray = None,
        faces: list[list[int]] | None | np.ndarray = None,
        face_markers: list[int] | None | np.ndarray = None,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.cells = np.asarray(cells, dtype=np.int32)
        self.cell_markers = np.asarray(cell_markers) if cell_markers is not None else None
        self.faces = np.asarray(faces, dtype=np.int32) if faces is not None else None
        self.face_markers = np.asarray(face_markers) if face_markers is not None else None

        # Data for normals and volumes that are only computed once
        self.cell_volumes: np.ndarray | None = None
        self.cell_probability_weights: np.ndarray | None = None
        self.boundary_normals: np.ndarray = np.empty((0, self.vertices.shape[1]))
        self.boundary_normals_at_vertex: np.ndarray = np.empty(
            (0, self.vertices.shape[1])
        )

        self.topological_dim = len(cells[0])
        self._find_boundary_facets()

        # TODO: Add names <-> marker connection

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def _find_boundary_facets(self):
        # Find boundary faces:
        n = self.cells.shape[1]
        # Collect all faces of each cell
        facets = np.concatenate(
            [np.sort(np.delete(self.cells, i, axis=1), axis=1) for i in range(n)],
            axis=0,
        )
        # Also remember where they are from at what is the missing vertex
        cell_ids = np.concatenate([np.arange(len(self.cells)) for _ in range(n)])
        missing_vertex_ids = np.concatenate(
            [np.full(len(self.cells), i) for i in range(n)]
        )
        # Find the elements that only appear once -> boundary face
        unique_facets, first_idx, counts = np.unique(
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
            normals = np.cross(
                b_vertex[:, 1] - b_vertex[:, 0], b_vertex[:, 2] - b_vertex[:, 0]
            )
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        elif self.boundary_faces.shape[1] == 2:
            normals = b_vertex[:, 1] - b_vertex[:, 0]
            normals_save = normals[:, 0].copy()
            normals[:, 0] = normals[:, 1]
            normals[:, 1] = -normals_save
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        else:  # 1d case:
            normals = np.array([[-1.0], [1.0]])
        # Fix sign of the normal vectors:
        flip = (
            np.sum(
                normals * (opposite_v - b_vertex.mean(axis=1)),
                axis=1,
            )
            > 0
        )
        normals[flip] *= -1
        self.boundary_normals = normals
        # Compute also the normals at the vertices (take the average of the
        # adjacent faces)
        vertex_ids = self.boundary_faces.ravel()
        num_vertices = vertex_ids.max() + 1
        self.boundary_normals_at_vertex = np.zeros(
            (num_vertices, self.boundary_normals.shape[1])
        )
        for f, verts in enumerate(self.boundary_faces):
            self.boundary_normals_at_vertex[verts[0]] += normals[f]
            self.boundary_normals_at_vertex[verts[1]] += normals[f]
        # np.add.at(
        #     self.boundary_normals_at_vertex,
        #     vertex_ids,
        #     np.repeat(normals, self.boundary_normals.shape[1], axis=0),
        # )
        self.boundary_normals_at_vertex /= np.linalg.norm(
            self.boundary_normals_at_vertex, axis=1, keepdims=True
        )

    @classmethod
    def load_mesh(
        cls, file_path, marker_key: str | None = None, default_cell_tags: int = -1
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
            msh, marker_key, default_cell_tags, p_key, priority[key_idx + 1]
        )
        return cls(
            vertices=msh.points,
            cells=cells,
            cell_markers=cell_markers,
            faces=faces,
            face_markers=face_markers,
        )

    @classmethod
    def _read_markers_from_file(
        cls, msh, marker_key, default_cell_tags, cell_key, face_key
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
        cell_markers = default_cell_tags * np.ones(len(cells), dtype=np.int32)
        face_markers = []
        # Try to read out the markers from the mesh:
        if marker_key in msh.cell_data:
            marker_counter = 0
            for i, block in enumerate(msh.cells):
                # Cell data has markers for each block
                markers = msh.cell_data[marker_key][i]
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

    def compute_cell_volumes(self) -> np.ndarray:
        if self.cell_volumes is not None:
            return self.cell_volumes
        dim = self.cells.shape[1] - 1
        cell_corners = self.vertices[self.cells]
        jacobian = (cell_corners[:, 1:, :] - cell_corners[:, :1, :]).transpose(0, 2, 1)
        gram_matrix = np.einsum("...ij,...ik->...jk", jacobian, jacobian)
        det = np.linalg.det(gram_matrix)
        self.cell_volumes = np.sqrt(np.maximum(det, 0.0)) / math.factorial(dim)
        return self.cell_volumes  # type: ignore

    def compute_cell_probability_weights(self) -> np.ndarray:
        if self.cell_probability_weights is not None:
            return self.cell_probability_weights
        cell_volumes = self.compute_cell_volumes()
        self.cell_probability_weights = cell_volumes / np.sum(cell_volumes)
        return self.cell_probability_weights  # type: ignore

    def get_boundary_mesh(self) -> Mesh:
        boundary_indices = self.boundary_faces.flatten()
        boundary_indices = np.unique(boundary_indices)
        boundary_vertices = self.vertices[boundary_indices]

        # Map faces to new vertex ordering:
        inverse_map = np.full(self.vertices.shape[0], -1, dtype=int)
        inverse_map[boundary_indices] = np.arange(len(boundary_indices))
        remapped_faces = inverse_map[self.boundary_faces]

        # Transfer face mapping
        # Check if a face was completely at the boundary and was marked
        # -> transfer this marking to a cell marking
        mapped_face_markers = None
        if self.faces is not None and self.face_markers is not None:
            default_marker = np.min(self.face_markers) - 1

            marked_faces_mapping = inverse_map[self.faces]
            # if we have a -1, the face is inside and we dont keep it
            mask = np.all(marked_faces_mapping != -1, axis=1)
            mapped_faces = self.faces[mask]
            mapped_face_markers = self.face_markers[mask]

            face_to_marker = {
                tuple(sorted(face)): marker
                for face, marker in zip(mapped_faces, mapped_face_markers)
            }

            mapped_face_markers = np.array(
                [
                    face_to_marker.get(tuple(sorted(face)), default_marker)
                    for face in self.boundary_faces
                ]
            )

        return Mesh(
            vertices=boundary_vertices,
            cells=remapped_faces,
            cell_markers=mapped_face_markers,
        )

    def get_submesh(self, marker: int) -> Mesh:
        if self.cell_markers is None:
            raise ValueError("No markers in mesh available.")
        mask = self.cell_markers == marker
        if not np.any(mask):
            raise ValueError(f"Marker {marker} not found in mesh. \
                Available markers: {np.unique(self.cell_markers)}")

        new_cells = self.cells[mask]
        remaining_indices = np.unique(new_cells.flatten())
        remaining_vertices = self.vertices[remaining_indices]

        inverse_map = np.full(self.vertices.shape[0], -1, dtype=int)
        inverse_map[remaining_indices] = np.arange(len(remaining_indices))
        new_cells = inverse_map[new_cells]

        return Mesh(
            vertices=remaining_vertices,
            cells=new_cells,
            cell_markers=self.cell_markers[mask],
        )

    def sample_random_from_vertices(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(
            np.arange(self.vertex_count),
            size=n_points,
            replace=(n_points > self.vertex_count),
        )
        return self.vertices[idx], idx

    def sample_grid_from_vertices(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Could be done distance based
        if n_points <= self.vertex_count:
            idx = np.random.permutation(self.vertex_count)[:n_points]
            return self.vertices[idx], idx
        # Else, more points wanted then exist in the mesh itself
        vertex_count = self.vertex_count

        reps = n_points // vertex_count
        rem = n_points % vertex_count

        base = np.tile(self.vertices, (reps, 1))
        base_idx = np.tile(np.arange(vertex_count), (reps, 1))

        extra_idx = np.random.permutation(vertex_count)[:rem]
        extra = self.vertices[extra_idx]

        return np.vstack([base, extra]), np.vstack([base_idx, extra_idx])

    def sample_random_inside(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        chosen_cells = np.random.choice(
            len(self.cells),
            size=n_points,
            p=self.compute_cell_probability_weights(),
        )

        simplices = self.cells[chosen_cells]
        verts = self.vertices[simplices]

        # (Dirichlet sampling via exponential function)
        w = np.random.exponential(size=verts.shape[:2])
        w /= w.sum(axis=1, keepdims=True)

        # Convex combination
        return np.einsum("ni,nid->nd", w, verts), chosen_cells
