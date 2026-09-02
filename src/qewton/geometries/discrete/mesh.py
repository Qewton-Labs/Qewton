from __future__ import annotations
from typing import Generic
import math

from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.config.devices import Device, cpu
from qewton.config.dtypes import Int32, Float32
from qewton.config.saving.saving import Serializable


class Mesh(Serializable, Generic[TensorType]):
    """A generic simplex mesh represented by vertices and cells.

    Args:
        vertices (list[list[float]] | TensorType): The vertices of the mesh.
        cells (list[list[int]] | TensorType): The cells of the mesh. A simplex always has
            d+1 corners, where d is the dimension of the vertices. For example in 2D, a
            vertex is some point [x, y] and a cell corresponds to a triangle given by
            [vertex_1, vertex_2, vertex_3].
        cell_markers (list[int] | None | TensorType, optional): Some markers of the
            cells. For each cell one marker must be provided. Defaults to None.
        faces (list[list[int]] | None | TensorType, optional): A list of special faces.
            Only needed if some faces should have special markers. Defaults to None.
        face_markers (list[int] | None | TensorType, optional): The markers of the faces.
            Defaults to None.
        marker_labels (dict[str, tuple[int, int]] | None, optional): The above markers
            are based on integers. If names should be used instead, they can be
            mapped with this dictionary. The dictionary has the name of the marker as
            a key and maps it to a tuple of (marker integer, entity dimension).
            Defaults to None.
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        vertices: list[list[float]] | TensorType,
        cells: list[list[int]] | TensorType,
        cell_markers: list[int] | None | TensorType = None,
        faces: list[list[int]] | None | TensorType = None,
        face_markers: list[int] | None | TensorType = None,
        marker_labels: dict[str, tuple[int, int]] | None = None,
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
        self.marker_labels = {} if marker_labels is None else marker_labels

        # Data for normals and volumes that are only computed once
        self.cell_volumes: TensorType | None = None
        self.cell_probability_weights: TensorType | None = None
        self.boundary_normals: TensorType = self.backend.math.empty(
            (0, self.vertices.shape[1])
        )
        self.boundary_normals_at_vertex: TensorType = self.backend.math.empty(
            (0, self.vertices.shape[1])
        )

        self._find_boundary_facets()

    @property
    def vertex_count(self) -> int:
        """Returns the number of vertices in this mesh.

        Returns:
            int: The number of vertices.
        """
        return len(self.vertices)

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
        """Load a mesh from the disk. The mesh should already be a "volume" mesh.
        Currently only simplex meshes are supported.

        Args:
            file_path: The file path to the mesh.
            marker_key (str | None, optional): Markers inside the mesh that should also
                be loaded/included in the mesh object.
            default_cell_tags (int, optional): A default cell tag for all cells that
                dont have any markers. Defaults to -1.
            backend (type[ComputingBackend[TensorType]], optional):
                Defaults to DEFAULT_DL_BACKEND.

        Raises:
            ImportError: Use Meshio to convert and read the mesh, raises an error if
            not installed.

        Returns:
            Mesh: The mesh object containing the mesh from the file.
        """
        try:
            from qewton.geometries.discrete.load_meshes_helper import (
                load_file_with_meshio,
            )  # pylint: disable=import-outside-toplevel # type: ignore
        except ImportError as e:
            raise ImportError(
                "For loading meshes the library meshio is required. Install it via pip"
                "or load the mesh manually to pass the mesh information."
            ) from e

        vertices, cells, cell_markers, faces, face_markers, marker_labels = (
            load_file_with_meshio(
                file_path=file_path,
                marker_key=marker_key,
                default_cell_tags=default_cell_tags,
                backend=backend,
            )
        )
        return cls(
            vertices=vertices,
            cells=cells,
            cell_markers=cell_markers,
            faces=faces,
            face_markers=face_markers,
            marker_labels=marker_labels,
            backend=backend,
        )

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
            marker_labels=self.marker_labels,
            backend=self.backend,
        )

    def get_submesh(self, marker: int | str) -> Mesh:
        if self.cell_markers is None:
            raise ValueError("No markers in mesh available.")
        if isinstance(marker, str):
            assert (
                marker in self.marker_labels
            ), f"The marker {marker} does not appear in {self.marker_labels}"
            marker, marker_dim = self.marker_labels[marker]
            assert (
                marker_dim + 1 == self.cells.shape[1]
            ), "The provided marker belongs to a different cell dimension, \
                please build the correct boundary mesh first."

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
            marker_labels=self.marker_labels,
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
