from typing import Any
from pathlib import Path
import numpy as np

from dolfinx import mesh as fx_mesh
from ufl import dx, ds, dS
from mpi4py import MPI
from dolfinx.io import XDMFFile
from basix import ElementFamily
from basix.ufl import element

from qewton.backends.fem.mesh import MeshBackend


class FenicsXMeshBackend(MeshBackend):
    """A Backend that implements mesh-related operations for FEniCSx."""

    # region Mesh creation and refinement
    @staticmethod
    def load_mesh(path: str | Path, mesh_name: str | None = None) -> Any:
        path = Path(path)
        assert path.suffix == ".xdmf" and mesh_name is not None
        with XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
            mesh = xdmf.read_mesh(name=mesh_name)
        return mesh

    @staticmethod
    def create_mesh(domain: Any, resolution: int) -> tuple[Any, Any, Any]:
        from qewton.geometries.continuous.base import ContinuousGeometry
        from qewton.geometries.discrete.mesh_geometry import MeshGeometry

        if isinstance(domain, ContinuousGeometry):
            domain = domain.create_mesh(max_vertex_distance=resolution)
        assert isinstance(
            domain, MeshGeometry
        ), "Required a MeshGeometry to build the mesh"

        # build the basic mesh:
        points = domain.mesh.vertices
        cells = domain.mesh.cells
        point_dim = points.shape[1]
        mesh_element = element(ElementFamily.P, "triangle", 1, shape=(point_dim,))

        fenics_mesh = fx_mesh.create_mesh(MPI.COMM_WORLD, cells, mesh_element, points)  # type: ignore

        cell_markers = (
            domain.mesh.cell_markers if domain.mesh.cell_markers is not None else []
        )
        # Build cell tags
        cell_indexing = np.arange(len(cell_markers), dtype=np.int32)
        cell_tags = fx_mesh.meshtags(
            fenics_mesh, fenics_mesh.topology.dim, cell_indexing, np.array(cell_markers)
        )
        # Build facet tags if available
        facet_markers = (
            domain.mesh.face_markers if domain.mesh.face_markers is not None else []
        )
        facet_indexing = np.arange(len(facet_markers), dtype=np.int32)
        facet_tags = fx_mesh.meshtags(
            fenics_mesh,
            fenics_mesh.topology.dim - 1,
            facet_indexing,
            np.array(facet_markers),
        )

        return fenics_mesh, cell_tags, facet_tags

    @staticmethod
    def refine_mesh(mesh: Any) -> Any:
        return fx_mesh.refine(mesh)

    @staticmethod
    def coarsen_mesh(mesh: Any) -> Any:
        raise NotImplementedError("Mesh coarsening is not supported by DOLFINx.")

    @staticmethod
    def mark_mesh(
        mesh: Any,
        marker_condition: Any,
        marker: int,
        dim: int,
        marker_tag_function: Any | None = None,
        boundary_only: bool = False,
    ) -> Any:
        if boundary_only:
            entities = fx_mesh.locate_entities_boundary(mesh, dim, marker_condition)
        else:
            entities = fx_mesh.locate_entities(mesh, dim, marker_condition)

        if marker_tag_function is None:
            indices = entities
            values = np.full(len(entities), marker, dtype=np.int32)
        else:
            # Merge existing markers with new markers
            indices = np.concatenate([marker_tag_function.indices, entities])
            values = np.concatenate(
                [
                    marker_tag_function.values,
                    np.full(len(entities), marker, dtype=np.int32),
                ]
            )

            # Sort by entity index
            order = np.argsort(indices)
            indices = indices[order]
            values = values[order]

            # Remove duplicate indices (new markers take priority)
            unique_indices, inverse = np.unique(indices, return_inverse=True)
            unique_values = np.zeros(len(unique_indices), dtype=np.int32)
            unique_values[inverse] = values

            indices = unique_indices
            values = unique_values

        return fx_mesh.meshtags(mesh, dim, indices, values)

    @staticmethod
    def load_marker(
        mesh: Any, path: str | Path, marker_dim: int, marker_name: str | None = None
    ) -> Any:
        path = Path(path)
        assert path.suffix == ".xdmf" and marker_name is not None
        # Add connectivity for the marker dimension if it is lower than the mesh dimension
        if marker_dim < mesh.topology.dim:
            mesh.topology.create_connectivity(mesh.topology.dim, marker_dim)
        with XDMFFile(MPI.COMM_WORLD, path, "r") as xdmf:
            marker = xdmf.read_meshtags(mesh, name=marker_name)
        return marker

    @staticmethod
    def get_mesh_coordinates(mesh: Any) -> Any:
        return mesh.geometry.x

    @staticmethod
    def get_mesh_elements(mesh: Any) -> Any:
        return mesh.topology.connectivity(0, mesh.topology.dim)

    @staticmethod
    def get_subdomain(mesh: Any, marker_fn: Any, marker: int, dim: int) -> Any:
        cell_indices = marker_fn.find(marker)
        fx_mesh.create_submesh(mesh, dim, cell_indices)

    # endregion

    # region Measure
    @staticmethod
    def volume_measure(mesh: Any, subdomain_marker: Any = None) -> Any:
        return dx(subdomain_data=subdomain_marker) if subdomain_marker is not None else dx

    @staticmethod
    def boundary_measure(mesh: Any, subdomain_marker: Any = None) -> Any:
        return ds(subdomain_data=subdomain_marker) if subdomain_marker is not None else ds

    @staticmethod
    def interface_measure(mesh: Any, subdomain_marker: Any = None) -> Any:
        return dS(subdomain_data=subdomain_marker) if subdomain_marker is not None else dS

    # endregion

    # region Boundary information
    @staticmethod
    def get_mesh_boundaries(mesh: Any) -> np.ndarray:
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_connectivity(fdim, tdim)

        return fx_mesh.exterior_facet_indices(mesh.topology)

    @staticmethod
    def get_boundary_facets(mesh: Any, marker_tags: Any, marker: int) -> np.ndarray:
        return marker_tags.find(marker)

    # endregion
