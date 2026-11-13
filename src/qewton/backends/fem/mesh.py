from abc import abstractmethod
from typing import Any
from pathlib import Path

from qewton.backends.base import Backend, TensorType


class MeshBackend(Backend[TensorType]):
    """A Backend that implements mesh-related operations for the finite element method."""

    # region Mesh creation and refinement
    @staticmethod
    @abstractmethod
    def load_mesh(path: str | Path, mesh_name: str | None = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def create_mesh(domain: Any, resolution: int) -> tuple[Any, Any, Any]:
        """Create a mesh from the given domain and resolution.

        Args:
            domain (qewton.Geometry) : The geometric domain to create the mesh from.
            resolution (int) : The resolution parameter for mesh creation.

        Returns:
            tuple[Any, Any, Any] : A tuple containing the created mesh as well as cell and
                facet mesh markers.
        """
        pass

    @staticmethod
    @abstractmethod
    def refine_mesh(mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def coarsen_mesh(mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def mark_mesh(
        mesh: Any,
        marker_condition: Any,
        marker: int,
        dim: int,
        marker_tag_function: Any,
        boundary_only: bool = False,
    ) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def load_marker(
        mesh: Any, path: str | Path, marker_dim: int, marker_name: str | None = None
    ) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_mesh_coordinates(mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_mesh_elements(mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_subdomain(mesh: Any, marker_fn: Any, marker: int, dim: int) -> Any:
        pass

    # endregion

    # region Measure
    @staticmethod
    @abstractmethod
    def volume_measure(mesh: Any, subdomain_marker: Any = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def boundary_measure(mesh: Any, subdomain_marker: Any = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def interface_measure(mesh: Any, subdomain_marker: Any = None) -> Any:
        pass

    # endregion

    # region Boundary information
    @staticmethod
    @abstractmethod
    def get_mesh_boundaries(mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_boundary_facets(mesh: Any, marker_tags: Any, marker: int) -> Any:
        pass

    # endregion
