from abc import abstractmethod
from typing import Any
from qewton.backends.base import Backend, TensorType


class MeshBackend(Backend[TensorType]):
    """A Backend that implements mesh-related operations for the finite element method."""

    # region Mesh creation and refinement
    @staticmethod
    @abstractmethod
    def create_mesh(domain: Any, resolution: Any) -> Any:
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
    def mark_mesh(mesh: Any, marker_condition: Any, marker: int) -> Any:
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
    def get_mesh_boundaries(mesh: Any) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_subdomain(mesh: Any, marker: int) -> Any:
        pass

    # endregion

    # region Measure
    @staticmethod
    @abstractmethod
    def volume_measure(mesh: Any, subdomain: Any = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def boundary_measure(mesh: Any, subdomain: Any = None) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def interface_measure(mesh: Any, subdomain: Any = None) -> Any:
        pass

    # endregion
