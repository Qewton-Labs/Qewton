from qewton.geometries.base import Geometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.visualization.plots.base import Plot


class GeometryPlot(Plot):
    """Visualizes a 2d or 3d geometry object. If the geometry is continuous, it is
    discretized into a mesh for visualization."""

    def __init__(
        self,
        geometry: Geometry,
        max_vertex_distance: float | None = None,
        show_edges: bool = True,
        title=None,
        theme=None,
    ):
        mesh = self._resolve_mesh(geometry, max_vertex_distance)
        self.dim = mesh.vertices.shape[1]
        if self.dim not in [2, 3]:
            raise NotImplementedError("GeometryPlot supports only 2D and 3D geometries.")

        is_volumetric = mesh.cells.shape[1] == mesh.vertices.shape[1] + 1
        self.boundary_mesh = mesh.get_boundary_mesh() if is_volumetric else mesh
        self.interior_mesh = mesh if is_volumetric else None
        self.show_edges = show_edges

        super().__init__(data=geometry, data_config=None, title=title, theme=theme)

    @staticmethod
    def _resolve_mesh(geometry: Geometry, max_vertex_distance):
        if isinstance(geometry, MeshGeometry):
            return geometry.mesh
        if hasattr(geometry, "create_mesh"):
            return geometry.create_mesh(max_vertex_distance=max_vertex_distance).mesh
        raise ValueError(
            f"{type(geometry).__name__} can not be discretized for visualization. \
                Please provide a MeshGeometry or a Geometry with a create_mesh() \
                    method."
        )

    def evaluate(self):
        return self.boundary_mesh.vertices, self.boundary_mesh.cells

    def create_artist(self, backend_figure, renderer):
        return (
            renderer.GeometryArtist2D.create(backend_figure, self)
            if self.dim == 2
            else renderer.GeometryArtist3d.create(backend_figure, self)
        )
