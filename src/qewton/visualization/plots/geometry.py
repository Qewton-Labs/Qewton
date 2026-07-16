from qewton.geometries.base import Geometry
from qewton.geometries.discrete.mesh_domain import MeshGeometry
from qewton.visualization.plots.base import Plot


class GeometryPlot(Plot):
    """Visualisiert die Boundary eines Geometry-Objekts wie in einem CAD-
    Programm - rein strukturell, keine Daten/DataConfiguration noetig.
    Aktuell nur 3D; 2D folgt separat (anderer Artist-Zweig)."""

    def __init__(
        self,
        geometry: Geometry,
        max_vertex_distance: float | None = None,
        show_edges: bool = True,
        title=None,
        theme=None,
    ):
        mesh = self._resolve_mesh(geometry, max_vertex_distance)

        if mesh.vertices.shape[1] != 3:
            raise NotImplementedError(
                "GeometryPlot unterstuetzt aktuell nur 3D-Geometrien. 2D folgt separat."
            )

        # Volumetrisches Mesh (z.B. Tetraeder: 4 Vertices/Zelle) -> Boundary
        # extrahieren. Ist es bereits eine Oberflaechen-Triangulierung
        # (3 Vertices/Zelle in 3D), direkt uebernehmen.
        is_volumetric = mesh.cells.shape[1] == mesh.vertices.shape[1] + 1
        self.boundary_mesh = mesh.get_boundary_mesh() if is_volumetric else mesh
        self.show_edges = show_edges

        super().__init__(data=geometry, data_config=None, title=title, theme=theme)

    @staticmethod
    def _resolve_mesh(geometry: Geometry, max_vertex_distance):
        if isinstance(geometry, MeshGeometry):
            return geometry.mesh
        if hasattr(geometry, "create_mesh"):
            return geometry.create_mesh(max_vertex_distance=max_vertex_distance).mesh
        raise ValueError(
            f"{type(geometry).__name__} kann nicht diskretisiert werden - "
            "es fehlt create_mesh()."
        )

    def evaluate(self):
        return self.boundary_mesh.vertices, self.boundary_mesh.cells

    def create_artist(self, backend_figure, renderer):
        return renderer.GeometryArtist.create(backend_figure, self)
