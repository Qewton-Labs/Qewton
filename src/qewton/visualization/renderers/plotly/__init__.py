from plotly import graph_objects as go

from qewton.visualization.renderers.base import Renderer
from qewton.visualization.renderers.plotly.curve import LineArtist, PathArtist
from qewton.visualization.renderers.plotly.geometry import GeometryArtist, GeometryArtist2D
from qewton.visualization.renderers.plotly.grid import (
    HeatmapArtist,
    ImageArtist,
    ParametricSurfaceArtist,
    SurfaceArtist,
)
from qewton.visualization.renderers.plotly.mesh import FilledMeshArtist, SurfaceMeshArtist


class PlotlyRenderer(Renderer):
    """Aggregation namespace: renderer.SurfaceMeshArtist.create(...) etc.

    The artists themselves live at module level (geometry.py, grid.py, mesh.py,
    vector.py) - this class only collects them so plots can reach them off the
    renderer instance without knowing which module each one lives in.
    """

    ImageArtist = ImageArtist
    HeatmapArtist = HeatmapArtist
    SurfaceArtist = SurfaceArtist
    ParametricSurfaceArtist = ParametricSurfaceArtist
    SurfaceMeshArtist = SurfaceMeshArtist
    FilledMeshArtist = FilledMeshArtist
    GeometryArtist = GeometryArtist
    GeometryArtist2D = GeometryArtist2D
    LineArtist = LineArtist
    PathArtist = PathArtist

    @staticmethod
    def setup():
        fig = go.Figure()
        fig.update_layout(uirevision=True)
        return fig

    @staticmethod
    def show(backend_figure):
        backend_figure.show()

    @staticmethod
    def save_html(backend_figure, path):
        backend_figure.write_html(path)
