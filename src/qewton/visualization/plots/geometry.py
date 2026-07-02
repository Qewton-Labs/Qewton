from qewton.geometries.base import Geometry
from qewton.visualization.plots.base import Plot


class GeometryPlot(Plot):
    def __init__(
        self,
        geometry: Geometry,
        title=None,
    ) -> None:
        super().__init__(geometry, None, title)

    def evaluate(self):
        raise NotImplementedError("For now, Geometries can not be parameter-dependent")

    def create_artist(self, backend_figure, renderer):
        return renderer.GeometryArtist.create(backend_figure, self)
