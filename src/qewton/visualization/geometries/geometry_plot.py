from qewton.geometries.base import Geometry
from qewton.visualization.plot import Plot


class GeometryPlot(Plot):
    def __init__(self, geometry: Geometry, theme=None, title=None) -> None:
        super().__init__(geometry, theme, title)

    @property
    def geometry(self):
        return self.data
