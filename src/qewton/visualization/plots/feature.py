from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.config import PlotConfig


class ImagePlot(Plot):
    def __init__(
        self, data, plot_config: PlotConfig, title=None, theme=None
    ) -> None:
        super().__init__(data, plot_config, title, theme)

    def create_artist(self, backend_figure, renderer):
        return renderer.ImageArtist.create(backend_figure, self)


class SurfacePlot(Plot):
    def create_artist(self, backend_figure, renderer):
        return renderer.SurfaceArtist.create(backend_figure, self)


class HeatmapPlot(Plot):
    def create_artist(self, backend_figure, renderer):
        return renderer.HeatmapArtist.create(backend_figure, self)
