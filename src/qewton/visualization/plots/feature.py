from qewton.visualization.plots.base import Plot


class ImagePlot(Plot):
    def create_artist(self, backend_figure, renderer):
        return renderer.ImageArtist.create(backend_figure, self)


class SurfacePlot(Plot):
    def create_artist(self, backend_figure, renderer):
        return renderer.SurfaceArtist.create(backend_figure, self)
