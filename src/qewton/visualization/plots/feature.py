from qewton.config.axes import Axes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.base import StructuredGridPlot
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec


class ImagePlot(StructuredGridPlot):
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, controls=controls)

    def create_artist(self, backend_figure, renderer):
        return renderer.ImageArtist.create(backend_figure, self)


class SurfacePlot(StructuredGridPlot):
    def create_artist(self, backend_figure, renderer):
        return renderer.SurfaceArtist.create(backend_figure, self)


class HeatmapPlot(StructuredGridPlot):
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, color=color, controls=controls)

    def create_artist(self, backend_figure, renderer):
        return renderer.HeatmapArtist.create(backend_figure, self)
