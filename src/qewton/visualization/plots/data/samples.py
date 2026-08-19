import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.data.curve import LinePlot
from qewton.visualization.plots.result import ScatterResult
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec


class ScatterPlot(DataPlot):
    """Points at (x, y), one per sample, optionally colored by a third
    variable. A third family alongside grids (StructuredGridPlot) and meshes
    (MeshPlot): no GeometryAxes at all, no cells, no coordinates - a good test
    that Plot really works without a geometry.

    Unlike LinePlot, x and y are both VALUE roles (channel slices via
    get_variable_slice), not one structural domain axis and one value -
    there's no single "domain" a scatter plot is indexed along, just an
    implicit samples axis that x/y/color are all extracted over and
    flattened. Whatever's left after apply_controls() and slicing out x/y/
    color's own channels is that samples axis (or axes, flattened together).
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable,
        y: AxisSpec | Variable,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)
        self.x = x if isinstance(x, AxisSpec) else AxisSpec(x)
        self.y = y if isinstance(y, AxisSpec) else AxisSpec(y)
        self.require_scalar(self.x, "x")
        self.require_scalar(self.y, "y")

        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color is not None
            else None
        )
        if self.color is not None:
            self.require_scalar(self.color, "color")

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()

        x_slc = self.data_config.get_variable_slice(self.x.variable_or_axes)
        y_slc = self.data_config.get_variable_slice(self.y.variable_or_axes)
        x_values = np.asarray(data[slice_map(x_slc)]).reshape(-1)
        y_values = np.asarray(data[slice_map(y_slc)]).reshape(-1)
        if x_values.shape != y_values.shape:
            raise ValueError(
                f"{self.x.name} yields {x_values.size} values but {self.y.name} "
                f"yields {y_values.size} - unresolved batch dimensions? Add a "
                "SliderSpec or FixedSpec for them."
            )

        color_values = None
        if self.color is not None:
            c_slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color_values = np.asarray(data[slice_map(c_slc)]).reshape(-1)
            if color_values.shape != x_values.shape:
                raise ValueError(
                    f"{self.color.name} yields {color_values.size} values but "
                    f"{self.x.name}/{self.y.name} yield {x_values.size} - they "
                    "must come from the same samples axis."
                )

        return ScatterResult(x=x_values, y=y_values, color=color_values)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ScatterArtist.create(backend_figure, self, row=row, col=col)


class BarPlot(LinePlot):
    """Bars at (x, height) - the exact same data shape as LinePlot (one value
    over one structural domain axis), just drawn as bars instead of a line.
    No new evaluate() needed: HeatmapPlot/SurfacePlot already establish the
    precedent of subclassing a shared base and overriding only
    create_artist() when two plot types agree on what data they need and
    differ only in the mark.

    The plot side of histograms: takes already-binned data (bin positions +
    counts), not raw samples - turning samples into bins is node work (see
    the visualization plan, roadmap item 4's "Statistical preprocessing"
    note). A future `HistogramPlot.from_samples(data, config, bins=20)`
    convenience constructor would build that node and hand its output to
    BarPlot, once the node exists; BarPlot itself needs no changes for that.
    """

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.BarArtist.create(backend_figure, self, row=row, col=col)
