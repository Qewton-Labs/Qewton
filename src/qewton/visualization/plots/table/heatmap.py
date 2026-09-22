import numpy as np

from qewton.visualization.plots.result import GridResult
from qewton.visualization.plots.spec import VariableSpec, ColorSpec, AxisSpec
from qewton.visualization.plots.table.base import TablePlot


class TableHeatMap(TablePlot):
    """ """

    def __init__(
        self,
        data,
        bins: int,
        axis_keys: list[str],
        objective_keys: list[str],
        log_axes: list[str] = [],
        title=None,
        theme=None,
    ):
        self.x = AxisSpec(VariableSpec(candidates=axis_keys, init_index=0))
        self.y = AxisSpec(VariableSpec(candidates=axis_keys, init_index=1))
        self.objective_keys = objective_keys
        if len(objective_keys) > 1:
            self.color = ColorSpec(variable_or_axes=VariableSpec(objective_keys))
        else:
            self.color = ColorSpec(variable_or_axes=objective_keys[0])
        super().__init__(columns=data, title=title, theme=theme, controls=[])
        self.log_axes = log_axes
        self.bins = bins
        self.controls = [self.x, self.y]

    def evaluate(self):
        data_x = self.columns[self.x.name].values  # type: ignore
        data_y = self.columns[self.y.name].values  # type: ignore
        color = self.columns[self.color.name].values  # type: ignore

        self.x.log_scale = self.x.name in self.log_axes
        self.y.log_scale = self.y.name in self.log_axes

        counts, x_edges, y_edges = np.histogram2d(
            data_x, data_y, bins=[self.bins, self.bins]
        )
        sums, _, _ = np.histogram2d(
            data_x, data_y, bins=[x_edges, y_edges], weights=color
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            grid = sums / counts

        grid[counts == 0] = np.nan
        grid = grid.reshape((self.bins, self.bins, 1))
        # values = np.array(np.meshgrid(x_edges, y_edges, indexing="ij")).T
        return GridResult(values=grid, color=grid)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.HeatmapArtist.create(backend_figure, self, row=row, col=col)
