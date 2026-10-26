import numpy as np

from qewton.visualization.plots.result import ParametricGridResult
from qewton.visualization.plots.spec import SelectorSpec, ColorSpec, AxisSpec
from qewton.visualization.plots.table.base import TablePlot


class TableHeatMap(TablePlot):
    """A heat map of a data table, with two axes and a color scale.

    Args:
        data: A table of data. Anything that can be indexed by column name
            and can be iterated over (e.g. a pandas DataFrame or a dict of
            lists).
        bins (int): The number of bins to use for the heat map. The data will
            be binned into a grid of size `bins x bins`.
        axis_keys (list[str]): The names of the columns to use for the x and y
            axes. The first two names in this list will be used for the x and y
            axes, respectively.
        objective_keys (list[str]): The names of the columns to use for the color
            scale. If more than one name is provided, a SelectorSpec will be used
            to allow the user to choose which column to use for the color scale.
        log_axes (list[str], optional): The names of the columns to use for the
            log scale. If a column name is in this list, the corresponding axis will
            be displayed on a log scale. Defaults to an empty list.
        title (str, optional): The title of the plot. Defaults to None.
        theme (str, optional): The theme of the plot. Defaults to None.
    """

    def __init__(
        self,
        data,
        bins_x: int,
        bins_y: int,
        axis_keys: list[str],
        objective_keys: list[str],
        log_axes: list[str] | None = None,
        title=None,
        theme=None,
    ):
        self.x = AxisSpec(SelectorSpec(candidates=axis_keys, init_index=0))
        self.y = AxisSpec(SelectorSpec(candidates=axis_keys, init_index=1))
        self.objective_keys = objective_keys
        if len(objective_keys) > 1:
            self.color = ColorSpec(variable_or_axes=SelectorSpec(objective_keys))
        else:
            self.color = ColorSpec(variable_or_axes=objective_keys[0])
        super().__init__(columns=data, title=title, theme=theme, controls=[])
        self.log_axes = log_axes if log_axes is not None else []
        self.bins_x = bins_x
        self.bins_y = bins_y
        self.controls = [self.x, self.y]

    def evaluate(self):
        data_x = self.columns[self.x.name].values  # type: ignore
        data_y = self.columns[self.y.name].values  # type: ignore
        color = self.columns[self.color.name].values  # type: ignore

        self.x.log_scale = self.x.name in self.log_axes
        self.y.log_scale = self.y.name in self.log_axes

        counts, x_edges, y_edges = np.histogram2d(
            data_x, data_y, bins=[self.bins_x, self.bins_y]
        )
        sums, _, _ = np.histogram2d(
            data_x, data_y, bins=[x_edges, y_edges], weights=color
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            grid = sums / counts

        grid[counts == 0] = np.nan
        grid = grid.reshape((self.bins_x, self.bins_y, 1))
        # values = np.array(np.meshgrid(x_edges, y_edges, indexing="ij")).T
        return ParametricGridResult(values=grid, x=x_edges, y=y_edges, z=grid)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ParametricHeatmapArtist.create(
            backend_figure, self, row=row, col=col
        )
