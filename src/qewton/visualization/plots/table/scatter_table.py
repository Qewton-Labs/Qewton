import numpy as np

from qewton.visualization.plots.result import ScatterResult
from qewton.visualization.plots.spec import SelectorSpec, ColorSpec, AxisSpec
from qewton.visualization.plots.table.base import TablePlot


class TableScatter(TablePlot):
    """A scatter plot whose x/y values come from two named columns of a table,
    and whose color comes from a third column. The table may be filtered by
    controls, which select rows by column value. The x/y axes are chosen by
    SelectorSpec controls, which are also part of the table's controls list.

    Args:
        data: a mapping of column names to arrays, or a pandas DataFrame, or
            the tuner's own result object - anything whose `.items()` yields
            (name, array-like) pairs.
        axis_keys (list[str]): names of columns to choose from for the x/y
            axes of the plot.
        objective_keys (list[str]): names of columns to choose from for the
            color of the plot.
        log_axes (list[str], optional): names of columns to be plotted on a
            log scale. Defaults to None.
        jitter (float, optional): amount of jitter to add to the x/y values.
            Defaults to 0.0.
        title (str, optional): title of the plot. Defaults to None.
        theme (dict, optional): theme of the plot. Defaults to None.
    """

    def __init__(
        self,
        data,
        axis_keys: list[str],
        objective_keys: list[str],
        log_axes: list[str] | None = None,
        jitter: float = 0.0,
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
        self.controls = [self.x, self.y]
        self.jitter = jitter

    def evaluate(self):
        data_x = self._transform_data(self.columns[self.x.name])  # type: ignore
        data_y = self._transform_data(self.columns[self.y.name])  # type: ignore

        self.x.log_scale = self.x.name in self.log_axes
        self.y.log_scale = self.y.name in self.log_axes

        color = self.columns[self.color.name].values  # type: ignore
        return ScatterResult(x=data_x, y=data_y, color=color)

    def _transform_data(self, data):
        if data.labels is not None:
            return np.asarray([data.labels[i] for i in data.values])
        if self.jitter > 0.0:
            vals = np.asarray(data.values)
            span = np.nanmax(vals) - np.nanmin(vals)
            return np.asarray(data.values) + np.random.normal(
                loc=0.0, scale=span * self.jitter, size=len(data)
            )
        return np.asarray(data.values)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ScatterArtist.create(backend_figure, self, row=row, col=col)
