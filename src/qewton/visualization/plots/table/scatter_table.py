import numpy as np

from qewton.visualization.plots.result import ScatterResult
from qewton.visualization.plots.spec import VariableSpec, ColorSpec, AxisSpec
from qewton.visualization.plots.table.base import TablePlot


class TableScatter(TablePlot):
    """."""

    def __init__(
        self,
        data,
        axis_keys: list[str],
        objective_keys: list[str],
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
        self.controls = [self.x, self.y]

    def evaluate(self):
        print(self.x.name, self.y.name, self.color.name)
        data_x = self.columns[self.x.name]  # type: ignore
        data_y = self.columns[self.y.name]  # type: ignore
        if data_x.labels is not None:
            data_x = np.asarray([data_x.labels[i] for i in data_x.values])
        else:
            data_x = np.asarray(data_x.values)
        if data_y.labels is not None:
            data_y = np.asarray([data_y.labels[i] for i in data_y.values])
        else:
            data_y = np.asarray(data_y.values)

        color = self.columns[self.color.name].values  # type: ignore
        return ScatterResult(x=data_x, y=data_y, color=color)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ScatterArtist.create(backend_figure, self, row=row, col=col)
