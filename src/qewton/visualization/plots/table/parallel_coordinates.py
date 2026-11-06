"""ParallelCoordinatesPlot: one axis per column, one line per row."""

import numpy as np

from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.result import (
    ParallelCoordinatesResult,
    ParallelDimension,
    Result,
)
from qewton.visualization.plots.table.base import TablePlot
from qewton.visualization.scales import ColorScale


class ParallelCoordinatesPlot(TablePlot):
    """One vertical axis per column, one line per row, e.g. the parameters
    and metrics of the runs of a hyperparameter tuning. The plot lays out
    its own axes (no position axes).

    Args:
        table, color, color_scale, log_columns, controls, title, label: As
            for :class:`~qewton.visualization.plots.table.base.TablePlot`.
        columns: The drawn columns, in order.
        column_titles: Display name per column; the column name if
            missing.

    Attributes:
        column_names (list of str): The drawn columns.
        column_titles (dict): Display name per column.
    """

    dim = 0

    def __init__(
        self,
        table,
        *,
        columns: list[str],
        color: str | Selector | None = None,
        color_scale: ColorScale | list[ColorScale] | None = None,
        column_titles: dict[str, str] | None = None,
        log_columns: tuple[str, ...] | list[str] = (),
        controls: tuple[ControlAxis, ...] | list[ControlAxis] = (),
        title: str | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(
            table,
            color=color,
            color_scale=color_scale,
            log_columns=log_columns,
            controls=controls,
            title=title,
            label=label,
        )
        self._check_names(columns)
        self.column_names = list(columns)
        self.column_titles = dict(column_titles or {})

    @property
    def positions(self) -> dict:
        return {}

    def build(self, rows: np.ndarray) -> Result:
        """See :meth:`~qewton.visualization.plots.table.base.TablePlot.build`."""
        every = np.ones(self.n_rows, dtype=bool)
        dimensions = []
        for name in self.column_names:
            span = self.drawn_values(name, every)
            dimensions.append(
                ParallelDimension(
                    title=self.column_titles.get(name, name),
                    values=self.drawn_values(name, rows),
                    range=(float(span.min()), float(span.max())),
                    ticks=self.column_ticks(name),
                )
            )
        return ParallelCoordinatesResult(dimensions=dimensions, values=self.color_values(rows))
