"""TableScatter: a marker per row of a table."""

import numpy as np

from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.result import MarkerResult, Result
from qewton.visualization.plots.table.base import TablePlot
from qewton.visualization.scales import ColorScale


class TableScatter(TablePlot):
    """A marker per row: x and y from two columns, colored by a third.

    Args:
        table, color, color_scale, log_columns, controls, title, label: As
            for :class:`~qewton.visualization.plots.table.base.TablePlot`.
        x, y: Column names, or Selectors of column names.
        jitter: Random offset of every marker, as a fraction of the drawn
            span of each column; separates rows with equal values. The
            offsets are fixed per row.
        size: Marker size, or None for the theme's.

    Attributes:
        jitter (float): The jitter.
        size (float or None): Marker size.
    """

    dim = 2

    def __init__(
        self,
        table,
        *,
        x: str | Selector,
        y: str | Selector,
        color: str | Selector | None = None,
        color_scale: ColorScale | list[ColorScale] | None = None,
        log_columns: tuple[str, ...] | list[str] = (),
        jitter: float = 0.0,
        size: float | None = None,
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
        self._positions = {"x": x, "y": y}
        for reference in self._positions.values():
            self._check_reference(reference)
        self.jitter = jitter
        self.size = size
        self._offsets = np.random.default_rng(0).normal(size=(self.n_rows, 2))

    @property
    def positions(self) -> dict:
        return dict(self._positions)

    def build(self, rows: np.ndarray) -> Result:
        """See :meth:`~qewton.visualization.plots.table.base.TablePlot.build`."""
        columns = []
        for k, reference in enumerate(self._positions.values()):
            name = self.current(reference)
            values = self.drawn_values(name, rows)
            if self.jitter > 0:
                drawn = self.drawn_values(name, np.ones(self.n_rows, dtype=bool))
                values = values + self._offsets[rows, k] * self.jitter * np.ptp(drawn)
            columns.append(values)
        return MarkerResult(
            positions=np.stack(columns, axis=1),
            values=self.color_values(rows),
            size=self.size,
        )
