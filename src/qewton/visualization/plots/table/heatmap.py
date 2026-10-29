"""TableHeatMap: the mean of a column over a raster of two others."""

import numpy as np

from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.base import PointEncoding
from qewton.visualization.plots.result import HeatmapResult, Result
from qewton.visualization.plots.table.base import TablePlot
from qewton.visualization.scales import ColorScale


class TableHeatMap(TablePlot):
    """The mean of the color column over the rows in each rectangle of a
    raster spanned by two columns; rectangles without rows stay empty. A
    log column is rastered in log space.

    Args:
        table, color_scale, log_columns, controls, title, label: As for
            :class:`~qewton.visualization.plots.table.base.TablePlot`.
        x, y: Numeric column names, or Selectors of them.
        color: The averaged column, or a Selector of columns.
        bins_x, bins_y: Number of rectangles along x and y.

    Raises:
        ValueError: As for TablePlot, and if a position column is
            categorical.

    Attributes:
        bins_x, bins_y (int): Number of rectangles along x and y.
    """

    dim = 2
    point_encoding = PointEncoding.COLOR

    def __init__(
        self,
        table,
        *,
        x: str | Selector,
        y: str | Selector,
        color: str | Selector,
        bins_x: int = 10,
        bins_y: int = 10,
        color_scale: ColorScale | list[ColorScale] | None = None,
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
        self._positions = {"x": x, "y": y}
        for reference in self._positions.values():
            self._check_reference(reference)
            names = reference.candidates if isinstance(reference, Selector) else [reference]
            categorical = [n for n in names if self.columns[n].categories is not None]
            if categorical:
                raise ValueError(f"TableHeatMap rasters numeric columns only, not {categorical}.")
        self.bins_x = bins_x
        self.bins_y = bins_y

    @property
    def positions(self) -> dict:
        return dict(self._positions)

    def build(self, rows: np.ndarray) -> Result:
        """See :meth:`~qewton.visualization.plots.table.base.TablePlot.build`."""
        every = np.ones(self.n_rows, dtype=bool)
        x_name, y_name = (self.current(r) for r in self._positions.values())
        x_edges = _edges(self.drawn_values(x_name, every), self.bins_x)
        y_edges = _edges(self.drawn_values(y_name, every), self.bins_y)
        x, y = self.drawn_values(x_name, rows), self.drawn_values(y_name, rows)
        counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        sums, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=self.color_values(rows))
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        return HeatmapResult(
            x=(x_edges[:-1] + x_edges[1:]) / 2,
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            values=means.T,
        )


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    # Raster edges over the span of every row, so they stay fixed between
    # the states of the controls.
    lo, hi = float(values.min()), float(values.max())
    return np.linspace(lo, hi if hi > lo else lo + 1.0, bins + 1)
