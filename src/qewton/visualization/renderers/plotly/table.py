"""Plotly artist for parallel-coordinates plots."""

from plotly import graph_objects as go
import numpy as np

from qewton.visualization.plots.result import TableResult
from qewton.visualization.renderers.plotly.common import PlotlyArtist, _apply_scale


def _log_tick_marks(log_values):
    """Powers-of-ten tick positions (in log10 units) and labels for log10 data."""
    lo, hi = log_values.min(), log_values.max()
    dist = int(np.ceil(hi) - np.floor(lo))
    dist = dist + 7 if dist < 2 else dist + 5
    tick_vals = np.linspace(np.floor(lo), np.ceil(hi), num=dist)
    tick_text = [f"{10**v:.0e}" for v in tick_vals]
    return tick_vals, tick_text


def _set_log_colorbar_ticks(scale_kwargs, log_color):
    """Labels a log-scaled color colorbar with powers of ten rather than the
    log10 exponents the colors are stored as."""
    tick_vals, tick_text = _log_tick_marks(log_color)
    colorbar = dict(scale_kwargs.get("colorbar", {}))
    colorbar.update(tickvals=tick_vals, ticktext=tick_text)
    scale_kwargs["colorbar"] = colorbar


class ParallelCoordinatesArtist(PlotlyArtist):
    """Draws a parallel-coordinates plot as ``go.Parcoords``.

    Used for
    :class:`~qewton.visualization.plots.table.parallel_coordinates.ParallelCoordinatesPlot`.
    One polyline per table row and one axis per column, optionally colored.
    Categorical columns get their labels as ticks; log-scale columns are
    drawn as ``log10`` values with ticks labeled in the original scale.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result: TableResult = plot.evaluate()
        line = dict()
        if result.color is not None:
            cmap = plot.theme.resolve_cmap(plot.color.cmap)
            line.update(color=result.color, colorscale=cmap)
            scale_kwargs = _apply_scale(
                plot.color.scale, backend_figure=backend_figure, row=row, col=col
            )
            if result.color_log_scale:
                _set_log_colorbar_ticks(scale_kwargs, result.color)
            line.update(scale_kwargs)
        trace = go.Parcoords(dimensions=cls._dimensions(plot, result), line=line)
        backend_figure.add_trace(trace, row=row, col=col)
        return cls(len(backend_figure.data) - 1)

    @staticmethod
    def _dimensions(plot, result: TableResult):
        dimensions = []
        for key, column in result.columns.items():
            dim = dict(label=plot.labels.get(key, key), values=column.values)
            if column.labels is not None:
                dim.update(
                    tickvals=list(range(len(column.labels))), ticktext=column.labels
                )
            if column.log_scale:
                log_values = np.log10(column.values)
                tick_vals, tick_text = _log_tick_marks(log_values)
                dim.update(values=log_values, tickvals=tick_vals, ticktext=tick_text)
            dimensions.append(dim)
        return dimensions

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.dimensions = self._dimensions(plot, result)
        if result.color is not None:
            trace.line.color = result.color
            if plot.color.scale is not None:
                trace.line.cmin, trace.line.cmax = plot.color.scale.range
            if result.color_log_scale:
                tick_vals, tick_text = _log_tick_marks(result.color)
                trace.line.colorbar.tickvals = tick_vals
                trace.line.colorbar.ticktext = tick_text
