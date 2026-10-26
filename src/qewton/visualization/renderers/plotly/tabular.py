"""Plotly artists for scatter and bar plots."""

from plotly import graph_objects as go
import numpy as np

from qewton.visualization.renderers.plotly.common import (
    PlotlyArtist,
    _apply_scale,
    _cycled_color,
)


def _axis_type(values, log_scale):
    # "category" for string/object values, else "log" or "linear".
    if np.asarray(values).dtype.kind in "OSU":  # object/bytes/unicode -> categorical
        return "category"
    return "log" if log_scale else "linear"


class ScatterArtist(PlotlyArtist):
    """Draws markers at (x, y), optionally colored by a third variable.

    Used for :class:`~qewton.visualization.plots.data.samples.ScatterPlot` and
    :class:`~qewton.visualization.plots.table.scatter_table.TableScatter`.
    Without a color field, markers use the theme's cycled palette color.
    Axes are categorical for string values. :meth:`update` also refreshes
    axis titles and types, since a selector can change the plotted x/y
    variables.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        marker = dict(size=plot.theme.marker_size)
        if result.color is not None:
            cmap = plot.theme.resolve_cmap(plot.color.cmap)
            marker.update(color=result.color, colorscale=cmap)
            marker.update(
                _apply_scale(
                    plot.color.scale,
                    backend_figure=backend_figure,
                    row=row,
                    col=col,
                    title=plot.color.math_name,
                )
            )
        else:
            marker.update(color=_cycled_color(plot))

        trace = go.Scatter(
            x=result.x,
            y=result.y,
            mode="markers",
            marker=marker,
            opacity=plot.theme.opacity_default,
        )
        backend_figure.add_trace(trace, row=row, col=col)

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type=_axis_type(result.x, plot.x.log_scale),
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type=_axis_type(result.y, plot.y.log_scale),
            row=row,
            col=col,
        )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.x = result.x
        trace.y = result.y
        if result.color is not None:
            trace.marker.color = result.color
            trace.marker.colorbar.title = dict(text=plot.color.math_name)
            if plot.color.scale is not None:
                trace.marker.cmin, trace.marker.cmax = plot.color.scale.range

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type=_axis_type(result.x, plot.x.log_scale),
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type=_axis_type(result.y, plot.y.log_scale),
            row=row,
            col=col,
        )


class BarArtist(PlotlyArtist):
    """Draws a bar plot as ``go.Bar``.

    Used for :class:`~qewton.visualization.plots.data.samples.BarPlot`.
    Bars use the theme's cycled palette color. Several bar traces on one
    axis are arranged by Plotly's layout ``barmode``.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        trace = go.Bar(
            x=result.x,
            y=result.y,
            name=plot.label or plot.y.name,
            marker=dict(color=_cycled_color(plot)),
            opacity=plot.theme.opacity_default,
        )
        backend_figure.add_trace(trace, row=row, col=col)

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type=_axis_type(result.x, plot.x.log_scale),
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type=_axis_type(result.y, plot.y.log_scale),
            row=row,
            col=col,
        )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.x = result.x
        trace.y = result.y

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type=_axis_type(result.x, plot.x.log_scale),
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type=_axis_type(result.y, plot.y.log_scale),
            row=row,
            col=col,
        )
