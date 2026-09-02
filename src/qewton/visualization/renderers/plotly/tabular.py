from plotly import graph_objects as go

from qewton.visualization.renderers.plotly.common import PlotlyArtist, _apply_scale, _cycled_color


class ScatterArtist(PlotlyArtist):
    """Points at (x, y), optionally colored by a third variable - the "no
    geometry" counterpart to HeatmapArtist/SurfaceMeshArtist: color comes
    straight from ScatterResult, no grid or mesh structure involved."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        marker = dict(size=plot.theme.marker_size)
        if result.color is not None:
            cmap = plot.color.cmap or plot.theme.default_cmap
            marker.update(color=result.color, colorscale=cmap)
            marker.update(
                _apply_scale(plot.color.scale, backend_figure=backend_figure, row=row, col=col)
            )
        else:
            # No data-driven ColorSpec - fall back to the theme's cycled
            # palette so multiple ScatterPlots overlaid in one Figure read
            # as distinct traces, not identical unthemed markers.
            marker.update(color=_cycled_color(plot))

        trace = go.Scatter(
            x=result.x, y=result.y, mode="markers", marker=marker,
            opacity=plot.theme.opacity_default,
        )
        backend_figure.add_trace(trace, row=row, col=col)

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type="log" if plot.x.log_scale else "linear",
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type="log" if plot.y.log_scale else "linear",
            row=row,
            col=col,
        )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.x = result.x
        trace.y = result.y
        if result.color is not None:
            trace.marker.color = result.color
            if plot.color.scale is not None:
                trace.marker.cmin, trace.marker.cmax = plot.color.scale.range


class BarArtist(PlotlyArtist):
    """Bars at (x, height) - go.Bar over the same CurveResult LineArtist
    draws as a line. Plotly stacks/groups multiple go.Bar traces sharing an
    axis automatically (barmode on the layout), so nothing extra is needed
    here for multiple BarPlots in one Figure."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        trace = go.Bar(
            x=result.x, y=result.y, name=plot.label or plot.y.name,
            marker=dict(color=_cycled_color(plot)),
            opacity=plot.theme.opacity_default,
        )
        backend_figure.add_trace(trace, row=row, col=col)

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type="log" if plot.x.log_scale else "linear",
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type="log" if plot.y.log_scale else "linear",
            row=row,
            col=col,
        )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.x = result.x
        trace.y = result.y
