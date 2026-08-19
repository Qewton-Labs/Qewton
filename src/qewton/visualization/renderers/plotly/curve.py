from plotly import graph_objects as go

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import PlotlyArtist


class LineArtist(PlotlyArtist):
    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        trace = go.Scatter(
            x=result.x,
            y=result.y,
            mode="lines",
            name=plot.title or plot.y.name,
            line=dict(width=plot.theme.line_width),
        )
        backend_figure.add_trace(trace, row=row, col=col)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        backend_figure.update_xaxes(
            title=plot.x.name,
            type="log" if plot.x.log_scale else "linear",
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.name,
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

    def remove(self, backend_figure):
        pass


class PathArtist(PlotlyArtist):
    """A trajectory or streamline - go.Scatter for 2D, go.Scatter3d for 3D.
    The two genuinely differ (different Plotly trace types), unlike e.g.
    LineArtist which is always a 2D chart regardless of what it plots."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        positions = plot.evaluate().positions
        dim = positions.shape[1]
        # PathPlot has no GeometryAxes to name axes from (a trajectory isn't
        # tied to a geometry) - its own VectorSpec.variable_or_axes is the
        # closest thing to an axis-naming source, e.g. X*Y*Z -> "x"/"y"/"z".
        names = axis_names_from_variable(plot.position.variable_or_axes, dim)

        if dim == 2:
            trace = go.Scatter(
                x=positions[:, 0], y=positions[:, 1], mode="lines",
                line=dict(width=plot.theme.line_width),
            )
        else:
            trace = go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="lines",
                line=dict(width=plot.theme.line_width),
            )

        backend_figure.add_trace(trace, row=row, col=col)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        if dim == 2:
            backend_figure.update_xaxes(title=names[0], row=row, col=col)
            backend_figure.update_yaxes(title=names[1], row=row, col=col)
        else:
            backend_figure.update_scenes(
                row=row, col=col,
                xaxis=dict(title=names[0]), yaxis=dict(title=names[1]), zaxis=dict(title=names[2]),
            )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot):
        positions = plot.evaluate().positions
        trace = backend_figure.data[self.figure_idx]
        trace.x = positions[:, 0]
        trace.y = positions[:, 1]
        if positions.shape[1] == 3:
            trace.z = positions[:, 2]

    def remove(self, backend_figure):
        pass
