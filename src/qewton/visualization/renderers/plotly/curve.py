from plotly import graph_objects as go

from qewton.visualization.renderers.plotly.common import PlotlyArtist


class LineArtist(PlotlyArtist):
    @classmethod
    def create(cls, backend_figure, plot):
        result = plot.evaluate()
        trace = go.Scatter(
            x=result.x, y=result.y, mode="lines", name=plot.title or plot.y.name
        )
        backend_figure.add_trace(trace)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        backend_figure.update_xaxes(
            title=plot.x.name,
            type="log" if plot.x.log_scale else "linear",
        )
        backend_figure.update_yaxes(
            title=plot.y.name,
            type="log" if plot.y.log_scale else "linear",
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
    def create(cls, backend_figure, plot):
        positions = plot.evaluate().positions

        if positions.shape[1] == 2:
            trace = go.Scatter(
                x=positions[:, 0], y=positions[:, 1], mode="lines"
            )
        else:
            trace = go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="lines"
            )

        backend_figure.add_trace(trace)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

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
