from plotly import graph_objects as go


from .base import Artist, Renderer


class PlotlyArtist(Artist):
    def __init__(self, idx) -> None:
        super().__init__()
        self.figure_idx = idx


class PlotlyRenderer(Renderer):

    @staticmethod
    def setup():
        fig = go.Figure()
        fig.update_layout(uirevision=True)
        return fig

    class ImageArtist(PlotlyArtist):
        @classmethod
        def create(
            cls,
            backend_figure,
            plot,
        ):
            image, _ = plot.evaluate()
            trace = trace = go.Image(z=image)

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
            image = plot.evaluate()
            backend_figure.data[self.figure_idx].z = image

        def remove(self, backend_figure):
            pass

    class HeatmapArtist(PlotlyArtist):

        @classmethod
        def create(
            cls,
            backend_figure,
            plot,
        ):
            data, color = plot.evaluate()
            c = plot.color

            cmap = (
                c.cmap
                if c is not None and c.cmap is not None
                else plot.theme.default_cmap
            )
            if color is not None:
                data = color

            trace = go.Heatmap(z=data[..., 0], colorscale=cmap)

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
            data, color = plot.evaluate()
            if color is not None:
                data = color
            backend_figure.data[self.figure_idx].z = data[..., 0]
            backend_figure.data[self.figure_idx].coloraxis = color

        def remove(self, backend_figure):
            pass

    class SurfaceArtist(PlotlyArtist):
        @classmethod
        def create(
            cls,
            backend_figure,
            plot,
        ):
            cmap = plot.theme.default_cmap
            if plot.color is not None:
                if plot.color.cmap is not None:
                    cmap = plot.color.cmap

            data, color = plot.evaluate()
            trace = trace = go.Surface(
                z=data[..., 0], surfacecolor=color, colorscale=cmap
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
            if plot.z is not None:
                backend_figure.update_layout(
                    scene=dict(
                        zaxis=dict(
                            title=plot.z.name,
                            type="log" if plot.z.log_scale else "linear",
                        )
                    )
                )

            return cls(len(backend_figure.data) - 1)

        def update(self, backend_figure, plot):
            data, color = plot.evaluate()
            backend_figure.data[self.figure_idx].z = data[..., 0]
            backend_figure.data[self.figure_idx].surfacecolor = color

        def remove(self, backend_figure):
            pass

    @staticmethod
    def show(backend_figure):
        backend_figure.show()

    @staticmethod
    def save_html(backend_figure, path):
        backend_figure.write_html(path)
