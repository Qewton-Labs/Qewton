from plotly import graph_objects as go

from qewton.visualization.plots.config import XAxis, YAxis, ZAxis

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
        axis_order = [XAxis, YAxis]

        @classmethod
        def create(
            cls,
            backend_figure,
            plot,
        ):
            image = plot.evaluate(cls.axis_order)
            trace = trace = go.Image(z=image)
            backend_figure.add_trace(trace)
            if plot.title is not None:
                backend_figure.update_layout(title=plot.title)

            x = plot.plot_config.get_axis(XAxis)
            y = plot.plot_config.get_axis(YAxis)
            if x is not None:
                backend_figure.update_xaxes(
                    title=x.name,
                    type="log" if x.log_scale else "linear",
                )

            if y is not None:
                backend_figure.update_yaxes(
                    title=y.name,
                    type="log" if y.log_scale else "linear",
                )

            return cls(len(backend_figure.data) - 1)

        def update(self, backend_figure, plot):
            image = plot.evaluate(self.axis_order)
            backend_figure.data[self.figure_idx].z = image
            if plot.title is not None:
                backend_figure.update_layout(title=plot.title)

            x = plot.plot_config.get_axis(XAxis)
            y = plot.plot_config.get_axis(YAxis)
            if x is not None:
                backend_figure.update_xaxes(
                    title=x.name,
                    type="log" if x.log_scale else "linear",
                )

            if y is not None:
                backend_figure.update_yaxes(
                    title=y.name,
                    type="log" if y.log_scale else "linear",
                )

        def remove(self, backend_figure):
            pass

    class SurfaceArtist(PlotlyArtist):
        axis_order = [XAxis, YAxis, ZAxis]

        @classmethod
        def create(
            cls,
            backend_figure,
            plot,
        ):
            data = plot.evaluate(cls.axis_order)
            trace = trace = go.Surface(z=data)
            backend_figure.add_trace(trace)
            if plot.title is not None:
                backend_figure.update_layout(title=plot.title)

            x = plot.plot_config.get_axis(XAxis)
            y = plot.plot_config.get_axis(YAxis)
            z = plot.plot_config.get_axis(ZAxis)
            if x is not None:
                backend_figure.update_xaxes(
                    title=x.name,
                    type="log" if x.log_scale else "linear",
                )

            if y is not None:
                backend_figure.update_yaxes(
                    title=y.name,
                    type="log" if y.log_scale else "linear",
                )

            if z is not None:
                backend_figure.update_yaxes(
                    title=z.name,
                    type="log" if z.log_scale else "linear",
                )

            return cls(len(backend_figure.data) - 1)

        def update(self, backend_figure, plot):
            data = plot.evaluate(self.axis_order)
            backend_figure.data[self.figure_idx].z = data
            if plot.title is not None:
                backend_figure.update_layout(title=plot.title)

            x = plot.plot_config.get_axis(XAxis)
            y = plot.plot_config.get_axis(YAxis)
            z = plot.plot_config.get_axis(ZAxis)
            if x is not None:
                backend_figure.update_xaxes(
                    title=x.name,
                    type="log" if x.log_scale else "linear",
                )

            if y is not None:
                backend_figure.update_yaxes(
                    title=y.name,
                    type="log" if y.log_scale else "linear",
                )

            if z is not None:
                backend_figure.update_yaxes(
                    title=z.name,
                    type="log" if z.log_scale else "linear",
                )

        def remove(self, backend_figure):
            pass

    @staticmethod
    def show(backend_figure):
        backend_figure.show()

    @staticmethod
    def save_html(backend_figure, path):
        backend_figure.write_html(path)
