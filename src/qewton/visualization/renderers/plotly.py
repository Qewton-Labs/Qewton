from plotly import graph_objects as go

from .base import Artist, Renderer


class PlotlyRenderer(Renderer):

    @staticmethod
    def setup():
        return go.Figure()

    class ImageArtist(Artist):
        def __init__(self, idx) -> None:
            super().__init__()
            self.figure_idx = idx

        @classmethod
        def create(
            cls,
            backend_figure,
            plot,
        ):
            image = plot.evaluate()
            trace = trace = go.Image(z=image)
            backend_figure.add_trace(trace)
            if plot.title is not None:
                backend_figure.update_layout(title=plot.title)
            return cls(len(backend_figure.data) - 1)

        def update(self, backend_figure, plot):
            image = plot.evaluate()
            backend_figure.data[self.figure_idx].z = image
            if plot.title is not None:
                backend_figure.update_layout(title=plot.title)

        def remove(self, backend_figure):
            pass

    @staticmethod
    def show(backend_figure):
        backend_figure.show()

    @staticmethod
    def save_html(backend_figure, path):
        backend_figure.write_html(path)

    @staticmethod
    def slider(
        backend_figure,
        slider_axis,
    ):
        pass
