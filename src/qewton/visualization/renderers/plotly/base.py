import plotly.graph_objects as go

from qewton.visualization.renderers.base import Renderer


class PlotlyRenderer(Renderer):
    def render(self, plot):

        fig = self._create_figure(plot)

        self._apply_theme(fig, plot.theme)

        return fig

    def show(self, plot):

        fig = self.render(plot)

        fig.show()

    def save_html(self, plot, path):

        fig = self.render(plot)

        fig.write_html(path)

    def save_image(self, plot, path):

        fig = self.render(plot)

        fig.write_image(path)
