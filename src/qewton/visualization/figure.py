from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.spec import ControlSpec, FacetSpec
from qewton.visualization.renderers.base import Artist, Renderer
from qewton.visualization.themes.base import Theme
from qewton.visualization.renderers import DEFAULT_RENDERER
from qewton.visualization.themes import DEFAULT_THEME


class Figure:
    def __init__(
        self,
        plots: Plot | list[Plot] | None = None,
        renderer: Renderer = DEFAULT_RENDERER,
        theme: Theme = DEFAULT_THEME,
        title=None,
    ):
        self.renderer = renderer
        self.theme = theme
        self.title = title
        self.plots = []
        self.controls = []

        if plots is not None:
            if isinstance(plots, list):
                for plot in plots:
                    assert isinstance(plot, Plot)
                    self.add_plot(plot)
            else:
                assert isinstance(plots, Plot)
                self.add_plot(plots)

        self.artists: dict[Plot, Artist] = {}  # store the already drawn objects
        self.legend = None
        self.backend_figure = renderer.setup()

    def add_plot(self, plot: Plot):
        plot.theme = self.theme
        self.plots.append(plot)
        for spec in plot.controls:
            if isinstance(spec, ControlSpec) and not isinstance(spec, FacetSpec):
                if spec not in self.controls:
                    self.controls.append(spec)

    @staticmethod
    def facet_specs(plot: Plot) -> dict[str, FacetSpec]:
        specs = [s for s in plot.controls if isinstance(s, FacetSpec)]
        by_orientation = {s.orientation: s for s in specs}
        assert len(by_orientation) == len(
            specs
        ), "At most one FacetSpec per orientation (row/col) allowed per plot."
        return by_orientation

    def draw(self):
        for plot in self.plots:

            artist = self.artists.get(plot)
            if artist is None:
                artist = plot.create_artist(self.backend_figure, self.renderer)
                self.artists[plot] = artist

            else:
                artist.update(self.backend_figure, plot)

        return self.backend_figure

    def show(self):
        self.draw()
        self.renderer.show(self.backend_figure)

    def save_html(self, path):
        self.draw()
        self.renderer.save_html(self.backend_figure, path)
