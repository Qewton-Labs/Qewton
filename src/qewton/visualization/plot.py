from qewton.visualization.themes.base import Theme


def plot(data, *args, **kwargs):
    plt = ...
    renderer = kwargs["renderer"] or DEFAULT_PLOT_RENDERER
    renderer.show(plot)


class Plot:
    def __init__(self, data, theme=None, title=None) -> None:
        self.data = data
        self.theme = theme or Theme.default()
        self._title = title

    @property
    def title(self):
        return self._title
