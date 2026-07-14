from qewton.visualization.plots.config import PlotConfig


class Plot:
    def __init__(
        self,
        data,
        plot_config: PlotConfig,
        title=None,
        theme=None,
    ) -> None:
        self.data = data
        self.plot_config = plot_config
        self._title = title
        self._theme = theme

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        if self._theme is None:
            self._theme = value

    @property
    def title(self):
        return self._title

    def evaluate(self, required_axis_order):
        return self.plot_config.evaluate_data(self.data, required_axis_order)

    def create_artist(self, backend_figure, renderer):
        # uses self.data_config.evaluate_data(self.data, self.data_config)
        # to make it use the current state of the PlotAxis
        raise NotImplementedError
