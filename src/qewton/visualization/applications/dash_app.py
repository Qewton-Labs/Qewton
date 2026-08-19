from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import SliderSpec


class DashApplication(RenderApplication):
    """Serves a Figure as a Dash app, with one slider widget per SliderSpec
    control - other control types (FixedSpec, FacetSpec, TimeSpec) are
    non-interactive and get no widget."""

    @staticmethod
    def create(figure):
        """Builds a Dash app displaying `figure`, with a callback that
        re-draws it whenever a slider's value changes."""
        app = Dash(__name__)
        app.title = figure.title

        DashApplication._build_layout(app, figure)
        DashApplication._register_callbacks(app, figure)
        return app

    @staticmethod
    def _build_layout(app, figure):

        widgets = []

        # Plot
        widgets.append(
            dcc.Graph(
                id="figure",
                figure=figure.draw(),
            )
        )

        # Controls - only SliderSpec gets an interactive widget. FixedSpec is
        # deliberately non-interactive (one plot pinned to one state, e.g.
        # several slices shown at once via separate FixedSpec-controlled
        # plots), and has none of the min/max/step/marks a slider needs.
        for control in figure.controls:
            if not isinstance(control, SliderSpec):
                continue
            widgets.append(
                html.Div(
                    [
                        html.Label(control.name),
                        DashApplication.create_slider(control),
                    ]
                )
            )

        app.layout = html.Div(widgets)

    @staticmethod
    def create_slider(sliderspec: SliderSpec):
        return dcc.Slider(
            id=sliderspec.name,
            min=sliderspec.minimum,
            max=sliderspec.maximum,
            step=sliderspec.step,
            value=sliderspec.state,
            marks=sliderspec.marks,
        )

    @staticmethod
    def _register_callbacks(app, figure):
        inputs = []
        controls = []
        for control in figure.controls:
            if isinstance(control, SliderSpec):
                inputs.append(Input(control.name, "value"))
                controls.append(control)

        @app.callback(
            Output("figure", "figure"),
            inputs,
        )
        def update(*values):
            for control, value in zip(controls, values):
                control.state = value
            return figure.draw()
