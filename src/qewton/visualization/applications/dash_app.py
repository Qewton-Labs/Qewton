from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.config import SliderAxis


class DashApplication(RenderApplication):
    @staticmethod
    def create(figure):
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

        # Controls
        for control in figure.controls:
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
    def create_slider(slideraxis: SliderAxis):
        return dcc.Slider(
            id=slideraxis.name,
            min=slideraxis.minimum,
            max=slideraxis.maximum,
            step=slideraxis.step,
            value=slideraxis.state,
            marks=slideraxis.marks,
        )

    @staticmethod
    def _register_callbacks(app, figure):
        inputs = []
        controls = []
        for control in figure.controls:
            if isinstance(control, SliderAxis):
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
