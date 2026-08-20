from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import SliderSpec, VariableSpec


class DashApplication(RenderApplication):
    """Serves a Figure as a Dash app, with one slider widget per SliderSpec
    control and one dropdown per VariableSpec - other control types
    (FixedSpec, FacetSpec, TimeSpec) are non-interactive and get no widget."""

    @staticmethod
    def create(figure):
        """Builds a Dash app displaying `figure`, with a callback that
        re-draws it whenever a slider's or dropdown's value changes."""
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

        # VariableSpecs aren't in figure.controls (see Plot.variable_specs) -
        # collected separately, one dropdown each.
        for spec in figure.variable_specs:
            widgets.append(
                html.Div(
                    [
                        html.Label(spec.name),
                        DashApplication.create_dropdown(spec),
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
    def create_dropdown(spec: VariableSpec):
        """Options are candidate indices, not the Variables themselves -
        dcc.Dropdown needs a JSON-serializable value. VariableSpec.state's
        setter already accepts an int index directly, so the callback below
        needs no translation either."""
        return dcc.Dropdown(
            id=spec.name,
            options=[
                {"label": candidate.name, "value": i}
                for i, candidate in enumerate(spec.candidates)
            ],
            value=spec.candidates.index(spec.state),
            clearable=False,
        )

    @staticmethod
    def _register_callbacks(app, figure):
        inputs = []
        controls = []
        for control in figure.controls:
            if isinstance(control, SliderSpec):
                inputs.append(Input(control.name, "value"))
                controls.append(control)
        for spec in figure.variable_specs:
            inputs.append(Input(spec.name, "value"))
            controls.append(spec)

        @app.callback(
            Output("figure", "figure"),
            inputs,
        )
        def update(*values):
            for control, value in zip(controls, values):
                control.state = value
            return figure.draw()
