"""Interactive Dash application for a Figure."""

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.axes import Control, Selector, SliderAxis
from qewton.visualization.renderers.plotly import PlotlyRenderer
from qewton.visualization.renderers.plotly import camera
from qewton.visualization.renderers.plotly.text import as_plain


def _control_id(index: int) -> str:
    return f"control-{index}"


class DashApplication(RenderApplication):
    """Serves a Figure as a Dash app: a slider per SliderAxis and a
    dropdown per Selector of the figure are connected, i.e. a callback sets
    their states and draws the figure again; the other controls are
    embedded into the drawn figure (a TimeAxis as an animation).

    See Also:
        :class:`~qewton.visualization.applications.base.RenderApplication`
    """

    @staticmethod
    def create(figure):
        """Build a Dash app displaying `figure`.

        Args:
            figure (Figure): The figure to serve.

        Returns:
            dash.Dash: The application; start it with ``app.run()``.

        Raises:
            ValueError: If the figure is not drawn by a PlotlyRenderer.
        """
        if not isinstance(figure.renderer, PlotlyRenderer):
            raise ValueError(
                "DashApplication shows Plotly figures; draw the figure with a "
                f"PlotlyRenderer, not {type(figure.renderer).__name__}."
            )
        app = Dash(__name__)
        if figure.title is not None:
            app.title = as_plain(figure.title)
        controls = DashApplication.interactive_controls(figure)
        embedded = DashApplication.embedded_controls(figure)
        widgets = [
            dcc.Graph(
                id="figure",
                figure=figure.draw(embedded=embedded),
                mathjax=figure.renderer.mathjax,
            )
        ]
        for index, control in enumerate(controls):
            widgets.append(
                html.Div(
                    [
                        DashApplication.label(control, figure.renderer.mathjax),
                        DashApplication.widget(control, _control_id(index)),
                    ]
                )
            )
        groups = figure.renderer.linked_scenes()
        if groups:
            # Linked 3D cells move their cameras together.
            widgets.append(dcc.Store(id="camera-sync"))
            app.clientside_callback(
                camera.dash_callback(groups, "figure"),
                Output("camera-sync", "data"),
                Input("figure", "relayoutData"),
            )
        app.layout = html.Div(widgets)

        @app.callback(
            Output("figure", "figure"),
            [Input(_control_id(i), "value") for i in range(len(controls))],
        )
        def update(*states):
            return DashApplication.update(figure, controls, states)

        return app

    @staticmethod
    def interactive_controls(figure) -> list[Control]:
        """The controls of `figure` that get a widget, in widget order.

        Args:
            figure (Figure): The figure.

        Returns:
            list of Control: Its SliderAxes and Selectors.
        """
        return [c for c in figure.controls if type(c) in _WIDGETS]

    @staticmethod
    def embedded_controls(figure) -> list[Control]:
        """The controls of `figure` without a widget, embedded into the
        drawn figure.

        Args:
            figure (Figure): The figure.

        Returns:
            list of Control: Every other control.
        """
        connected = DashApplication.interactive_controls(figure)
        return [c for c in figure.controls if c not in connected]

    @staticmethod
    def update(figure, controls: list[Control], states) -> object:
        """Set the states of `controls` and draw `figure` with the other
        controls embedded, as the app's callback does.

        Args:
            figure (Figure): The figure.
            controls: Its interactive controls.
            states: One state per control.

        Returns:
            object: The redrawn backend figure.
        """
        for control, state in zip(controls, states):
            control.state = int(state)
        return figure.draw(embedded=DashApplication.embedded_controls(figure))

    @staticmethod
    def label(control: Control, mathjax: bool):
        """The label of a control's widget: its name, with math rendered by
        MathJax if `mathjax`.

        Args:
            control: The control.
            mathjax: Whether MathJax renders math.

        Returns:
            dash component: The label.
        """
        if mathjax:
            return dcc.Markdown(control.name, mathjax=True)
        return html.Label(as_plain(control.name))

    @staticmethod
    def widget(control: Control, id: str):
        """A slider for a SliderAxis, a dropdown for a Selector.

        Args:
            control: The control the widget sets.
            id: Component id.

        Returns:
            dash component: The widget, valued by state index.
        """
        return _WIDGETS[type(control)](control, id)


def _slider(control: Control, id: str):
    states = control.states
    marks = {s: as_plain(control.label(s)) for s in states[:: max(1, len(states) // 10)]}
    return dcc.Slider(
        id=id, min=states[0], max=states[-1], step=1, value=control.state, marks=marks
    )


def _dropdown(control: Control, id: str):
    options = [{"label": as_plain(control.label(s)), "value": s} for s in control.states]
    return dcc.Dropdown(id=id, options=options, value=control.state, clearable=False)


#: The widget of each kind of control that has one.
_WIDGETS = {SliderAxis: _slider, Selector: _dropdown}
