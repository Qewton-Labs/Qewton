"""Interactive Dash application for a Figure."""

from qewton.visualization.applications.base import RenderApplication, require
from qewton.visualization.axes import Control, Selector, SliderAxis
from qewton.visualization.renderers.text import as_plain

try:
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output

    from qewton.visualization.renderers.plotly import PlotlyRenderer, camera
except ImportError:  # optional packages; create() names the missing ones
    pass


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

    connected = (SliderAxis, Selector)

    @classmethod
    def renderer_classes(cls) -> tuple[type, ...]:
        """See :meth:`RenderApplication.renderer_classes
        <qewton.visualization.applications.base.RenderApplication.renderer_classes>`."""
        return (PlotlyRenderer,)

    @classmethod
    def create(cls, figure):
        """Build a Dash app displaying `figure`.

        Args:
            figure (Figure): The figure to serve.

        Returns:
            dash.Dash: The application; start it with ``app.run()``.

        Raises:
            ImportError: If Dash or Plotly is not installed.
            TypeError: If the figure is not drawn by a PlotlyRenderer.
        """
        require(cls.__name__, "dash", "plotly")
        cls.check_renderer(figure)
        app = Dash(__name__)
        if figure.title is not None:
            app.title = as_plain(figure.title)
        controls = cls.interactive_controls(figure)
        embedded = cls.embedded_controls(figure)
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
                        cls.label(control, figure.renderer.mathjax),
                        cls.widget(control, _control_id(index)),
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
            return cls.update(figure, controls, states)

        return app

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
