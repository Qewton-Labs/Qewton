"""Interactive Dash application for a Figure."""

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import numpy as np

from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import PlotSpec, SliderSpec, SelectorSpec


#: How many marks of a slider over coordinates are labeled.
_N_LABELED_MARKS = 10


def _selector_id(index: int) -> str:
    return f"selector-spec-{index}"


def _slider_id(index: int) -> str:
    return f"slider-spec-{index}"


class DashApplication(RenderApplication):
    """Serves a Figure as a Dash app with interactive widgets.

    Each :class:`~qewton.visualization.plots.spec.SliderSpec` of the
    figure gets a slider and each
    :class:`~qewton.visualization.plots.spec.SelectorSpec` a dropdown.
    Other control types (FixedSpec, FacetSpec, TimeSpec) get no widget.

    See Also:
        :class:`~qewton.visualization.applications.base.RenderApplication`
    """

    @staticmethod
    def create(figure):
        """Build a Dash app displaying `figure`.

        A callback sets the controls' states and redraws the figure
        whenever a slider or dropdown changes.

        Args:
            figure (Figure): The figure to serve.

        Returns:
            dash.Dash: The application; start it with ``app.run()``.
        """
        app = Dash(__name__)
        if figure.title is not None:
            # Only set when given: Dash requires a string here.
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
                # Needed to render $...$ LaTeX in titles and labels.
                mathjax=True,
            )
        )

        # Controls - only SliderSpec gets an interactive widget.
        for i, control in enumerate(DashApplication._slider_controls(figure)):
            widgets.append(
                html.Div(
                    [
                        html.Label(DashApplication._slider_label(figure, control)),
                        DashApplication.create_slider(control, id=_slider_id(i)),
                    ]
                )
            )

        # SelectorSpecs are not in figure.controls (see Plot.selector_specs).
        for i, spec in enumerate(figure.selector_specs):
            widgets.append(
                html.Div(
                    [
                        html.Label(DashApplication._selector_label(figure, spec)),
                        DashApplication.create_dropdown(spec, id=_selector_id(i)),
                    ]
                )
            )

        app.layout = html.Div(widgets)

    @staticmethod
    def _slider_controls(figure) -> list[SliderSpec]:
        """Every SliderSpec in figure.controls. A widget's position in this
        list determines its component id (_slider_id)."""
        return [c for c in figure.controls if isinstance(c, SliderSpec)]

    @staticmethod
    def _slider_label(figure, control: SliderSpec) -> str:
        """Label naming the plot(s) a SliderSpec belongs to (by title, label
        or position) and the control's name, since two sliders can share a
        name (e.g. two same-shaped batch axes)."""
        owners = [
            plot.title or plot.label or f"plot {plot.color_index}"
            for plot in figure.plots
            if control in plot.controls
        ]
        if not owners:
            return control.name
        return f"{' / '.join(owners)}: {control.name}"

    @staticmethod
    def _selector_label(figure, spec: SelectorSpec) -> str:
        """Label naming the plot role(s) a SelectorSpec is embedded in (x,
        y, color, ...) and its candidates."""
        roles = []
        for plot in figure.plots:
            for role, value in vars(plot).items():
                if (
                    isinstance(value, PlotSpec)
                    and value.embedded_selector_spec is spec
                    and role not in roles
                ):
                    roles.append(role)
        if not roles:
            return spec.name
        return f"{' / '.join(roles)}: {spec.name}"

    @staticmethod
    def create_slider(sliderspec: SliderSpec, id: str | None = None):
        """Create a Dash slider for a SliderSpec.

        Without coordinates, the slider runs over the integer states. With
        coordinates (e.g. the physical time of each step), it runs over the
        coordinates, snapping to the states' coordinates, and about ten of
        them are labeled unless ``sliderspec.marks`` is set.

        Args:
            sliderspec: The control the slider sets.
            id: Component id, unique per control. Defaults to
                ``sliderspec.name``, which is not unique for two unnamed
                axes of the same shape.

        Returns:
            dash.dcc.Slider: The slider component.
        """
        id = id if id is not None else sliderspec.name
        if sliderspec.coordinates is None:
            return dcc.Slider(
                id=id,
                min=sliderspec.minimum,
                max=sliderspec.maximum,
                step=sliderspec.step,
                value=sliderspec.state,
                marks=sliderspec.marks,
            )
        states = list(
            range(sliderspec.minimum, sliderspec.maximum + 1, sliderspec.step)
        )
        marks = sliderspec.marks
        if marks is None:
            labeled = set(
                np.linspace(0, len(states) - 1, min(len(states), _N_LABELED_MARKS))
                .round()
                .astype(int)
            )
            marks = {
                sliderspec.coordinate(state): (
                    sliderspec.label(state) if i in labeled else ""
                )
                for i, state in enumerate(states)
            }
        return dcc.Slider(
            id=id,
            min=sliderspec.coordinate(states[0]),
            max=sliderspec.coordinate(states[-1]),
            step=None,
            value=sliderspec.coordinate(sliderspec.state),
            marks=marks,
        )

    @staticmethod
    def create_dropdown(spec: SelectorSpec, id: str | None = None):
        """Create a Dash dropdown for a SelectorSpec.

        The option values are candidate indices, which
        ``SelectorSpec.state`` accepts directly.

        Args:
            spec: The SelectorSpec the dropdown sets.
            id: Component id, unique per spec. Defaults to ``spec.name``,
                which two specs over the same candidates share.

        Returns:
            dash.dcc.Dropdown: The dropdown component.
        """
        return dcc.Dropdown(
            id=id if id is not None else spec.name,
            options=[
                {"label": SelectorSpec.candidate_name(candidate), "value": i}
                for i, candidate in enumerate(spec.candidates)
            ],
            value=spec.candidates.index(spec.state),
            clearable=False,
        )

    @staticmethod
    def _register_callbacks(app, figure):
        inputs = []
        controls = []
        for i, control in enumerate(DashApplication._slider_controls(figure)):
            inputs.append(Input(_slider_id(i), "value"))
            controls.append(control)
        for i, spec in enumerate(figure.selector_specs):
            inputs.append(Input(_selector_id(i), "value"))
            controls.append(spec)

        @app.callback(
            Output("figure", "figure"),
            inputs,
        )
        def update(*values):
            for control, value in zip(controls, values):
                if getattr(control, "coordinates", None) is not None:
                    value = control.state_at(value)
                control.state = value
            return figure.draw()
