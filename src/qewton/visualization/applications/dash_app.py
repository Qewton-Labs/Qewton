from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from qewton.visualization.applications.base import RenderApplication
from qewton.visualization.plots.spec import PlotSpec, SliderSpec, SelectorSpec


def _selector_id(index: int) -> str:
    return f"selector-spec-{index}"


def _slider_id(index: int) -> str:
    return f"slider-spec-{index}"


class DashApplication(RenderApplication):
    """Serves a Figure as a Dash app, with one slider widget per SliderSpec
    control and one dropdown per SelectorSpec - other control types
    (FixedSpec, FacetSpec, TimeSpec) are non-interactive and get no widget."""

    @staticmethod
    def create(figure):
        """Builds a Dash app displaying `figure`, with a callback that
        re-draws it whenever a slider's or dropdown's value changes."""
        app = Dash(__name__)
        if figure.title is not None:
            # Left at Dash's own default ("Dash") otherwise - assigning
            # None here crashes the very first page render, since Dash
            # interpolates it directly into the index HTML template via
            # str.replace(), which requires a string.
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
                # Dash's dcc.Graph doesn't render $...$ LaTeX (PlotSpec.
                # math_name/axis_names_from_variable) unless explicitly
                # opted into - it's off by default.
                mathjax=True,
            )
        )

        # Controls - only SliderSpec gets an interactive widget. FixedSpec is
        # deliberately non-interactive (one plot pinned to one state, e.g.
        # several slices shown at once via separate FixedSpec-controlled
        # plots), and has none of the min/max/step/marks a slider needs.
        for i, control in enumerate(DashApplication._slider_controls(figure)):
            widgets.append(
                html.Div(
                    [
                        html.Label(DashApplication._slider_label(figure, control)),
                        DashApplication.create_slider(control, id=_slider_id(i)),
                    ]
                )
            )

        # SelectorSpecs aren't in figure.controls (see Plot.selector_specs) -
        # collected separately, one dropdown each.
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
        """Every SliderSpec in figure.controls, in a fixed order shared by
        _build_layout() and _register_callbacks() - each widget's position
        in this list is its component id (_slider_id), so the two stay in
        sync without either needing the other's ids up front."""
        return [c for c in figure.controls if isinstance(c, SliderSpec)]

    @staticmethod
    def _slider_label(figure, control: SliderSpec) -> str:
        """Which plot(s) a SliderSpec belongs to, identified by title/label
        if either is set, else by position - same purpose as
        _selector_label for dropdowns: two sliders can easily share a name
        (e.g. two different panels' same-shaped batch axis both stringify
        to e.g. "BatchAxes([32])"), so the label needs its own
        disambiguation independent of `.name`."""
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
        """The plot attribute(s) a SelectorSpec is embedded in (x, y, color,
        ...) followed by its candidates, so two dropdowns over the same
        candidates stay distinguishable."""
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
        """`id` must be unique per control: SliderSpec.name falls back to
        str(variable_or_axes) (PlotSpec.name) when its axis isn't a named
        Variable, and two different Axes instances of the same shape (e.g.
        two panels' own same-sized batch axis) stringify identically -
        component ids that collided this way, in the past, left Dash
        tracking one shared value for what looked like several independent
        sliders."""
        return dcc.Slider(
            id=id if id is not None else sliderspec.name,
            min=sliderspec.minimum,
            max=sliderspec.maximum,
            step=sliderspec.step,
            value=sliderspec.state,
            marks=sliderspec.marks,
        )

    @staticmethod
    def create_dropdown(spec: SelectorSpec, id: str | None = None):
        """Options are candidate indices, not the Variables themselves -
        dcc.Dropdown needs a JSON-serializable value. SelectorSpec.state's
        setter already accepts an int index directly, so the callback below
        needs no translation either.

        `id` must be unique per spec: SelectorSpec.name is derived from the
        candidates, so two specs over the same candidates (a scatter's x and
        y) share a name and would collide as component ids."""
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
                control.state = value
            return figure.draw()
