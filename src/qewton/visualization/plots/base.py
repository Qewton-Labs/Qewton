import itertools

import numpy as np

from qewton.visualization.plots.spec import ControlSpec, SliderSpec


def axis_names_from_variable(variable, n: int) -> list[str]:
    """Best-effort per-axis names for a plot family whose positions come
    directly from a geometry's own coordinate Variable rather than an
    explicit AxisSpec (mesh/geometry/vector/embedded-grid plots have no
    "x ="/"y =" role for the user to name - the geometry already names its
    own axes via composition, e.g. `Variable("x",1) * Variable("y",1) *
    Variable("z",1)`, or auto-names them for a plain `Variable("x", dim=3)`
    -> `x_0`/`x_1`/`x_2`).

    Falls back to generic x/y/z (or axis_0, axis_1, ... beyond 3) if the
    variable is missing or doesn't decompose into exactly `n` leaves - never
    raises, since a wrong/missing axis label is a cosmetic problem, not
    worth failing a render over.
    """
    if variable is not None:
        leaves = variable.leaves
        if len(leaves) == n:
            return [leaf.name for leaf in leaves]
    return (["x", "y", "z"] + [f"axis_{i}" for i in range(3, n)])[:n]


class Plot:
    """Something that can be evaluated and drawn - not necessarily something
    with a DataConfiguration. Four input families sit under this (see the
    implementation plan, section 1, "The input-family boundary"): DataPlot
    (data + DataConfiguration, plots/data/), TablePlot (named columns,
    plots/table/), GeometryPlot (a Geometry, plots/geometry/) and GraphPlot
    (a Graph, plots/graph/, not built yet). Everything below evaluate() -
    specs, Scale, controls, Figure, artists, the result-object contract - is
    shared across all four; only evaluate()/apply_controls() differ per
    family.
    """

    def __init__(
        self,
        title=None,
        theme=None,
        controls: list[ControlSpec] | None = None,
    ) -> None:
        self._title = title
        self._theme = theme
        self.controls = controls or []

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        if self._theme is None:
            self._theme = value

    def preprocess(self, data):
        return data

    @property
    def title(self):
        return self._title

    @property
    def embedding_dim(self) -> int | None:
        """How many spatial dimensions this plot draws into - 2, 3, or None.
        Renderer-agnostic on purpose: a facet grid needs to know this to size
        each cell's subplot, but the translation into a specific backend's
        vocabulary (Plotly's make_subplots `specs`: "xy"/"scene"/"domain"; a
        hypothetical Matplotlib renderer's `projection="3d"`; ...) belongs in
        that renderer, not here. Default 2; plots that always draw into 3D
        space (go.Surface/go.Mesh3d-equivalents) override it. None means
        non-spatial - no cartesian axes at all (ParallelCoordinatesPlot,
        GraphPlot), laid out by the backend in a slot with no x/y semantics.
        Unused for a non-faceted (1x1) Figure."""
        return 2

    def evaluate(self):
        raise NotImplementedError

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        # uses self.data_config.evaluate_data(self.data, self.data_config)
        # to make it use the current state of the PlotAxis
        raise NotImplementedError

    def color_values(self):
        """Values to train this plot's shared color Scale, or None.

        None both when the plot has no ColorSpec and when its ColorSpec has
        no Scale attached - the common case, letting Figure.draw() skip the
        extra evaluate() call entirely. Generic across plot families because
        every evaluate() returns a result dataclass with a `.color` field.

        For a plot with SliderSpec controls, trains on every slider state, not
        just whichever is current when Figure.draw() runs - otherwise the
        colorbar would silently reset to whatever's on screen at redraw time
        (jumping around during scrubbing) instead of staying fixed. FixedSpec
        controls are left alone: they never move, so only their fixed state
        is ever drawn.
        """
        spec = getattr(self, "color", None)
        if spec is None or spec.scale is None:
            return None

        sliders = [c for c in self.controls if isinstance(c, SliderSpec)]
        if not sliders:
            return getattr(self.evaluate(), "color", None)

        originals = [s.state for s in sliders]
        try:
            all_values = []
            state_ranges = (range(s.minimum, s.maximum + 1, s.step) for s in sliders)
            for states in itertools.product(*state_ranges):
                for slider, state in zip(sliders, states):
                    slider.state = state
                values = getattr(self.evaluate(), "color", None)
                if values is not None:
                    all_values.append(np.asarray(values).reshape(-1))
            return np.concatenate(all_values) if all_values else None
        finally:
            for slider, original in zip(sliders, originals):
                slider.state = original
