import itertools

import numpy as np

from qewton.visualization.plots.spec import ControlSpec, PlotSpec, SliderSpec, VariableSpec


def axis_names_from_variable(variable, n: int) -> list[str]:
    """Derives per-axis labels from a geometry's coordinate Variable.

    Used by plot families whose positions come directly from a geometry
    (mesh, geometry, vector, embedded-grid plots) rather than an explicit
    AxisSpec, so there's no separate "x="/"y=" name for the user to supply.
    A composed Variable such as ``Variable("x", 1) * Variable("y", 1) *
    Variable("z", 1)`` yields ``["x", "y", "z"]``; a plain ``Variable("x",
    dim=3)`` is auto-named ``["x_0", "x_1", "x_2"]``.

    Falls back to generic labels (``x``, ``y``, ``z``, then ``axis_3``,
    ``axis_4``, ...) if `variable` is None or doesn't decompose into
    exactly `n` leaves. Never raises - a wrong or missing axis label is
    cosmetic, not worth failing a render over.
    """
    if variable is not None:
        leaves = variable.leaves
        if len(leaves) == n:
            return [leaf.name for leaf in leaves]
    return (["x", "y", "z"] + [f"axis_{i}" for i in range(3, n)])[:n]


class Plot:
    """Base class for anything that can be evaluated and drawn.

    A Plot doesn't necessarily carry a DataConfiguration - four input
    families derive from it, each describing a different kind of input:
    DataPlot (data + DataConfiguration, ``plots/data/``), TablePlot (named
    columns, ``plots/table/``), GeometryPlot (a Geometry, ``plots/geometry/``)
    and GraphPlot (a computation Graph, ``plots/graph/``). Everything below
    `evaluate()` - specs, Scale, controls, Figure, artists, and the result-
    object contract - is shared across all four; only `evaluate()` and
    `apply_controls()` differ per family.
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
        """How many spatial dimensions this plot draws into: 2, 3, or None.

        A faceted Figure needs this to size each grid cell's subplot; the
        translation into a specific renderer's vocabulary (e.g. Plotly's
        `"xy"`/`"scene"` subplot types) happens in the renderer, not here.
        Defaults to 2; plots that always draw into 3D space override it.
        None means non-spatial - no cartesian axes at all (e.g.
        ParallelCoordinatesPlot, GraphPlot), laid out by the renderer in a
        slot with no x/y semantics. Unused for a non-faceted (1x1) Figure.
        """
        return 2

    def evaluate(self):
        raise NotImplementedError

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Creates the renderer-specific Artist that draws this plot into
        `backend_figure`, evaluating it in its current control state."""
        raise NotImplementedError

    @property
    def variable_specs(self) -> list[VariableSpec]:
        """Every VariableSpec embedded in one of this plot's own PlotSpec
        attributes (color, vector, x, y, ...), found generically by scanning
        for PlotSpec-typed attributes rather than needing per-role
        registration. Unlike `self.controls` (SliderSpec/FixedSpec/...,
        passed explicitly via the `controls=` constructor argument), a
        VariableSpec is discovered this way because it is never itself a
        whole-axis control - it only ever appears wrapped inside another
        spec's `variable_or_axes`."""
        found = []
        for value in vars(self).values():
            if isinstance(value, PlotSpec):
                spec = value.embedded_variable_spec
                if spec is not None and spec not in found:
                    found.append(spec)
        return found

    def color_values(self):
        """Values used to train this plot's shared color Scale, or None.

        Returns None both when the plot has no ColorSpec and when its
        ColorSpec has no Scale attached, letting `Figure.draw()` skip the
        extra `evaluate()` call in the common case. Works generically across
        plot families because every `evaluate()` returns a result dataclass
        with a `.color` field.

        For a plot with SliderSpec controls, values are collected across
        every slider state rather than only the state current when this
        runs, so the color range stays fixed while scrubbing instead of
        jumping around as the slider moves. FixedSpec controls are left
        alone since they never change state.
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
