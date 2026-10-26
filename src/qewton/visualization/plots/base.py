"""Base class :class:`Plot` shared by every plot family."""

import itertools

import numpy as np

from qewton.visualization.plots.spec import (
    ControlSpec,
    PlotSpec,
    SliderSpec,
    SelectorSpec,
    TimeSpec,
)


def axis_names_from_variable(variable, n: int) -> list[str]:
    """Derive per-axis labels from a geometry's coordinate Variable.

    Used by plot families whose positions come directly from a geometry
    (mesh, geometry, vector, embedded-grid plots) rather than from an
    :class:`~qewton.visualization.plots.spec.AxisSpec`.

    Args:
        variable (Variable or None): The geometry's coordinate
            :class:`~qewton.config.variables.Variable`.
        n: Number of axes to label.

    Returns:
        list of str: One label per axis. If `variable` decomposes into exactly
            `n` leaves, each label is the leaf's ``math_name`` (TeX math mode,
            e.g. ``Variable("x", 1) * Variable("y", 1)`` gives ``["$x$",
            "$y$"]``). Otherwise the plain-text fallback ``["axis_1", ...,
            "axis_n"]``. Never raises.
    """
    if variable is not None:
        leaves = variable.leaves
        if len(leaves) == n:
            return [leaf.math_name for leaf in leaves]
    return [f"axis_{i}" for i in range(1, n + 1)]


class Plot:
    """Base class for anything that can be evaluated and drawn.

    Four input families derive from it:
    :class:`~qewton.visualization.plots.data.base.DataPlot` (data plus a
    DataConfiguration),
    :class:`~qewton.visualization.plots.table.base.TablePlot` (named
    columns), :class:`~qewton.visualization.plots.geometry.GeometryPlot` (a
    Geometry) and :class:`~qewton.visualization.plots.graph.GraphPlot` (a
    computation Graph). Specs, Scale, controls, Figure, artists and the
    result-object contract are shared by all of them.

    Args:
        title (str, optional): Panel heading, shown as subplot annotation or
            figure title.
        label (str, optional): Legend entry text. None falls back to the
            family's default, e.g. the name of the plotted variable.
        theme (Theme, optional): Theme used to draw this plot. If None, the
            Figure's theme is assigned when the plot is added.
        controls: Control specs (slider, fixed, facet, time) that reduce or
            partition this plot's data.

    Attributes:
        controls (list of ControlSpec): The control specs passed as `controls`.
        color_index (int or None): Position of this plot among all plots of its
            Figure, in add order; set by ``Figure.add_plot()``, None before.
            Artists without data-driven coloring use it with
            ``theme.get_color(index)`` so overlaid traces get distinct colors.
        theme (Theme or None): Theme of this plot. Setting it only has an
            effect while it is None.
        title (str or None): Panel heading.
        label (str or None): Legend entry text, distinct from `title`.
        embedding_dim (int or None): Number of spatial dimensions the plot
            draws into: 2 (default), 3, or None for non-spatial plots without
            cartesian axes (e.g. ParallelCoordinatesPlot, GraphPlot). Used by a
            faceted Figure to choose each cell's subplot type.
        selector_specs (list of SelectorSpec): Every
            :class:`~qewton.visualization.plots.spec.SelectorSpec` embedded in
            one of this plot's
            :class:`~qewton.visualization.plots.spec.PlotSpec` attributes,
            found by scanning the instance attributes. Selector specs are never
            listed in `controls`.

    Notes:
        Subclasses must implement :meth:`evaluate` and :meth:`create_artist`.
        They may override `embedding_dim` and, if they support controls,
        provide an ``apply_controls()`` that reduces their input to the current
        control states. A plot with a ``color`` attribute (a
        :class:`~qewton.visualization.plots.spec.ColorSpec`) whose result has a
        ``color`` field gets :meth:`color_values` for free.
    """

    def __init__(
        self,
        title=None,
        label=None,
        theme=None,
        controls: list[ControlSpec] | None = None,
    ) -> None:
        self._title = title
        self._label = label
        self._theme = theme
        self.controls = controls or []
        self.color_index: int | None = None

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

    @property
    def label(self):
        return self._label

    @property
    def embedding_dim(self) -> int | None:
        return 2

    def evaluate(self):
        """Evaluate the plot in its current control state.

        Returns:
            object: A renderer-agnostic result dataclass from
                :mod:`qewton.visualization.plots.result`, e.g.
                :class:`~qewton.visualization.plots.result.GridResult`.
        """
        raise NotImplementedError

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer-specific artist that draws this plot.

        Args:
            backend_figure (object): The renderer's figure object to draw into.
            renderer (Renderer): The
                :class:`~qewton.visualization.renderers.base.Renderer` whose
                artist class is used.
            row (int, optional): Facet grid row (1-indexed), or None for a
                non-faceted figure.
            col (int, optional): Facet grid column (1-indexed), or None for a
                non-faceted figure.

        Returns:
            Artist: The new
                :class:`~qewton.visualization.renderers.base.Artist`, already
                drawn from the plot's current control state.
        """
        raise NotImplementedError

    @property
    def selector_specs(self) -> list[SelectorSpec]:
        found = []
        for value in vars(self).values():
            if isinstance(value, PlotSpec):
                spec = value.embedded_selector_spec
                if spec is not None and spec not in found:
                    found.append(spec)
        return found

    def color_values(self):
        """Collect the values that train this plot's shared color Scale.

        With SliderSpec or TimeSpec controls, the plot is evaluated in every
        combination of their states and the values are concatenated; the
        original states are restored afterwards. Other controls keep their
        current state.

        Returns:
            numpy.ndarray or None: The color values (flattened when collected
                over several states), or None if the plot has no ``color``
                spec, the spec has no
                :class:`~qewton.visualization.plots.spec.Scale`, or the result
                has no color values.
        """
        spec = getattr(self, "color", None)
        if spec is None or spec.scale is None:
            return None

        stepping = [c for c in self.controls if isinstance(c, (SliderSpec, TimeSpec))]
        if not stepping:
            return getattr(self.evaluate(), "color", None)

        originals = [s.state for s in stepping]
        try:
            all_values = []
            state_ranges = (
                s.values
                if isinstance(s, TimeSpec)
                else range(s.minimum, s.maximum + 1, s.step)
                for s in stepping
            )
            for states in itertools.product(*state_ranges):
                for control, state in zip(stepping, states):
                    control.state = state
                values = getattr(self.evaluate(), "color", None)
                if values is not None:
                    all_values.append(np.asarray(values).reshape(-1))
            return np.concatenate(all_values) if all_values else None
        finally:
            for control, original in zip(stepping, originals):
                control.state = original
