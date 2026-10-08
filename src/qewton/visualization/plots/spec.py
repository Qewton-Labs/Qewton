import numpy as np

from qewton.config.axes import Axes, EllipsisAxes, EllipsisDim, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable


def _contains_variable(variable: Variable, searched: Variable) -> bool:
    """Whether `searched` is `variable` itself or part of it."""
    if searched == variable:
        return True
    return any(_contains_variable(child, searched) for child in variable.children)


def _targets_axes(control, axes: Axes, data_config: DataConfiguration) -> bool:
    """Whether `control` reduces every dimension of `axes`."""
    try:
        axis_slc, entry_slc = PlotSpec.get_slice(control.variable_or_axes, data_config)
    except (ValueError, KeyError):
        return False
    if entry_slc is not None or len(axes.shape) != 1:
        return False
    n_dims = sum(len(a.shape) for a in data_config.axes)
    if isinstance(axis_slc, slice):
        if axis_slc.stop - axis_slc.start != 1:
            return False
        axis_slc = axis_slc.start
    axis_slc = axis_slc if axis_slc >= 0 else n_dims + axis_slc
    offset = 0
    for candidate in data_config.axes:
        if candidate is axes:
            return axis_slc == offset
        offset += len(candidate.shape)
    return False


def drawn_geometry_axes(
    data_config: DataConfiguration, controls: list | None = None
) -> GeometryAxes | list[GeometryAxes] | None:
    """The GeometryAxes of `data_config` that a plot draws: all GeometryAxes
    except those a control steps through, e.g. the time axis of space-time
    data with a TimeSpec on it.

    Args:
        data_config (DataConfiguration): The configuration of the data.
        controls (list | None, optional): The plot's ControlSpecs.
            Defaults to None.

    Returns:
        GeometryAxes | list[GeometryAxes] | None: The single drawn
            GeometryAxes, a list if there are several, None if there is none
            - the same convention as DataConfiguration.geometry_axes.
    """
    drawn = [
        axes
        for axes in data_config.axes
        if isinstance(axes, GeometryAxes)
        and not any(_targets_axes(c, axes, data_config) for c in controls or [])
    ]
    if len(drawn) == 1:
        return drawn[0]
    return drawn or None


class PlotSpec:
    """Base class declaring how a Plot maps one role (x, y, color, ...) onto
    a Variable/Axes (for a DataPlot) or a column name (for a TablePlot)."""

    def __init__(
        self, n_dimensions: int, variable_or_axes: Variable | Axes | str | None
    ) -> None:
        # `variable_or_axes` is a plain str column key for TablePlot, a
        # Variable/Axes for every DataPlot family - or a SelectorSpec
        # (defined at the bottom of this module), transparently unwrapped to
        # its currently selected candidate by the property below. Every
        # consumer (get_variable_slice, this class's own `name`, artists,
        # ...) already just reads `.variable_or_axes` as if it were a plain
        # Variable, so none of them need to know SelectorSpec exists.
        # None (a ControlSpec built as `controls=FixedSpec(init_state=3)`,
        # its axis not known yet) is filled in later via the setter below -
        # see auto_plot()'s _resolve_control().
        self.n_dimensions = n_dimensions
        self._variable_or_axes = variable_or_axes

    @property
    def variable_or_axes(self):
        if isinstance(self._variable_or_axes, SelectorSpec):
            return self._variable_or_axes.state
        return self._variable_or_axes

    @variable_or_axes.setter
    def variable_or_axes(self, value) -> None:
        self._variable_or_axes = value

    @property
    def embedded_selector_spec(self) -> "SelectorSpec | None":
        """The SelectorSpec this spec's `variable_or_axes` was given, or
        None for a plain Variable/Axes/str - used by Plot.selector_specs to
        discover SelectorSpecs for widget-building, since they're never
        listed in a Plot's own `controls=`."""
        return (
            self._variable_or_axes
            if isinstance(self._variable_or_axes, SelectorSpec)
            else None
        )

    @property
    def _named_variable(self) -> "Variable | None":
        """The Variable this spec's `variable_or_axes` ultimately names,
        even when it's a GeometryAxes rather than a Variable directly - the
        geometry's own coordinate Variable, if it has exactly one. None for
        a GeometryAxes with more than one coordinate component.
        """
        variable_or_axes = self.variable_or_axes
        if isinstance(variable_or_axes, Variable):
            return variable_or_axes
        if isinstance(variable_or_axes, GeometryAxes):
            geometry_variable = variable_or_axes.geometry.variable
            if geometry_variable is not None and geometry_variable.dim == 1:
                return geometry_variable
        return None

    @property
    def name(self):
        variable = self._named_variable
        if variable is not None:
            return variable.name
        return str(self.variable_or_axes)

    @property
    def math_name(self) -> str:
        """`.name`, wrapped for TeX math-mode rendering via Variable.
        math_name when it names an actual Variable - axis/colorbar titles
        use this so plotted quantities render as math symbols rather than
        plain text. Falls back to the plain `.name` for a non-Variable spec
        (a plain Axes or TablePlot column key) - including a SelectorSpec
        currently on a plain-string candidate, which `_named_variable`
        already isn't a Variable for; SelectorSpec doesn't synthesize
        Variables around its string candidates, so this needs no special
        case for them.
        """
        variable = self._named_variable
        return variable.math_name if variable is not None else self.name

    @staticmethod
    def get_slice(variable_or_axes, data_config: DataConfiguration):
        try:
            axis_slc, entry_slc = PlotSpec._find_axis_idx(
                variable_or_axes, data_config.axes
            )
        except ValueError:
            try:
                reverse_axis_slc, entry_slc = PlotSpec._find_axis_idx(
                    variable_or_axes, data_config.axes[::-1]
                )
                if isinstance(reverse_axis_slc, int):
                    axis_slc = -1 - reverse_axis_slc
                else:
                    axis_slc = slice(
                        -1 - reverse_axis_slc.stop, -1 - reverse_axis_slc.start
                    )
            except ValueError as exc:
                raise ValueError(f"Axis {variable_or_axes} not found in data \
                        config {data_config}.") from exc
        return axis_slc, entry_slc

    @staticmethod
    def as_single_dim(axis_slc: int | slice) -> int:
        """Normalizes a get_slice() axis result to a plain int dimension
        index, when it unambiguously names one dimension. A length-1 slice
        does - e.g. one child variable of a multi-dim GeometryAxes resolves
        this way, the same mechanism HeatmapPlot's x/y and control resolution
        (Plot._resolve_controls) already rely on. Raises otherwise."""
        if isinstance(axis_slc, slice):
            length = axis_slc.stop - axis_slc.start
            assert length == 1, "Multiple axes do not work with a single control."
            return axis_slc.start
        return axis_slc

    @staticmethod
    def _find_axis_idx(
        variable_or_axis, axes: list[Axes]
    ) -> tuple[int | slice, slice | None]:
        counter = 0
        for i_axis in axes:
            if isinstance(i_axis, EllipsisAxes):
                raise ValueError
            if any(isinstance(i_dim, EllipsisDim) for i_dim in i_axis.shape):
                raise ValueError

            if i_axis is variable_or_axis:

                if len(i_axis.shape) == 1:
                    return counter, None
                else:
                    return slice(counter, counter + len(i_axis.shape)), None
            if isinstance(variable_or_axis, Variable):
                if isinstance(i_axis, (FeatureAxes)):
                    i_var = i_axis.variables
                    if variable_or_axis in i_var:
                        assert len(i_axis.shape) == 1, "for now only 1-axis variables"
                        return counter, i_var.get_slice(variable_or_axis)
                if isinstance(i_axis, GeometryAxes):
                    geometry_variable = i_axis.geometry.variable
                    if geometry_variable.is_empty and len(i_axis.shape) == 1:
                        return counter, None
                    if not _contains_variable(geometry_variable, variable_or_axis):
                        counter += len(i_axis.shape)
                        continue
                    if geometry_variable.dim == len(i_axis.shape):
                        axis_slc = geometry_variable.get_slice(variable_or_axis)
                        # Variable.get_slice(self) returns slice(None) (the
                        # "whole thing" convention for array indexing, e.g.
                        # data[..., :]) when `variable_or_axis` is the
                        # geometry's own root variable, not a decomposed
                        # child - substitute the numeric bounds it implies
                        # here, since counter + None below would TypeError.
                        start = axis_slc.start if axis_slc.start is not None else 0
                        stop = (
                            axis_slc.stop
                            if axis_slc.stop is not None
                            else geometry_variable.dim
                        )
                        return (slice(counter + start, counter + stop), None)
                    elif len(i_axis.shape) == 1:
                        return counter, None
                    else:
                        raise ValueError(
                            "geometries with mixed grid/pointcloud not supported yet"
                        )

            counter += len(i_axis.shape)

        raise ValueError


class AxisSpec(PlotSpec):
    """Declares a single structural domain axis (e.g. a LinePlot's `x`/`y`).
    `variable_or_axes` may also be a SelectorSpec, to switch which Variable
    fills this role."""

    def __init__(
        self, variable_or_axes: "Variable | Axes | SelectorSpec", log_scale: bool = False
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale


class VectorSpec(PlotSpec):
    """Declares a 2D or 3D vector-valued role (e.g. a QuiverPlot's arrows),
    with the display parameters that control how it's drawn.

    Args:
        variable_or_axes: The 2D or 3D Variable/Axes the vector components
            come from - or a SelectorSpec, to switch between several
            same-dim candidates.
        scale: Multiplier applied to every vector's length.
        normalize: If True, normalizes each vector to unit length before
            applying `scale`.
        cmap: Colormap for `color_by_magnitude`; falls back to the theme's
            default if unset.
        color_by_magnitude: If True, colors arrows by vector magnitude.
        subsample: Draws every `subsample`-th vector only, decimated after
            `normalize`/`scale` are applied - a display decision about which
            arrows to draw, not a change to what the field itself is.
    """

    def __init__(
        self,
        variable_or_axes,
        scale=1.0,
        normalize=False,
        cmap=None,
        color_by_magnitude=False,
        subsample=1,
    ):
        dim = variable_or_axes.dim
        assert dim in [2, 3], "VectorSpec only supports 2D or 3D variables"
        super().__init__(dim, variable_or_axes)
        self.scale = scale
        self.normalize = normalize
        self.cmap = cmap
        self.color_by_magnitude = color_by_magnitude
        assert subsample >= 1, "subsample must be >= 1"
        self.subsample = subsample


class Scale:
    """Shared value range for one or more ColorSpecs.

    Plots that reference the same Scale instance train it together, so e.g.
    prediction/reference/difference heatmaps side by side get one common
    cmin/cmax instead of each auto-scaling independently - without a shared
    range, differently-scaled heatmaps look identical, which is misleading.
    """

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = False,
    ) -> None:
        self.vmin = vmin  # explicit bounds always win over observed ones
        self.vmax = vmax
        self.symmetric = symmetric  # centre the range on zero (error plots)
        self._observed_min: float | None = None
        self._observed_max: float | None = None
        self._colorbar_claimed = False
        #: (row, col) of the rightmost grid cell among every plot
        #: referencing this Scale, 1-indexed - None outside a grid, or
        #: before Figure.draw() has computed it this cycle. Set once per
        #: draw() (see Figure._assign_colorbar_cells()), read by whichever
        #: renderer draws the one trace that ends up showing this Scale's
        #: colorbar, so it's placed after every panel sharing the scale,
        #: not just the first (claiming) one.
        self.colorbar_cell: tuple[int, int] | None = None

    def observe(self, values) -> None:
        """Widens the observed range to cover `values`. Called in pass 1 of
        Figure.draw() for every plot whose ColorSpec references this scale."""
        values = np.asarray(values)
        if values.size == 0:
            return
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        self._observed_min = (
            lo if self._observed_min is None else min(self._observed_min, lo)
        )
        self._observed_max = (
            hi if self._observed_max is None else max(self._observed_max, hi)
        )

    @property
    def range(self) -> tuple[float, float] | None:
        """Merged (lo, hi), preferring explicit vmin/vmax over observed
        values. None if neither explicit bounds nor any observation exist."""
        lo = self.vmin if self.vmin is not None else self._observed_min
        hi = self.vmax if self.vmax is not None else self._observed_max
        if lo is None or hi is None:
            return None
        if self.symmetric:
            bound = max(abs(lo), abs(hi))
            lo, hi = -bound, bound
        return lo, hi

    def claim_colorbar(self) -> bool:
        """First artist to call this each draw() cycle gets True; the rest
        get False, so only one colorbar is shown per shared scale."""
        if self._colorbar_claimed:
            return False
        self._colorbar_claimed = True
        return True

    def reset(self) -> None:
        """Clears the observed range, colorbar claim, and colorbar cell.
        Must run before every draw(), or the colorbar disappears on the
        second render."""
        self._observed_min = None
        self._observed_max = None
        self._colorbar_claimed = False
        self.colorbar_cell = None


class ColorSpec(PlotSpec):
    """Declares which Variable/Axes (DataPlot) or column (TablePlot) colors
    a plot, with an optional colormap and shared Scale. `variable_or_axes`
    may also be a SelectorSpec, to switch which Variable colors the plot."""

    def __init__(
        self,
        variable_or_axes: "Variable | str | SelectorSpec",
        cmap=None,
        scale: Scale | None = None,
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.cmap = cmap  # if not specified, plots resort to default cmap of theme
        self.scale = scale  # if set, shared with every other spec using the same Scale


class ControlSpec(PlotSpec):
    """Base class for a spec that reduces or partitions a plot's data by a
    dimension's current state (SliderSpec, FixedSpec, FacetSpec, TimeSpec)."""

    def __init__(
        self, init_state=None, n_dimensions: int = 1, variable_or_axes=None
    ) -> None:
        # variable_or_axes defaults to None so a ControlSpec can be built
        # before its axis is known (`controls=FixedSpec(init_state=3)`) and
        # resolved later against whichever axis actually needs one - see
        # auto_plot()'s _resolve_control().
        super().__init__(n_dimensions=n_dimensions, variable_or_axes=variable_or_axes)
        self._state = init_state
        #: Physical coordinate of each state (e.g. the time of each time
        #: step), indexed by state - None if the dimension has none.
        self.coordinates: list[float] | None = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def resolve(self, values):
        """Fills in whatever was left None - bounds, facet values, initial
        state - from `values`, the list of selectable states for this
        control's dimension. Explicit values set by the user are never
        overwritten. `values` is computed by the owning Plot (a DataPlot
        passes `range(data.shape[dim])`, a TablePlot passes the column's
        sorted unique values), so the same ControlSpec subclasses work with
        every input family."""

    def set_coordinates(self, coordinates) -> None:
        """Sets the physical coordinate of each state, e.g. the time of each
        time step, unless coordinates were already set. Called by the
        owning Plot when the control's dimension has coordinates."""
        if self.coordinates is None and coordinates is not None:
            self.coordinates = [float(c) for c in coordinates]

    def coordinate(self, state) -> float | None:
        """The physical coordinate of `state`, or None if unknown."""
        if self.coordinates is None:
            return None
        return self.coordinates[state]

    def label(self, state) -> str:
        """Display text for `state`: its coordinate if known, else the
        state itself."""
        coordinate = self.coordinate(state)
        return str(state) if coordinate is None else f"{coordinate:.4g}"

    def state_at(self, coordinate: float):
        """The state whose coordinate is closest to `coordinate`."""
        return int(np.argmin(np.abs(np.asarray(self.coordinates) - coordinate)))


class SliderSpec(ControlSpec):
    """An interactive slider over one dimension's states."""

    def __init__(
        self,
        variable_or_axes=None,
        init_state=None,
        minimum=None,
        maximum=None,
        step=1,
        marks=None,
    ):
        super().__init__(init_state, n_dimensions=1, variable_or_axes=variable_or_axes)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.marks = marks

    def resolve(self, values):
        if self.minimum is None or self.maximum is None:
            values = list(values)
            self.minimum = self.minimum if self.minimum is not None else min(values)
            self.maximum = self.maximum if self.maximum is not None else max(values)
        if self._state is None:
            self._state = self.minimum


class FixedSpec(ControlSpec):
    """Selects one fixed index of a dimension - never changes state."""

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        raise ValueError("Cannot set state of FixedAxis. It is fixed.")


class FacetSpec(ControlSpec):
    """Splits a dimension into a grid of side-by-side subplot panels, one
    per value, instead of an interactive control.
    """

    def __init__(
        self,
        variable_or_axes=None,
        values=None,
        orientation: str = "col",
        labels: list[str] | None = None,
    ):
        assert orientation in ("row", "col"), "orientation must be 'row' or 'col'"
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes, init_state=0)
        self.orientation = orientation
        self.values = values
        self.labels = labels

    def resolve(self, values):
        if self.values is None:
            self.values = list(values)
        if self.labels is not None:
            assert len(self.labels) == len(self.values), (
                f"FacetSpec labels ({len(self.labels)}) must match its "
                f"values ({len(self.values)})."
            )
        if self._state is None:
            self._state = self.values[0]


class SelectorSpec(ControlSpec):
    """A categorical, single-select control letting the user swap which of
    several candidates currently feeds another spec's role - pass one
    anywhere a Variable is expected, e.g. `ColorSpec(SelectorSpec(
    [temperature, pressure]))`. Rendered as one dropdown (DashApplication)
    or one button menu (PlotlyRenderer.apply_selector) per instance, always
    picking exactly one candidate at a time - not a whole-axis control that
    reduces/partitions data the way SliderSpec/FacetSpec/TimeSpec do.
    `PlotSpec.variable_or_axes` transparently unwraps it to whichever
    candidate is currently selected, so every existing consumer
    (get_variable_slice, axis/colorbar titles, artists, ...) keeps working
    unchanged - selecting a candidate is exactly the same "pick a slice of
    the FeatureAxes" operation a fixed Variable already describes, just
    made to react to `state` instead of staying fixed.

    Unlike SliderSpec/FacetSpec/TimeSpec, this is never passed via a Plot's
    own `controls=` - it isn't a whole-axis control for apply_controls() to
    reduce, so it never appears in Plot.controls the way those do.

    All `candidates` must share the same dim, so the role they feed stays
    valid (same required shape) regardless of which is selected. Candidates
    are kept exactly as given - a plain string stays a plain string, never
    synthesized into a Variable - so a TablePlot column key like "loss"
    never looks like a math quantity to PlotSpec.math_name; see
    candidate_name()/candidate_dim() below for how dim/name are read from
    whichever kind of candidate this instance actually has.
    """

    def __init__(self, candidates: list[Variable] | list[str], init_index: int = 0):
        assert (
            len(candidates) >= 2
        ), "SelectorSpec needs at least 2 candidates to choose between."
        dims = {self.candidate_dim(c) for c in candidates}
        assert len(dims) == 1, f"All candidates must share the same dim, got {dims}."
        self.candidates = candidates
        super().__init__(
            init_state=candidates[init_index], n_dimensions=1, variable_or_axes=None
        )

    @staticmethod
    def candidate_dim(candidate: "Variable | str") -> int:
        """A candidate's dim - `.dim` for a real Variable, 1 for a plain
        string (it names one scalar column, the same as `Variable(name,
        dim=1)` would)."""
        return candidate.dim if isinstance(candidate, Variable) else 1

    @staticmethod
    def candidate_name(candidate: "Variable | str") -> str:
        """A candidate's display name - `.name` for a real Variable, or the
        string itself, which already is its own name. The one place this
        distinction is handled, so callers building a widget/button label
        per candidate (DashApplication.create_dropdown, PlotlyRenderer.
        apply_selector) don't each need their own isinstance check."""
        return candidate.name if isinstance(candidate, Variable) else candidate

    @property
    def dim(self):
        return self.candidate_dim(self.candidates[0])

    @property
    def name(self):
        # PlotSpec.name derives from self.variable_or_axes, which is None
        # here (this spec doesn't itself wrap a Variable - its candidates
        # do) - override with something a widget can actually label itself
        # with.
        return " / ".join(self.candidate_name(c) for c in self.candidates)

    @property
    def state(self) -> "Variable | str":
        return self._state

    @state.setter
    def state(self, value: "int | Variable | str"):
        if isinstance(value, int):
            value = self.candidates[value]
        assert (
            value in self.candidates
        ), f"{value!r} is not one of this spec's candidates."
        self._state = value


class TimeSpec(ControlSpec):
    """Declares that a dimension/column advances over animation frames,
    rather than being an interactive widget (SliderSpec) or a grid of panels
    (FacetSpec).

    A renderer that materializes frames up front (e.g. Plotly) builds one
    frame per value in `values`, plus play/pause controls and a frame
    slider. Unlike SliderSpec, this does not get an interactive Dash widget.
    """

    def __init__(self, variable_or_axes, values=None, duration=500):
        super().__init__(
            n_dimensions=1, variable_or_axes=variable_or_axes, init_state=None
        )
        self.values = values
        #: Average ms per frame while the Play button runs - see frame_weights().
        self.duration = duration

    def resolve(self, values):
        if self.values is None:
            self.values = list(values)
        if self._state is None:
            self._state = self.values[0]

    def frame_weights(self) -> list[float]:
        """Relative display time of each frame in `values`, with mean 1.

        With coordinates (e.g. physical time), each frame is shown in
        proportion to the time until the next frame, so non-uniform time
        steps play back in physical time; the last frame gets the mean step.
        Without coordinates, every frame is shown equally long.
        """
        n = len(self.values)
        if self.coordinates is None or n < 2:
            return [1.0] * n
        times = np.array([self.coordinates[v] for v in self.values])
        steps = np.abs(np.diff(times))
        if steps.sum() == 0:
            return [1.0] * n
        steps = np.append(steps, steps.mean())
        return list(steps / steps.mean())
