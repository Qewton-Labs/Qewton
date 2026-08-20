import numpy as np

from qewton.config.axes import Axes, EllipsisAxes, EllipsisDim, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable


class PlotSpec:
    """Base class declaring how a Plot maps one role (x, y, color, ...) onto
    a Variable/Axes (for a DataPlot) or a column name (for a TablePlot)."""

    def __init__(self, n_dimensions: int, variable_or_axes: Variable | Axes | str) -> None:
        # `variable_or_axes` is a plain str column key for TablePlot, a
        # Variable/Axes for every DataPlot family - or a VariableSpec
        # (defined at the bottom of this module), transparently unwrapped to
        # its currently selected candidate by the property below. Every
        # consumer (get_variable_slice, this class's own `name`, artists,
        # ...) already just reads `.variable_or_axes` as if it were a plain
        # Variable, so none of them need to know VariableSpec exists.
        self.n_dimensions = n_dimensions
        self._variable_or_axes = variable_or_axes

    @property
    def variable_or_axes(self):
        if isinstance(self._variable_or_axes, VariableSpec):
            return self._variable_or_axes.state
        return self._variable_or_axes

    @property
    def embedded_variable_spec(self) -> "VariableSpec | None":
        """The VariableSpec this spec's `variable_or_axes` was given, or
        None for a plain Variable/Axes/str - used by Plot.variable_specs to
        discover VariableSpecs for widget-building, since they're never
        listed in a Plot's own `controls=`."""
        return self._variable_or_axes if isinstance(self._variable_or_axes, VariableSpec) else None

    @property
    def name(self):
        variable_or_axes = self.variable_or_axes
        if isinstance(variable_or_axes, Variable):
            return variable_or_axes.name
        return str(variable_or_axes)

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
                    if i_axis.geometry.variable.dim == len(i_axis.shape):
                        axis_slc = i_axis.geometry.variable.get_slice(variable_or_axis)
                        return (
                            slice(counter + axis_slc.start, counter + axis_slc.stop),
                            None,
                        )
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
    `variable_or_axes` may also be a VariableSpec, to switch which Variable
    fills this role."""

    def __init__(
        self, variable_or_axes: "Variable | Axes | VariableSpec", log_scale: bool = False
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale

        self.coordinates: np.ndarray | None = None


class VectorSpec(PlotSpec):
    """Declares a 2D or 3D vector-valued role (e.g. a QuiverPlot's arrows),
    with the display parameters that control how it's drawn.

    Args:
        variable_or_axes: The 2D or 3D Variable/Axes the vector components
            come from - or a VariableSpec, to switch between several
            same-dim candidates.
        scale: Multiplier applied to every vector's length.
        normalize: If True, normalizes each vector to unit length before
            applying `scale`.
        cmap: Colormap for `color_by_magnitude`; falls back to the theme's
            default if unset.
        color_by_magnitude: If True, colors arrows by vector magnitude.
        n_color_bins: Number of discrete magnitude bins when
            `color_by_magnitude` is set.
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
        n_color_bins=8,
        subsample=1,
    ):
        dim = variable_or_axes.dim
        assert dim in [2, 3], "VectorSpec only supports 2D or 3D variables"
        super().__init__(dim, variable_or_axes)
        self.scale = scale
        self.normalize = normalize
        self.cmap = cmap
        self.color_by_magnitude = color_by_magnitude
        self.n_color_bins = n_color_bins
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

    def observe(self, values) -> None:
        """Widens the observed range to cover `values`. Called in pass 1 of
        Figure.draw() for every plot whose ColorSpec references this scale."""
        values = np.asarray(values)
        if values.size == 0:
            return
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        self._observed_min = lo if self._observed_min is None else min(self._observed_min, lo)
        self._observed_max = hi if self._observed_max is None else max(self._observed_max, hi)

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
        """Clears the observed range and colorbar claim. Must run before
        every draw(), or the colorbar disappears on the second render."""
        self._observed_min = None
        self._observed_max = None
        self._colorbar_claimed = False


class ColorSpec(PlotSpec):
    """Declares which Variable/Axes (DataPlot) or column (TablePlot) colors
    a plot, with an optional colormap and shared Scale. `variable_or_axes`
    may also be a VariableSpec, to switch which Variable colors the plot."""

    def __init__(
        self,
        variable_or_axes: "Variable | str | VariableSpec",
        cmap=None,
        scale: Scale | None = None,
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.cmap = cmap  # if not specified, plots resort to default cmap of theme
        self.scale = scale  # if set, shared with every other spec using the same Scale


class ControlSpec(PlotSpec):
    """Base class for a spec that reduces or partitions a plot's data by a
    dimension's current state (SliderSpec, FixedSpec, FacetSpec, TimeSpec)."""

    def __init__(self, init_state, n_dimensions, variable_or_axes) -> None:
        super().__init__(n_dimensions=n_dimensions, variable_or_axes=variable_or_axes)
        self._state = init_state

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


class SliderSpec(ControlSpec):
    """An interactive slider over one dimension's states."""

    def __init__(
        self,
        variable_or_axes,
        init_state,
        minimum,
        maximum,
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
    per value, instead of an interactive control."""

    def __init__(
        self,
        variable_or_axes,
        values=None,
        orientation: str = "col",
    ):
        assert orientation in ("row", "col"), "orientation must be 'row' or 'col'"
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes, init_state=0)
        self.orientation = orientation
        self.values = values

    def resolve(self, values):
        if self.values is None:
            self.values = list(values)
        if self._state is None:
            self._state = self.values[0]


class VariableSpec(ControlSpec):
    """Selects which of several distinct Variables currently feeds another
    spec's role - pass one anywhere a Variable is expected, e.g.
    `ColorSpec(VariableSpec([temperature, pressure]))`. `PlotSpec.
    variable_or_axes` transparently unwraps it to whichever candidate is
    currently selected, so every existing consumer (get_variable_slice,
    axis/colorbar titles, artists, ...) keeps working unchanged - selecting
    a variable is exactly the same "pick a slice of the FeatureAxes"
    operation a fixed Variable already describes, just made to react to
    `state` instead of staying fixed.

    Unlike SliderSpec/FacetSpec/TimeSpec, this is never passed via a Plot's
    own `controls=` - it isn't a whole-axis control for apply_controls() to
    reduce, so it never appears in Plot.controls the way those do.

    All `candidates` must share the same dim, so the role they feed stays
    valid (same required shape) regardless of which is selected.
    """

    def __init__(self, candidates: list[Variable], init_index: int = 0):
        assert len(candidates) >= 2, "VariableSpec needs at least 2 candidates to choose between."
        dims = {c.dim for c in candidates}
        assert len(dims) == 1, f"All candidates must share the same dim, got {dims}."
        self.candidates = candidates
        super().__init__(init_state=candidates[init_index], n_dimensions=1, variable_or_axes=None)

    @property
    def dim(self):
        return self.candidates[0].dim

    @property
    def name(self):
        # PlotSpec.name derives from self.variable_or_axes, which is None
        # here (this spec doesn't itself wrap a Variable - its candidates
        # do) - override with something a widget can actually label itself
        # with.
        return " / ".join(c.name for c in self.candidates)

    @property
    def state(self) -> Variable:
        return self._state

    @state.setter
    def state(self, value: int | Variable):
        if isinstance(value, int):
            value = self.candidates[value]
        assert value in self.candidates, f"{value!r} is not one of this spec's candidates."
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
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes, init_state=None)
        self.values = values
        self.duration = duration  # ms per frame while the Play button runs

    def resolve(self, values):
        if self.values is None:
            self.values = list(values)
        if self._state is None:
            self._state = self.values[0]
