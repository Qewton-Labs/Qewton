import numpy as np

from qewton.config.axes import Axes, EllipsisAxes, EllipsisDim, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable


class PlotSpec:
    def __init__(self, n_dimensions: int, variable_or_axes: Variable | Axes | str) -> None:
        # str is TablePlot's case: a column key, no Variable/Axes involved -
        # see "variable_or_axes keeps its name" in the implementation plan.
        self.n_dimensions = n_dimensions
        self.variable_or_axes = variable_or_axes

    @property
    def name(self):
        if isinstance(self.variable_or_axes, Variable):
            return self.variable_or_axes.name
        return str(self.variable_or_axes)

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
    def __init__(
        self, variable_or_axes: Variable | Axes, log_scale: bool = False
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale

        self.coordinates: np.ndarray | None = None


class VectorSpec(PlotSpec):
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
        # Naive every-n-th-arrow decimation - a pure display decision (which
        # arrows to draw, not what the field is), so it stays a plot-local
        # render parameter rather than node work (implementation plan,
        # section 1's node-layer criterion only covers operations that
        # produce a new dataset; this only changes what's shown).
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
    def __init__(
        self, variable_or_axes: Variable | str, cmap=None, scale: Scale | None = None
    ) -> None:
        # variable_or_axes is a Variable for DataPlot families, a column key
        # (str) for TablePlot ones - see PlotSpec's "variable_or_axes keeps
        # its name" note in the implementation plan, section 4.
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.cmap = cmap  # if not specified, plots resort to default cmap of theme
        self.scale = scale  # if set, shared with every other spec using the same Scale


class ControlSpec(PlotSpec):
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
        state - from `values`, the list of selectable states. Explicit values
        set by the user are never overwritten. `values` is computed and
        handed in by the owning family (DataPlot: range(data.shape[dim]);
        TablePlot: sorted(unique(column.values))) - the spec itself no longer
        touches data_config/data, so the same ControlSpec subclasses serve
        every input family."""


class SliderSpec(ControlSpec):
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
    # selects just one fixed index
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        raise ValueError("Cannot set state of FixedAxis. It is fixed.")


class FacetSpec(ControlSpec):
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


class TimeSpec(ControlSpec):
    """Declares that a dimension/column advances over animation frames,
    rather than being an interactive widget (SliderSpec) or a grid of panels
    (FacetSpec) - the third way a ControlSpec's declared state can be driven,
    same "declared once, applied differently" split as every other control
    (section 6).

    Backends that materialize frames up front (Plotly) build one frame per
    value in `values`, plus play/pause + a frame slider - genuinely backend
    vocabulary, so Figure.draw() delegates it to Renderer.animate() once the
    normal two-pass draw is done, the same way Renderer.setup() owns
    subplot-grid construction. Does not get an interactive Dash widget,
    unlike SliderSpec - DashApplication only builds those for SliderSpec.
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
