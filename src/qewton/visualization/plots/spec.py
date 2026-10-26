"""Specs that map plot roles onto data, plus shared color scales and
controls."""

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
    """Return the GeometryAxes of `data_config` that a plot draws.

    These are all :class:`~qewton.config.axes.GeometryAxes` except those a
    control steps through, e.g. the time axis of space-time data with a
    :class:`TimeSpec` on it.

    Args:
        data_config: The configuration of the data.
        controls: The plot's control specs.

    Returns:
        GeometryAxes or list of GeometryAxes or None: The single drawn
            GeometryAxes, a list if there are several, None if there is none -
            the same convention as ``DataConfiguration.geometry_axes``.
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
    """Base class declaring how a plot maps one role onto its input.

    A role (x, y, color, vector, a control, ...) is mapped onto a
    :class:`~qewton.config.variables.Variable` or
    :class:`~qewton.config.axes.Axes` for a
    :class:`~qewton.visualization.plots.data.base.DataPlot`, or onto a
    column name for a
    :class:`~qewton.visualization.plots.table.base.TablePlot`.

    Args:
        n_dimensions: Number of dimensions the role covers.
        variable_or_axes: The Variable, Axes or column name filling the role,
            or a :class:`SelectorSpec` choosing between several candidates.
            None if it is resolved later, e.g. for a control built before its
            axis is known.

    Attributes:
        n_dimensions (int): Number of dimensions the role covers.
        variable_or_axes (Variable or Axes or str or None): The Variable, Axes
            or column name filling the role. A :class:`SelectorSpec` given at
            construction is unwrapped to its currently selected candidate.
            Settable.
        embedded_selector_spec (SelectorSpec or None): The SelectorSpec given
            as `variable_or_axes`, or None.
        name (str): Name of the named Variable (also the single coordinate
            Variable of a GeometryAxes), else ``str(variable_or_axes)``.
        math_name (str): The Variable's TeX math-mode name, used for axis and
            colorbar titles; the plain `name` for a spec that names no Variable
            (an Axes or a column name).
    """

    def __init__(
        self, n_dimensions: int, variable_or_axes: Variable | Axes | str | None
    ) -> None:
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
        return (
            self._variable_or_axes
            if isinstance(self._variable_or_axes, SelectorSpec)
            else None
        )

    @property
    def _named_variable(self) -> "Variable | None":
        """The Variable `variable_or_axes` names: itself, or the coordinate
        Variable of a GeometryAxes with exactly one component. Else None."""
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
        variable = self._named_variable
        return variable.math_name if variable is not None else self.name

    @staticmethod
    def get_slice(variable_or_axes, data_config: DataConfiguration):
        """Locate a Variable or Axes in the dimensions of `data_config`.

        Searches from the front of ``data_config.axes`` first and, if that
        fails (e.g. because of an ellipsis), from the back.

        Args:
            variable_or_axes (Variable or Axes): The Variable or Axes to
                locate.
            data_config: The configuration of the data.

        Returns:
            tuple: ``(axis_slc, entry_slc)``, where

            - ``axis_slc`` (int or slice) is the data dimension(s) it
              occupies. Negative when found from the back.
            - ``entry_slc`` (slice or None) is the entries within that
              dimension for a Variable of a FeatureAxes, else None.

        Raises:
            ValueError: If `variable_or_axes` is not part of `data_config`.
        """
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
        """Normalize an axis result of :meth:`get_slice` to one dimension.

        Args:
            axis_slc: An int, or a slice covering exactly one dimension (e.g.
                one child Variable of a multi-dimensional GeometryAxes).

        Returns:
            int: The dimension index.

        Raises:
            AssertionError: If `axis_slc` is a slice covering more than one
                dimension.
        """
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
                        # Variable.get_slice() returns slice(None) for the
                        # geometry's own root variable; substitute numeric
                        # bounds so they can be offset by `counter`.
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
    """Declares a single structural axis role (e.g. a line plot's `x`).

    Args:
        variable_or_axes: The Variable or Axes filling the role, or a
            :class:`SelectorSpec` to switch between several Variables.
        log_scale: Whether the axis is drawn logarithmically.

    Attributes:
        log_scale (bool): Whether the axis is drawn logarithmically.
    """

    def __init__(
        self, variable_or_axes: "Variable | Axes | SelectorSpec", log_scale: bool = False
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.log_scale = log_scale


class VectorSpec(PlotSpec):
    """Declares a 2D or 3D vector-valued role (e.g. a quiver plot's arrows).

    How the arrows are colored is a separate role, the plot's own `color`
    (a :class:`ColorSpec`), which defaults to the vector magnitude.

    Args:
        variable_or_axes (Variable or Axes or SelectorSpec): The 2D or 3D
            Variable or Axes the vector components come from, or a
            :class:`SelectorSpec` switching between candidates of equal
            dimension.
        scale (float, optional): Multiplier applied to every vector's length.
            None autoscales: the longest vector in the whole data is drawn
            about as long as the typical spacing between drawn arrows.
        normalize (bool, optional): If True, normalizes each vector to unit
            length before applying `scale`.
        subsample (int, optional): Draw only every `subsample`-th vector.
            Applied after `normalize` and `scale`. Must be at least 1.

    Raises:
        AssertionError: If the dimension is not 2 or 3, or `subsample` is below
            1.

    Attributes:
        scale (float or None): Length multiplier, or None for autoscaling.
        normalize (bool): Whether vectors are normalized to unit length.
        subsample (int): Draw every `subsample`-th vector only.
    """

    def __init__(
        self,
        variable_or_axes,
        scale=None,
        normalize=False,
        subsample=1,
    ):
        dim = variable_or_axes.dim
        assert dim in [2, 3], "VectorSpec only supports 2D or 3D variables"
        super().__init__(dim, variable_or_axes)
        self.scale = scale
        self.normalize = normalize
        assert subsample >= 1, "subsample must be >= 1"
        self.subsample = subsample


class Scale:
    """Shared value range for one or more ColorSpecs.

    All plots whose :class:`ColorSpec` references the same Scale instance
    train it together during ``Figure.draw()`` and are drawn with one
    common color range and a single colorbar.

    Args:
        vmin: Explicit lower bound. Takes precedence over the observed one.
        vmax: Explicit upper bound. Takes precedence over the observed one.
        symmetric: If True, the range is centered on zero.

    Attributes:
        vmin (float or None): Explicit lower bound.
        vmax (float or None): Explicit upper bound.
        symmetric (bool): Whether the range is centered on zero.
        colorbar_cell (tuple of int or None): ``(row, col)`` (1-indexed) of the
            rightmost grid cell among all plots referencing this Scale, where
            its colorbar is placed. Set once per draw by the Figure; None
            outside a grid or before that.
        range (tuple of float or None): Merged ``(lo, hi)``, explicit bounds
            taking precedence over observed values and made symmetric if
            requested. None if neither bound is known.
    """

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = False,
    ) -> None:
        self.vmin = vmin
        self.vmax = vmax
        self.symmetric = symmetric
        self._observed_min: float | None = None
        self._observed_max: float | None = None
        self._colorbar_claimed = False
        self.colorbar_cell: tuple[int, int] | None = None

    def observe(self, values) -> None:
        """Widen the observed range to cover `values`, ignoring NaN.

        Args:
            values (array_like): Color values of one plot. Empty input is
                ignored.
        """
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
        lo = self.vmin if self.vmin is not None else self._observed_min
        hi = self.vmax if self.vmax is not None else self._observed_max
        if lo is None or hi is None:
            return None
        if self.symmetric:
            bound = max(abs(lo), abs(hi))
            lo, hi = -bound, bound
        return lo, hi

    def claim_colorbar(self) -> bool:
        """Claim this Scale's colorbar for the calling artist.

        Returns:
            bool: True for the first call since the last :meth:`reset`, False
                afterwards, so only one colorbar is shown per Scale.
        """
        if self._colorbar_claimed:
            return False
        self._colorbar_claimed = True
        return True

    def reset(self) -> None:
        """Clear the observed range, colorbar claim, and colorbar cell.

        Must run before every draw, otherwise the colorbar is already
        claimed and missing on the next render.
        """
        self._observed_min = None
        self._observed_max = None
        self._colorbar_claimed = False
        self.colorbar_cell = None


class ColorSpec(PlotSpec):
    """Declares what colors a plot, with an optional colormap and Scale.

    Args:
        variable_or_axes: The Variable or Axes (data plots) or column name
            (table plots) whose values color the plot, or a
            :class:`SelectorSpec` to switch between several.
        cmap (ColormapRole or QewtonColormap or str or list, optional): A
            qewton colormap role (e.g. ``cmaps.diverging``, resolved against
            the Figure's theme at draw time), a concrete qewton colormap (e.g.
            ``cmaps.meadow``), or a renderer-native colormap passed through
            as-is. None uses the theme's sequential colormap. See
            :meth:`~qewton.visualization.themes.base.Theme.resolve_cmap`.
        scale: Shared value range; all specs with the same :class:`Scale` get
            one common range.

    Attributes:
        cmap (ColormapRole or QewtonColormap or str or list or None): The
            colormap.
        scale (Scale or None): The shared value range.
    """

    def __init__(
        self,
        variable_or_axes: "Variable | str | SelectorSpec",
        cmap=None,
        scale: Scale | None = None,
    ) -> None:
        super().__init__(n_dimensions=1, variable_or_axes=variable_or_axes)
        self.cmap = cmap
        self.scale = scale


class ControlSpec(PlotSpec):
    """Base class for a spec that reduces or partitions a plot's data by the
    current state of one dimension.

    Args:
        init_state (object, optional): Initial state, usually an index along
            the dimension.
        n_dimensions: Number of dimensions the control covers.
        variable_or_axes (Variable or Axes or str or SelectorSpec, optional):
            The dimension (or table column) the control acts on. None if it is
            resolved later, e.g. by
            :func:`~qewton.visualization.auto.auto_plot`.

    Attributes:
        state (object): The current state. Settable.
        coordinates (list of float or None): Physical coordinate of each state
            (e.g. the time of each time step), indexed by state; None if the
            dimension has none.

    Notes:
        The owning plot calls :meth:`resolve` with the selectable states of the
        dimension. Subclasses override :meth:`resolve` to fill in their
        defaults; the base implementation does nothing. Subclasses may override
        the `state` property, e.g. to forbid or validate changes.

    See Also:
        :class:`SliderSpec`, :class:`FixedSpec`, :class:`FacetSpec`,
        :class:`TimeSpec`, :class:`SelectorSpec`
    """

    def __init__(
        self, init_state=None, n_dimensions: int = 1, variable_or_axes=None
    ) -> None:
        super().__init__(n_dimensions=n_dimensions, variable_or_axes=variable_or_axes)
        self._state = init_state
        self.coordinates: list[float] | None = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def resolve(self, values):
        """Fill in settings left None from the selectable states.

        Explicitly set values are never overwritten. The base
        implementation does nothing.

        Args:
            values (iterable): The selectable states of the control's
                dimension, computed by the owning plot: ``range(size)`` for a
                data plot, the sorted unique column values for a table plot.
        """

    def set_coordinates(self, coordinates) -> None:
        """Set the physical coordinate of each state, unless already set.

        Args:
            coordinates (array_like or None): One coordinate per state, e.g.
                the time of each time step. None is ignored.
        """
        if self.coordinates is None and coordinates is not None:
            self.coordinates = [float(c) for c in coordinates]

    def coordinate(self, state) -> float | None:
        """Return the physical coordinate of `state`, or None if unknown."""
        if self.coordinates is None:
            return None
        return self.coordinates[state]

    def label(self, state) -> str:
        """Return the display text for `state`.

        Returns:
            str: The coordinate of `state` (4 significant digits) if known,
                else ``str(state)``.
        """
        coordinate = self.coordinate(state)
        return str(state) if coordinate is None else f"{coordinate:.4g}"

    def state_at(self, coordinate: float):
        """Return the state whose coordinate is closest to `coordinate`.

        Requires `coordinates` to be set.
        """
        return int(np.argmin(np.abs(np.asarray(self.coordinates) - coordinate)))


class SliderSpec(ControlSpec):
    """An interactive slider over one dimension's states.

    Args:
        variable_or_axes (Variable or Axes or str, optional): The dimension (or
            table column) the slider acts on.
        init_state (int, optional): Initial state. Defaults to `minimum`.
        minimum (int, optional): Lower slider bound. Defaults to the smallest
            selectable state.
        maximum (int, optional): Upper slider bound. Defaults to the largest
            selectable state.
        step (int, optional): Slider step.
        marks (optional): Slider marks passed to the interactive widget.

    Attributes:
        minimum (int or None): Lower slider bound, filled in by
            :meth:`resolve`.
        maximum (int or None): Upper slider bound, filled in by
            :meth:`resolve`.
        step (int): Slider step.
        marks (object): Slider marks.
    """

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
        """Fill in missing bounds from `values` and the state from
        `minimum`."""
        if self.minimum is None or self.maximum is None:
            values = list(values)
            self.minimum = self.minimum if self.minimum is not None else min(values)
            self.maximum = self.maximum if self.maximum is not None else max(values)
        if self._state is None:
            self._state = self.minimum


class FixedSpec(ControlSpec):
    """Selects one fixed index of a dimension.

    Takes the parameters of :class:`ControlSpec`; `init_state` is the fixed
    index.

    Raises:
        ValueError: When `state` is set.

    Attributes:
        state (object): The fixed index.
    """

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        raise ValueError("Cannot set state of FixedAxis. It is fixed.")


class FacetSpec(ControlSpec):
    """Splits a dimension into a grid of subplot panels, one per value.

    Args:
        variable_or_axes (Variable or Axes or str, optional): The dimension (or
            table column) to facet over.
        values (list, optional): States to draw a panel for. Defaults to every
            selectable state.
        orientation: ``"row"`` or ``"col"``: whether panels are laid out as
            rows or columns.
        labels: Panel titles, one per value.

    Raises:
        AssertionError: If `orientation` is invalid, or (in :meth:`resolve`)
            the number of `labels` does not match the number of `values`.

    Attributes:
        orientation (str): ``"row"`` or ``"col"``.
        values (list or None): States drawn as panels, filled in by
            :meth:`resolve`.
        labels (list of str or None): Panel titles.
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
        """Fill in missing `values` and check `labels` against them."""
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
    """A single-select control choosing which candidate fills another
    spec's role.

    Passed in place of a Variable to another spec, e.g.
    ``ColorSpec(SelectorSpec([temperature, pressure]))``, never via a
    plot's `controls`. :attr:`PlotSpec.variable_or_axes` unwraps it to the
    selected candidate. Rendered as one dropdown (Dash) or one button menu
    (Plotly) per instance.

    Args:
        candidates: At least two Variables or column names of equal dimension.
            Kept as given; strings are not turned into Variables.
        init_index: Index of the initially selected candidate.

    Raises:
        AssertionError: If there are fewer than two candidates, their
            dimensions differ, or `state` is set to something that is not a
            candidate.

    Attributes:
        candidates (list of Variable or list of str): The candidates.
        dim (int): Common dimension of the candidates.
        name (str): The candidate names joined by ``" / "``, used as widget
            label.
        state (Variable or str): The selected candidate. Can be set to a
            candidate or to its index.
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
        """Return a candidate's dimension: ``.dim`` for a Variable, 1 for a
        column name."""
        return candidate.dim if isinstance(candidate, Variable) else 1

    @staticmethod
    def candidate_name(candidate: "Variable | str") -> str:
        """Return a candidate's display name: ``.name`` for a Variable, the
        string itself for a column name."""
        return candidate.name if isinstance(candidate, Variable) else candidate

    @property
    def dim(self):
        return self.candidate_dim(self.candidates[0])

    @property
    def name(self):
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
    """Declares that a dimension or column advances over animation frames.

    A renderer that materializes frames up front (e.g. Plotly) builds one
    frame per value in `values`, plus play/pause controls and a frame
    slider. There is no interactive Dash widget for it.

    Args:
        variable_or_axes (Variable or Axes or str): The dimension (or table
            column) to animate over.
        values (list, optional): States to show as frames. Defaults to every
            selectable state.
        duration (int, optional): Average display time per frame in
            milliseconds.

    Attributes:
        values (list or None): States shown as frames, filled in by
            :meth:`resolve`.
        duration (int): Average display time per frame in milliseconds; see
            :meth:`frame_weights`.
    """

    def __init__(self, variable_or_axes, values=None, duration=500):
        super().__init__(
            n_dimensions=1, variable_or_axes=variable_or_axes, init_state=None
        )
        self.values = values
        self.duration = duration

    def resolve(self, values):
        """Fill in missing `values` and set the state to the first one."""
        if self.values is None:
            self.values = list(values)
        if self._state is None:
            self._state = self.values[0]

    def frame_weights(self) -> list[float]:
        """Return the relative display time of each frame, with mean 1.

        With `coordinates` (e.g. physical time), each frame is shown in
        proportion to the time until the next frame; the last frame gets
        the mean step. Without coordinates, or if all coordinates are
        equal, every frame is shown equally long.

        Returns:
            list of float: One weight per entry of `values`.
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
