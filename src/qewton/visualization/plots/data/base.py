"""Base class :class:`DataPlot` for plots of data described by a
DataConfiguration, and the binding of channels to references."""

import itertools
from dataclasses import dataclass

import numpy as np

from qewton.backends import resolve_backend
from qewton.config.axes import Axes, EllipsisAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry, Structure
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry
from qewton.visualization.axes import POSITION_CHANNELS, ControlAxis, Selector
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.color_channel import candidate_scales, unique
from qewton.visualization.plots.result import Result
from qewton.visualization.references import component_titles, title_of
from qewton.visualization.scales import ColorScale, Scale


def drawable_geometry(axes: list[Axes]) -> DiscreteGeometry:
    """The geometry whose points are the elements of the given axes.

    Args:
        axes: Axes of a DataConfiguration, none of them a FeatureAxes.

    Returns:
        DiscreteGeometry: The geometry of a single GeometryAxes; otherwise
            an IndexGridGeometry over the axes' shape (one point for no
            axes), whose coordinates are the indices.

    Raises:
        ValueError: If a GeometryAxes holds no discrete geometry.
    """
    if len(axes) == 1 and isinstance(axes[0], GeometryAxes):
        geometry = axes[0].geometry
        if not isinstance(geometry, DiscreteGeometry):
            raise ValueError(f"{type(geometry).__name__} is not a discrete geometry.")
        return geometry
    shape = tuple(dim.size for a in axes for dim in a.shape) or (1,)
    return IndexGridGeometry(Variable("index", len(shape)), shape)


def grid_values(domain: DiscreteGeometry, values: np.ndarray) -> tuple:
    """Values per point of a rectilinear 2D grid, arranged as an image.

    Args:
        domain: A geometry of :attr:`Structure.GRID` with two axes.
        values: One row per point (row-major over the domain's shape),
            shape ``(n,)`` or ``(n, k)``.

    Returns:
        tuple: The coordinates along x and along y, and the values of
            shape ``(ny, nx)`` or ``(ny, nx, k)``.
    """
    x_axis, y_axis = (domain.axis_components.index(k) for k in (0, 1))
    coordinates = domain.axis_coordinates()
    grid = values.reshape(*domain.shape, *values.shape[1:])
    order = (y_axis, x_axis, *range(2, grid.ndim))
    return coordinates[x_axis], coordinates[y_axis], grid.transpose(order)


def _check_axes(data, data_config: DataConfiguration) -> None:
    # Fails early on data a DataPlot cannot draw.
    for axes in data_config.axes:
        if isinstance(axes, EllipsisAxes) or any(d.size is None for d in axes.shape):
            raise ValueError("Data with an unresolved (ellipsis) shape cannot be drawn.")
    if tuple(data.shape) != tuple(data_config.shape):
        raise ValueError(
            f"The data has the shape {tuple(data.shape)}, its configuration "
            f"describes {tuple(data_config.shape)}."
        )
    features = [a for a in data_config.axes if isinstance(a, FeatureAxes)]
    if any(len(a.shape) != 1 for a in features):
        raise ValueError("Feature axes with more than one tensor axis are not supported.")
    if features and data_config.axes[-1] is not features[0]:
        raise ValueError("The feature axes must be the last axes of the data.")


def controlled_axes(data_config: DataConfiguration, control: ControlAxis) -> Axes:
    """The axes of `data_config` a control reduces.

    Args:
        data_config: A DataConfiguration.
        control: A control axis referencing an Axes of it or a Variable.

    Returns:
        Axes: Axes of one tensor axis of the domain: a batch axis, or a
            GeometryAxes whose whole geometry the control steps through.

    Raises:
        ValueError: If the reference is no such axes, e.g. a feature, a
            geometry of several tensor axes, or a component of a geometry
            that is no factor of it (e.g. the time of space-time points
            without a product structure).
    """
    reference = control.variable_or_axes
    axes = data_config.axes_of(reference)
    if isinstance(axes, FeatureAxes) or len(axes.shape) != 1:
        raise ValueError(
            f"{type(control).__name__} '{control.name}' must reduce one tensor axis "
            "of the domain, e.g. a batch axis or a one-dimensional geometry."
        )
    if isinstance(reference, Variable) and reference != axes.geometry.variable:
        raise ValueError(
            f"{type(control).__name__} '{control.name}' names a part of "
            f"{axes.geometry.variable.name}, not a factor of it: a control steps "
            "through a whole one-dimensional geometry or a factor of a product "
            "geometry."
        )
    return axes


def split_products(data, data_config: DataConfiguration, references: list) -> tuple:
    """Give every factor of a product geometry that a reference names
    tensor axes of its own (see
    :meth:`~qewton.config.axes.GeometryAxes.split`).

    Args:
        data: Tensor described by `data_config`, of any backend.
        data_config: Its DataConfiguration.
        references: Bound references: Variables, Axes, tuples of them,
            Selectors or None.

    Returns:
        tuple: The data reshaped and its configuration; both unchanged if no
            reference names a factor.
    """
    variables = []
    for reference in references:
        items = reference if isinstance(reference, tuple) else (reference,)
        for item in items:
            candidates = item.candidates if isinstance(item, Selector) else [item]
            variables += [c for c in candidates if isinstance(c, Variable)]
    split = data_config
    for axes in data_config.axes:
        factors = axes.geometry.factors if isinstance(axes, GeometryAxes) else None
        if factors is None:
            continue
        named = [v for v in variables if v in [f.variable for f in factors]]
        if named:
            split = split.replace(axes, axes.split(named))
    if split is data_config:
        return data, data_config
    return data.reshape(split.shape), split


@dataclass(frozen=True)
class Binding:
    """One channel bound to one component of a reference.

    Args:
        channel: The channel, e.g. ``"x"`` or ``"color"``.
        reference: A Variable or Axes of the plot's DataConfiguration, or a
            Selector of Variables.
        component: Index of the component within the reference, or None
            for all of its components.
        in_domain: Whether the reference lies in the domain D (geometry,
            batch or index axes) rather than in the values V (features).
    """

    channel: str
    reference: Variable | Axes | Selector
    component: int | None
    in_domain: bool

    @property
    def current(self) -> Variable | Axes:
        """The reference, or the chosen candidate of a Selector."""
        return (
            self.reference.selected
            if isinstance(self.reference, Selector)
            else self.reference
        )

    @property
    def title(self) -> str | None:
        """The name of the bound component."""
        reference = self.current
        if self.component is None:
            return title_of(reference)
        return component_titles(reference, n_components(reference))[self.component]


def n_components(reference: Variable | Axes) -> int:
    """The number of values a reference gives per point.

    Args:
        reference: A Variable or Axes.

    Returns:
        int: The geometry's dimension for a GeometryAxes, the Variable's
            dimension for a Variable, otherwise 1.
    """
    if isinstance(reference, GeometryAxes):
        return reference.geometry.dim
    if isinstance(reference, Variable):
        return reference.dim
    return 1


class DataPlot(Plot):
    """Draws one kind of mark from data described by a DataConfiguration.

    Positions are bound as ``position=`` (a geometry Variable, a
    GeometryAxes or BatchAxes, a feature with up to 3 components, or a
    tuple of one-component references; fills x, y, z in order), per
    channel as ``x=``, ``y=``, ``z=`` (one component each), or both for
    channels `position` leaves free, e.g. ``position=X, z=u`` for a 2D
    domain lifted by a value. The channels bound are x, y and possibly z.
    Where a binding comes from follows from its reference:

    - a position from the domain D: the drawn domain is the geometry (or
      batch axes) the positions belong to; every other dimension of D must
      be reduced by a control axis.
    - all positions from the values V: every point of the dimensions of D
      not reduced by a control is one element.

    On a product geometry (e.g. a space-time mesh in one GeometryAxes), a
    factor named by a binding or control gets tensor axes of its own, e.g.
    ``position=X, controls=[SliderAxis(T)]`` on data over T x X.

    A tuple of variables in ``y`` or ``z`` draws one copy per variable; an
    :class:`~qewton.visualization.axes.OverlayAxis` one copy per state.

    Args:
        data: Tensor described by `data_config`, of any backend.
        data_config: Its DataConfiguration.
        position, x, y, z: Position bindings.
        color: Reference or Selector for the color channel.
        color_scale: ColorScale of the color channel; a new one if None.
            Plot types whose color values are colors themselves take none.
            With a Selector: one ColorScale shared by the candidates (one
            range over all of them), a list with one per candidate, or None
            for a new one per candidate.
        controls: Control axes for the dimensions not drawn.
        title, label: As for
            :class:`~qewton.visualization.plots.base.Plot`.
        opacity: Opacity of every element.

    Raises:
        ValueError: At construction, if a binding names a channel the plot
            type does not have or a reference not in `data_config`, a
            dimension of D is neither drawn nor controlled, the number of
            positions or the drawn domain is not supported by the plot type,
            a required channel is missing, or `color_scale` is a list that
            does not give one scale per Selector candidate.

    Attributes:
        channels (tuple of str): Channels the plot type accepts (class
            attribute).
        required (tuple of str): Channels that must be bound (class
            attribute).
        results (dict): ``{(Structure, topological_dim): Result class}``
            for the drawn domains the plot type supports (class attribute).
        dims (tuple of int): Supported numbers of position components
            (class attribute).
        domain_positions (bool): Whether positions must come from the
            domain (class attribute).
        color_mapped (bool): Whether color values are mapped through a
            ColorScale; otherwise they are colors themselves, passed to
            :meth:`build` with all their components (class attribute).
        data, data_config: The data and its configuration, with the axes of
            a product geometry split into the factors the references name
            (see :func:`split_products`).
        domain (DiscreteGeometry): The drawn domain.
        result_class (type): The Result class the plot draws.
        n_positions (int): Number of bound position components.
        bindings (list of Binding): The position bindings, then the color
            binding (if any), of the first copy.
        color_scale (ColorScale or None): Scale of the color channel in the
            current states (of the chosen Selector candidate), None without
            a color binding.
        opacity (float or None): Opacity of every element.

    Notes:
        Subclasses declare the class attributes and implement :meth:`build`.
    """

    channels: tuple[str, ...] = ("x", "y", "z", "color")
    required: tuple[str, ...] = ()
    results: dict[tuple[Structure, int], type[Result]] = {}
    dims: tuple[int, ...] = (2, 3)
    domain_positions = False
    color_mapped = True

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        *,
        position=None,
        x=None,
        y=None,
        z=None,
        color: Variable | Selector | None = None,
        color_scale: ColorScale | list[ColorScale] | None = None,
        controls: tuple[ControlAxis, ...] | list[ControlAxis] = (),
        title: str | None = None,
        label: str | None = None,
        opacity: float | None = None,
    ) -> None:
        super().__init__(title=title, label=label, controls=controls)
        _check_axes(data, data_config)
        references = [position, x, y, z, color]
        references += [c.variable_or_axes for c in self.controls]
        self.data, self.data_config = split_products(data, data_config, references)
        self.opacity = opacity
        self._variants = self._position_variants(position, x, y, z)
        self._color = self._bind("color", color, None) if color is not None else None
        self._color_scales = self._candidate_scales(color, color_scale)
        self._check_channels()
        self._control_of = self._resolve_controls()
        self.domain = self._resolve_domain()
        self.result_class = self.result_type(self.domain)
        self._coordinates = self.domain.coordinates()

    # --- declarations and checks -------------------------------------------

    def result_type(self, domain: DiscreteGeometry) -> type[Result]:
        """The Result class the plot type draws on `domain`.

        Args:
            domain: The drawn domain.

        Returns:
            type: A subclass of Result, looked up in `results`.

        Raises:
            ValueError: If the plot type cannot draw on `domain`.
        """
        if not self.draws(domain):
            raise ValueError(
                f"{type(self).__name__} cannot draw on a {domain.structure.name.lower()} "
                f"domain of topological dimension {domain.topological_dim}"
                f" with {domain.dim} coordinates."
            )
        return self.results[(domain.structure, domain.topological_dim)]

    @classmethod
    def draws(cls, domain: DiscreteGeometry) -> bool:
        """Whether the plot type draws on `domain`, as declared in
        `results`.

        Args:
            domain: A drawn domain.

        Returns:
            bool: True if it does.
        """
        return (domain.structure, domain.topological_dim) in cls.results

    @classmethod
    def supports(cls, domain: DiscreteGeometry, n_positions: int) -> bool:
        """Whether the plot type draws on `domain` with `n_positions`
        position components, as declared in `results` and `dims`.

        Args:
            domain: A drawn domain.
            n_positions: Number of position components.

        Returns:
            bool: True if it does.
        """
        return n_positions in cls.dims and cls.draws(domain)

    def _check_channels(self) -> None:
        if len(self._variants[0]) not in self.dims:
            raise ValueError(
                f"{type(self).__name__} draws {' or '.join(map(str, self.dims))} "
                "position "
                f"components, got {len(self._variants[0])}."
            )
        channels = [b.channel for b in self._variants[0]]
        if channels != list(POSITION_CHANNELS[: len(channels)]):
            raise ValueError(
                f"Positions bind the channels {channels}; bind x, y and possibly z."
            )
        bound = set(channels)
        if self._color is not None:
            bound.add("color")
        unknown = bound - set(self.channels)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no channel {sorted(unknown)}.")
        missing = set(self.required) - bound
        if missing:
            raise ValueError(
                f"{type(self).__name__} needs the channel(s) {sorted(missing)}."
            )
        if self.domain_positions and not all(b.in_domain for b in self._variants[0]):
            raise ValueError(f"{type(self).__name__} draws positions of the domain only.")

    def _candidate_scales(self, color, color_scale) -> list[ColorScale]:
        # One ColorScale per candidate of the color binding (one for a
        # reference, none without a color binding or mapping).
        if not self.color_mapped:
            if color_scale is not None:
                raise ValueError(
                    f"{type(self).__name__} draws its color values as colors; "
                    "it takes no color_scale."
                )
            return []
        return candidate_scales(color, color_scale)

    # --- bindings ------------------------------------------------------------

    def _bind(self, channel: str, reference, component: int | None) -> Binding:
        if isinstance(reference, Selector):
            sources = {self._in_domain(c) for c in reference.candidates}
            if sources != {False}:
                raise ValueError("A Selector chooses between values (features) only.")
            return Binding(channel, reference, component, False)
        return Binding(channel, reference, component, self._in_domain(reference))

    def _in_domain(self, reference) -> bool:
        return not isinstance(self.data_config.axes_of(reference), FeatureAxes)

    def _position_variants(self, position, x, y, z) -> list[list[Binding]]:
        # One list of position bindings per copy from a tuple in y or z.
        per_channel = {"x": x, "y": y, "z": z}
        fixed = self._split_position(position) if position is not None else []
        taken = {b.channel for b in fixed}
        for channel, reference in per_channel.items():
            if reference is not None and channel in taken:
                raise ValueError(
                    f"position= binds {channel} already; bind each channel once."
                )
            if isinstance(reference, tuple) and channel == "x":
                raise ValueError("Only y and z take a tuple of variables.")
        options = [
            [
                (channel, r)
                for r in (reference if isinstance(reference, tuple) else (reference,))
            ]
            for channel, reference in per_channel.items()
            if reference is not None
        ]
        if sum(len(o) > 1 for o in options) > 1:
            raise ValueError("Only one channel takes a tuple of variables.")
        return [
            fixed
            + [
                self._bind_component(channel, reference)
                for channel, reference in combination
            ]
            for combination in itertools.product(*options)
        ]

    def _bind_component(self, channel: str, reference) -> Binding:
        binding = self._bind(channel, reference, None)
        if not isinstance(reference, Selector) and n_components(reference) != 1:
            raise ValueError(
                f"{channel}= takes one component, {reference} has "
                f"{n_components(reference)}; use position= for several."
            )
        return Binding(channel, reference, 0, binding.in_domain)

    def _split_position(self, position) -> list[Binding]:
        if isinstance(position, Selector):
            raise ValueError(
                "position= takes no Selector; bind a Selector of values per channel, "
                "e.g. x=Selector([u, p])."
            )
        if isinstance(position, tuple):
            if len(position) > len(POSITION_CHANNELS):
                raise ValueError(f"At most 3 positions, got {len(position)}.")
            return [
                self._bind_component(c, r) for c, r in zip(POSITION_CHANNELS, position)
            ]
        n = n_components(position)
        if n > len(POSITION_CHANNELS):
            raise ValueError(f"At most 3 positions, {position} has {n}.")
        in_domain = self._in_domain(position)
        return [
            Binding(c, position, k, in_domain)
            for k, c in enumerate(POSITION_CHANNELS[:n])
        ]

    # --- controls and domain -------------------------------------------------

    def _resolve_controls(self) -> dict[int, ControlAxis]:
        # Tensor axis of each control, with the control resolved to its size.
        control_of = {}
        for control in self.controls:
            axes = controlled_axes(self.data_config, control)
            dim = self._tensor_dims(axes)[0]
            if dim in control_of:
                raise ValueError(f"Two controls reduce the same axis {axes}.")
            coordinates = None
            if isinstance(axes, GeometryAxes):
                coordinates = drawable_geometry([axes]).coordinates()[:, 0]
            control.resolve(axes.shape[0].size, coordinates)
            control_of[dim] = control
        return control_of

    def _tensor_dims(self, axes: Axes) -> list[int]:
        offset = 0
        for candidate in self.data_config.axes:
            if candidate is axes:
                return list(range(offset, offset + len(candidate.shape)))
            offset += len(candidate.shape)
        raise ValueError(f"{axes} is not part of the data configuration.")

    def _resolve_domain(self) -> DiscreteGeometry:
        free = [
            axes
            for axes in self.data_config.axes
            if not isinstance(axes, FeatureAxes)
            and any(d not in self._control_of for d in self._tensor_dims(axes))
        ]
        positions = [b for b in self._variants[0] if b.in_domain]
        if not positions:
            return drawable_geometry(free)
        drawn = {id(self.data_config.axes_of(b.current)): b for b in positions}
        if len(drawn) > 1:
            raise ValueError(
                "Positions from the domain must belong to one geometry or axis."
            )
        axes = self.data_config.axes_of(positions[0].current)
        if axes not in free:
            raise ValueError(f"{axes} is drawn and reduced by a control at once.")
        others = [a for a in free if a is not axes]
        if others:
            raise ValueError(
                f"{type(self).__name__}: the dimension(s) {', '.join(map(str, others))} "
                "are neither drawn nor reduced by a control; add a control for them."
            )
        domain = drawable_geometry([axes])
        bound = sorted(b.component for b in positions)
        if bound != list(range(domain.dim)):
            raise ValueError(
                f"Positions must bind every coordinate of {axes} once; drawing part of a "
                "geometry is not supported."
            )
        return domain

    # --- copies --------------------------------------------------------------

    @property
    def _overlays(self) -> list[ControlAxis]:
        return [c for c in self._control_of.values() if c.copies]

    @property
    def n_copies(self) -> int:
        return len(self._variants) * int(np.prod([len(c.states) for c in self._overlays]))

    def _copies(self):
        # (position bindings, {overlay control: state}) per drawn copy.
        overlays = self._overlays
        for bindings in self._variants:
            for states in itertools.product(*(c.states for c in overlays)):
                yield bindings, dict(zip(overlays, states))

    # --- Plot interface ------------------------------------------------------

    @property
    def n_positions(self) -> int:
        return len(self._variants[0])

    @property
    def dim(self) -> int:
        return self.n_positions

    @property
    def selectors(self) -> list[Selector]:
        bindings = self._variants[0] + ([self._color] if self._color is not None else [])
        found = []
        for binding in bindings:
            if isinstance(binding.reference, Selector) and binding.reference not in found:
                found.append(binding.reference)
        return found

    @property
    def color_scale(self) -> ColorScale | None:
        if not self._color_scales:
            return None
        selector = self._color.reference
        index = selector.state if isinstance(selector, Selector) else 0
        return self._color_scales[index]

    @property
    def color_scales(self) -> list[ColorScale]:
        return unique(self._color_scales)

    @property
    def active_color_scales(self) -> list[ColorScale]:
        return [self.color_scale] if self.color_scale is not None else []

    @property
    def bindings(self) -> list[Binding]:
        return self._variants[0] + ([self._color] if self._color is not None else [])

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        """The position channels bound to components of one reference,
        e.g. x and y of ``position=X``; not those of different references,
        e.g. x and z of ``position=X, z=u``."""
        groups: dict[int, list[str]] = {}
        for binding in self._variants[0]:
            if binding.component is not None:
                groups.setdefault(id(binding.reference), []).append(binding.channel)
        largest = max(groups.values(), key=len, default=[])
        return tuple(largest) if len(largest) > 1 else ()

    @property
    def color_title(self) -> str | None:
        return self._color.title if self._color is not None else None

    def channels_of(self, selector: Selector) -> tuple[str, ...]:
        """See :meth:`~qewton.visualization.plots.base.Plot.channels_of`."""
        return tuple(b.channel for b in self.bindings if b.reference is selector)

    def axis_title(self, channel: str) -> str | None:
        """See :meth:`~qewton.visualization.plots.base.Plot.axis_title`."""
        titles = {b.title for v in self._variants for b in v if b.channel == channel}
        return titles.pop() if len(titles) == 1 else None

    def train_scales(self, axes: dict[str, Scale]) -> None:
        """See :meth:`~qewton.visualization.plots.base.Plot.train_scales`."""
        for bindings, overlay_states in self._trained_copies():
            positions, color, _ = self._collect(bindings, overlay_states)
            for k, binding in enumerate(bindings):
                axes[binding.channel].train(positions[:, k])
            if self.color_scale is not None:
                self.color_scale.train(color)

    def _trained_copies(self):
        # Every copy in every trained state of the controls and selectors;
        # restores their states afterwards.
        stepping = [c for c in self.controls if not c.copies] + self.selectors
        originals = [c.state for c in stepping]
        try:
            for states in itertools.product(*(c.trained_states for c in stepping)):
                for control, state in zip(stepping, states):
                    control.state = state
                yield from self._copies()
        finally:
            for control, state in zip(stepping, originals):
                control.state = state

    def evaluate(self) -> list[Result]:
        """See :meth:`~qewton.visualization.plots.base.Plot.evaluate`."""
        results = []
        for index, (bindings, overlay_states) in enumerate(self._copies()):
            positions, color, features = self._collect(bindings, overlay_states)
            result = self.build(positions, color, features)
            result.label = self._copy_label(bindings, overlay_states)
            result.opacity = self.opacity if self.opacity is not None else 1.0
            result.color_index = self.color_index + index
            if self.color_scale is not None:
                result.color_scale = self.color_scale
                result.color_title = self.color_title
            results.append(result)
        return results

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """The Result of one copy in one state, without the common fields
        (label, colors, opacity), which :meth:`evaluate` sets.

        Args:
            positions: Shape ``(n, dim)``, one row per point of `domain`
                (row-major over its shape).
            color: Color value per point, shape ``(n,)``; with all
                components, shape ``(n, k)``, if not `color_mapped`; None
                without a color binding.
            features: The values of every feature in the copy's state,
                shape ``(n, n_features)``, for plot types drawing further
                features.

        Returns:
            Result: An instance of :meth:`result_type`.
        """
        raise NotImplementedError

    # --- evaluation ----------------------------------------------------------

    def _collect(self, bindings: list[Binding], overlay_states: dict) -> tuple:
        # Positions (n, dim), color values (n,) or None, and the feature
        # matrix of one copy.
        features = self._features(overlay_states)
        columns = [self._channel_values(b, features) for b in bindings]
        positions = np.stack(columns, axis=1)
        color = None
        if self._color is not None:
            color = self.feature_values(self._color.current, features)
            if self.color_scale is not None:
                color = self.color_scale.reduce_components(color)
        return positions, color, features

    def _channel_values(self, binding: Binding, features: np.ndarray) -> np.ndarray:
        if binding.in_domain:
            return self._coordinates[:, binding.component]
        return self.feature_values(binding.current, features)[:, binding.component]

    def feature_values(self, variable: Variable, features: np.ndarray) -> np.ndarray:
        """The columns of a feature in a feature matrix.

        Args:
            variable: A Variable of the FeatureAxes.
            features: A feature matrix as passed to :meth:`build`.

        Returns:
            numpy.ndarray: Shape ``(n, variable.dim)``.
        """
        feature_axes = self.data_config.feature_axes
        return features[:, feature_axes.variables.get_slice(variable)]

    def _features(self, overlay_states: dict) -> np.ndarray:
        # The data in the current states, shape (n_points, n_features).
        index = []
        for dim in range(len(self.data.shape)):
            control = self._control_of.get(dim)
            if control is None:
                index.append(slice(None))
            else:
                index.append(overlay_states.get(control, control.state))
        reduced = np.asarray(resolve_backend(self.data).to_numpy(self.data[tuple(index)]))
        n_features = reduced.shape[-1] if self.data_config.feature_axes is not None else 1
        return reduced.reshape(-1, n_features).astype(float)

    def _copy_label(self, bindings: list[Binding], overlay_states: dict) -> str | None:
        parts = []
        if len(self._variants) > 1:
            varying = [
                b
                for k, b in enumerate(bindings)
                if len({id(v[k].reference) for v in self._variants}) > 1
            ]
            parts.append(varying[0].title if varying else None)
        else:
            parts.append(
                self.label if self.label is not None else self._quantity_title(bindings)
            )
        parts += [f"{c.name} = {c.label(s)}" for c, s in overlay_states.items()]
        parts = [p for p in parts if p]
        return ", ".join(parts) if parts else None

    def _quantity_title(self, bindings: list[Binding]) -> str | None:
        # The plotted quantity: the color, else the last position from V.
        if self._color is not None:
            return self._color.title
        values = [b for b in bindings if not b.in_domain]
        return values[-1].title if values else None
