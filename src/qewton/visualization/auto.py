"""Automatic choice of a plot type and its bindings from a
DataConfiguration.

:func:`auto_plot` decides in three steps, without knowing any geometry
class:

1. The drawn domain: the GeometryAxes not reduced by a control (1D factors
   of a product, or 1D GeometryAxes, are split off where needed), else
   the first other axis. Every axis not drawn gets a control.
2. The kind of value, from the quantities of the FeatureAxes' Variable.
3. The plot type: the first entry of :data:`CHOICES` for that kind whose
   type declares that it draws the domain (:meth:`DataPlot.supports`).
"""

import inspect
from dataclasses import dataclass
from enum import Enum, auto

from qewton.config.axes import Axes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.axes import POSITION_CHANNELS, ControlAxis, Selector, SliderAxis
from qewton.visualization.plots.data import (
    BarPlot,
    DataPlot,
    HeatmapPlot,
    LinePlot,
    ScatterPlot,
    SurfacePlot,
    VectorPlot,
    controlled_axes,
    drawable_geometry,
    n_components,
    split_products,
)
from qewton.visualization.scales import ColorScale, VectorScale

#: Highest dimension a drawn domain can have.
MAX_DRAWN_DIM = 3

# Bindings given as plot arguments, which override those of a Choice.
_CHANNELS = ("position", *POSITION_CHANNELS, "color", "vector")


class Kind(Enum):
    """What the plotted value is, relative to the drawn domain.

    Attributes:
        SCALAR: One value per point (or a Selector of such values).
        VECTOR: A vector with one component per coordinate of the domain.
        COORDINATES: The coordinates of the domain itself, e.g. sampled
            points or the grid of a data loader; drawn as the points of the
            domain (on the x axis for a 1D domain).
        VALUES: Values without a domain, drawn in value space; every point
            of the data is one element.
    """

    SCALAR = auto()
    VECTOR = auto()
    COORDINATES = auto()
    VALUES = auto()


class _Placeholder(Enum):
    DOMAIN = auto()
    VALUE = auto()
    FIRST = auto()
    SECOND = auto()


D, V = _Placeholder.DOMAIN, _Placeholder.VALUE
V0, V1 = _Placeholder.FIRST, _Placeholder.SECOND


@dataclass(frozen=True)
class Choice:
    """A plot type with its bindings; placeholders stand for the drawn
    domain (``D``) and the value (``V``, or ``V0``/``V1`` for two)."""

    plot_type: type[DataPlot]
    bindings: dict


#: Plot types per kind of value, in order of preference.
CHOICES: dict[Kind, list[Choice]] = {
    Kind.SCALAR: [
        Choice(HeatmapPlot, dict(position=D, color=V)),
        Choice(LinePlot, dict(x=D, y=V)),
        Choice(BarPlot, dict(x=D, y=V)),
        Choice(SurfacePlot, dict(position=D, color=V)),
        Choice(ScatterPlot, dict(position=D, color=V)),
    ],
    Kind.VECTOR: [Choice(VectorPlot, dict(position=D, vector=V, color=V))],
    Kind.COORDINATES: [Choice(ScatterPlot, dict(position=D))],
    Kind.VALUES: [
        Choice(ScatterPlot, dict(x=V0, y=V1)),
        Choice(ScatterPlot, dict(position=V)),
    ],
}


def auto_plot(
    data,
    data_config: DataConfiguration,
    plot_type: type[DataPlot] | None = None,
    *,
    value: Variable | Selector | None = None,
    controls: list[ControlAxis] = (),
    default_control: type[ControlAxis] = SliderAxis,
    color_scale: ColorScale | None = None,
    vector_scale: VectorScale | None = None,
    **plot_kwargs,
) -> DataPlot:
    """Build a plot for `data`, choosing its type and bindings from
    `data_config` (see the module documentation).

    Args:
        data: Tensor described by `data_config`, of any backend.
        data_config: Its DataConfiguration, with a FeatureAxes of a named
            Variable.
        plot_type: Restricts the choice to this plot type.
        value: What to draw: a Variable of the FeatureAxes, or a Selector of
            them (e.g. one shared by several plots); None for every
            quantity of the FeatureAxes.
        controls: Control axes for chosen dimensions, e.g.
            ``[TimeAxis(t)]``; a control on a factor of a product splits it
            off.
        default_control: Control axis class for every other dimension not
            drawn.
        color_scale: Passed to the plot's color channel.
        vector_scale: Passed to a VectorPlot.
        **plot_kwargs: Further arguments of the plot type, e.g. ``title``;
            bindings among them (e.g. ``z=u``) override the chosen ones.

    Returns:
        DataPlot: The plot.

    Raises:
        ValueError: If the data has no named feature Variable, the drawn
            domain or the value is ambiguous, no plot type (or not
            `plot_type`) fits, or `color_scale` / `vector_scale` is given
            for a plot without that channel.
    """
    controls = list(controls)
    explicit = {c: plot_kwargs.pop(c) for c in _CHANNELS if c in plot_kwargs}
    references = [c.variable_or_axes for c in controls]
    data, data_config = split_products(data, data_config, references)
    selector = value if isinstance(value, Selector) else None
    if selector is not None:
        quantities = list(selector.candidates)
    elif value is not None:
        quantities = [value]
    else:
        quantities = _quantities(data_config)
    drawn, data, data_config, leftover = _drawn_axes(
        data, data_config, controls, quantities
    )
    kind, value = _kind(drawn, quantities, selector)
    if kind is not Kind.VALUES:
        controls += [
            default_control(_control_reference(axes, data_config)) for axes in leftover
        ]

    choices = [
        c for c in CHOICES[kind] if plot_type is None or c.plot_type is plot_type
    ]
    domain = drawable_geometry([drawn]) if drawn is not None else None
    scales = {"color_scale": color_scale, "vector_scale": vector_scale}
    scales = {name: scale for name, scale in scales.items() if scale is not None}
    reasons = []
    for choice in choices:
        bindings = _bind(choice.bindings, drawn, value)
        if bindings is not None:
            bindings.update(explicit)
        reason = _mismatch(choice, bindings, drawn, domain, {**plot_kwargs, **scales})
        if reason is not None:
            reasons.append(f"{choice.plot_type.__name__} {reason}")
            continue
        return choice.plot_type(
            data, data_config, controls=controls, **bindings, **scales, **plot_kwargs
        )
    given = sorted([*plot_kwargs, *scales, *explicit])
    raise ValueError(
        f"No plot type draws {kind.name.lower()} values on {_name(drawn)}"
        + (f" with the arguments {', '.join(given)}" if given else "")
        + ": "
        + "; ".join(reasons or [f"{plot_type.__name__} draws no such values"])
        + ". Pass plot_type with its bindings, or construct the plot explicitly."
    )


def _quantities(data_config: DataConfiguration) -> list[Variable]:
    features = data_config.feature_axes
    variable = features.variables if features is not None else None
    if variable is None or variable.is_empty or variable.name is None:
        raise ValueError(
            "auto_plot needs a FeatureAxes of a named Variable to choose what to draw."
        )
    return variable.quantities


def _controlled(data_config: DataConfiguration, controls: list[ControlAxis]) -> list:
    return [controlled_axes(data_config, c) for c in controls]


def _drawn_axes(data, data_config, controls, quantities) -> tuple:
    # The drawn axes (None for values without a domain), the data and
    # configuration with products split where needed, and the axes left
    # over for default controls.
    controlled = _controlled(data_config, controls)
    free = [
        a
        for a in data_config.axes
        if not isinstance(a, FeatureAxes) and not any(a is c for c in controlled)
    ]
    geometries = [a for a in free if isinstance(a, GeometryAxes)]
    if len(geometries) == 1 and geometries[0].geometry.factors is not None:
        target = _target_dim(quantities)
        named = _factors_to_split(geometries[0], target)
        if named:
            data, data_config = split_products(data, data_config, named)
            return _drawn_axes(data, data_config, controls, quantities)
    if len(geometries) > 1:
        wide = [a for a in geometries if n_components(a) > 1]
        if len(wide) != 1:
            raise ValueError(
                f"auto_plot cannot choose which of {len(geometries)} geometries to "
                "draw; pass controls for all but one."
            )
        geometries = wide
    others = [a for a in free if not isinstance(a, GeometryAxes)]
    drawn = geometries[0] if geometries else (others[0] if others else None)
    leftover = [a for a in free if a is not drawn]
    return drawn, data, data_config, leftover


def _control_reference(axes: Axes, data_config: DataConfiguration) -> Variable | Axes:
    # A geometry is controlled through its Variable, so the control fits
    # every configuration of that Variable (e.g. shared by several plots),
    # unless the Variable names data values too.
    if not isinstance(axes, GeometryAxes) or axes.geometry.variable.is_empty:
        return axes
    variable = axes.geometry.variable
    return variable if data_config.axes_of(variable) is axes else axes


def _target_dim(quantities: list[Variable]) -> int:
    # The dimension the drawn domain may have for the plotted value.
    if len(quantities) == 1 and quantities[0].dim in (2, 3):
        return quantities[0].dim
    return MAX_DRAWN_DIM


def _factors_to_split(axes: GeometryAxes, target: int) -> list[Variable]:
    # The 1D factors to split off a product so that `target` dimensions
    # remain; none if it fits already.
    geometry = axes.geometry
    excess = geometry.dim - target
    if excess <= 0:
        return []
    one_d = [f.variable for f in geometry.factors if f.variable.dim == 1]
    if len(one_d) == excess:
        return one_d
    names = ", ".join(v.name for v in one_d)
    raise ValueError(
        f"{geometry.variable.name} has {geometry.dim} dimensions, {target} can be "
        f"drawn: pass a control for {excess} of its 1D factors ({names})."
    )


def _kind(
    drawn: Axes | None, quantities: list[Variable], selector: Selector | None
) -> tuple:
    # The kind of value and what V stands for: a Variable, a Selector (the
    # given one, else a new one), or the two quantities of a pair.
    on_geometry = isinstance(drawn, GeometryAxes)
    scalars = all(q.dim == 1 for q in quantities)
    single = quantities[0] if len(quantities) == 1 else None
    if on_geometry and single == drawn.geometry.variable:
        return Kind.COORDINATES, single
    if scalars and (selector is not None or on_geometry or len(quantities) != 2):
        if selector is None and single is None:
            selector = Selector(quantities)
        return Kind.SCALAR, single if selector is None else selector
    if scalars:
        return Kind.VALUES, quantities
    if single is not None and on_geometry and single.dim == n_components(drawn):
        return Kind.VECTOR, single
    if single is not None and not on_geometry and single.dim in (2, 3):
        return Kind.VALUES, single
    names = ", ".join(q.name for q in quantities)
    raise ValueError(
        f"auto_plot draws scalars, or one vector with a component per coordinate; "
        f"got {names}{_product_hint(drawn)}. Construct the plot explicitly."
    )


def _product_hint(drawn: Axes | None) -> str:
    # How to pass points of a product as such, if they look like one.
    if not isinstance(drawn, GeometryAxes) or drawn.geometry.factors is not None:
        return ""
    variable = drawn.geometry.variable
    if len(variable.quantities) < 2:
        return ""
    return (
        f" on points of {variable.name}; for a product domain, pass one "
        "GeometryAxes per factor"
    )


def _bind(template: dict, drawn: Axes | None, value) -> dict | None:
    # The bindings of a choice; None if a position channel would take a
    # reference with more than one component.
    substitutes = {D: drawn, V: value}
    if isinstance(value, list):
        substitutes.update({V0: value[0], V1: value[1]})
    bindings = {}
    for channel, placeholder in template.items():
        if placeholder not in substitutes or substitutes[placeholder] is None:
            return None
        reference = substitutes[placeholder]
        if isinstance(reference, list):
            return None
        if channel in POSITION_CHANNELS and _n_reference(reference) != 1:
            return None
        bindings[channel] = reference
    return bindings


def _n_reference(reference) -> int:
    if isinstance(reference, Selector):
        return 1
    return n_components(reference)


def _n_positions(bindings: dict) -> int:
    n = n_components(bindings["position"]) if "position" in bindings else 0
    return n + sum(channel in bindings for channel in POSITION_CHANNELS)


def _name(drawn: Axes | None) -> str:
    # The drawn domain as named in messages.
    if drawn is None:
        return "no domain"
    if isinstance(drawn, GeometryAxes) and not drawn.geometry.variable.is_empty:
        return f"the geometry of {drawn.geometry.variable.name}"
    return f"{type(drawn).__name__.removesuffix('Axes').lower()} axes"


def _mismatch(choice, bindings, drawn, domain, arguments: dict) -> str | None:
    # Why a choice cannot draw the value with the given arguments; None if
    # it can.
    plot_type = choice.plot_type
    if bindings is None:
        if drawn is not None and n_components(drawn) > 1:
            return (
                f"puts the domain on one axis, but it has {n_components(drawn)} "
                "coordinates"
            )
        return "needs two values to draw against each other"
    accepted = inspect.signature(plot_type).parameters
    unknown = sorted(name for name in arguments if name not in accepted)
    if unknown:
        return f"takes no {', '.join(unknown)}"
    for name, channel in (("color_scale", "color"), ("vector_scale", "vector")):
        if name in arguments and channel not in bindings:
            return f"binds no {channel} for {name}"
    if domain is None:
        return None
    n_positions = _n_positions(bindings)
    if n_positions not in plot_type.dims:
        dims = " or ".join(map(str, plot_type.dims))
        return f"draws {dims} position components, not {n_positions}"
    if not plot_type.draws(domain):
        return (
            f"draws no {domain.structure.name.lower()} domain of topological "
            f"dimension {domain.topological_dim}"
        )
    return None
