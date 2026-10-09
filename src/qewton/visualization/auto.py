from qewton.config.axes import Axes, EllipsisAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.data.curve import LinePlot, PathPlot
from qewton.visualization.plots.data.grid import EmbeddedGridPlot, HeatmapPlot, QuiverPlot
from qewton.visualization.plots.data.mesh import MeshFieldPlot, MeshVectorPlot
from qewton.visualization.plots.data.points import PointCloudPlot
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.product_view import product_view
from qewton.visualization.plots.spec import (
    ColorSpec,
    Scale,
    ControlSpec,
    SliderSpec,
    SelectorSpec,
    VectorSpec,
    drawn_geometry_axes,
)


def auto_plot(
    data, data_config: DataConfiguration, plot_type: type[Plot] | None = None, **kwargs
) -> Plot:
    """Builds a Plot for `data`, choosing a sensible type and its required
    roles (x/y/color/vector) from `data_config`'s axes alone.

    Data on a product geometry (e.g. a meshed space-time domain, or the
    output of a ProductSampler) is first restructured by product_view(),
    with or without `plot_type`: each factor whose Variable a control in
    `controls=` is bound to, e.g. `[TimeSpec(T)]`, becomes an axis of its
    own, and the remaining factors are drawn as one geometry. If those
    would exceed 3 dimensions, the 1-dimensional factors get a default
    control instead, as long as that choice is unambiguous.

    With `plot_type=None` (default):
        - A GeometryAxes wrapping a MeshGeometry, plus a scalar (dim=1)
          FeatureAxes variable, becomes a MeshFieldPlot colored by it; a
          vector one (dim matching the mesh's own dimension) becomes a
          MeshVectorPlot.
        - A GeometryAxes wrapping a GridGeometry (a parametric grid) is
          dispatched on its discretization_points' own coordinate count
          (coord_dim), not on how many structural grid axes it has: 2
          (index coordinates only) becomes a HeatmapPlot for a scalar
          variable or a 2-component QuiverPlot for a vector; 3 (embedded in
          3D space) becomes an EmbeddedGridPlot or a 3-component QuiverPlot.
          More than two structural grid axes get a default SliderSpec each
          on every axis but the last two, which stay as the drawn grid;
          exactly one axis becomes a LinePlot instead.
        - No GeometryAxes, plus a FeatureAxes whose Variable has exactly
          one scalar leaf, becomes a LinePlot - the first remaining axis is
          its domain (`x`).
        - No GeometryAxes, plus a FeatureAxes whose Variable has exactly
          two scalar leaves, becomes a ScatterPlot (one leaf per x/y); any
          remaining axes are left alone, since ScatterPlot already flattens
          them into its implicit samples axis.
        - Any axis besides the ones just described - an extra batch/step
          axis alongside a GeometryAxes, or beyond a LinePlot's domain -
          gets a default SliderSpec, unless a control for it was already
          passed in `controls=`. Without one, the constructed Plot would
          fail at evaluate() time asking for exactly this.
        - Anything else raises ValueError with a specific explanation,
          rather than guessing - including cases with an equally valid but
          differently-drawn alternative (BarPlot instead of LinePlot,
          PathPlot instead of ScatterPlot, MeshSurfacePlot instead of
          MeshFieldPlot for a scalar on a 2D mesh), and a FeatureAxes
          bundling multiple distinct named variables (e.g. temperature
          *and* pressure, not one variable's auto-expanded components):
          those need an explicit `plot_type` and role.

    With an explicit `plot_type`, this is a plain pass-through -
    `plot_type(data, data_config, **kwargs)` - no auto-selection happens,
    and `**kwargs` must supply that type's required roles directly, exactly
    as if it were constructed directly - `controls=` then has to be a list.

    `scale=` (a Scale) sets the color range of the ColorSpec auto_plot
    creates, e.g. `Scale(vmin=0.0, vmax=1.0)` for one fixed colorbar over
    every slider state and animation frame.

    Raises:
        ValueError: If part of `controls=` would be ignored: a dict key, or
            a single ControlSpec instance, that no axis of the data takes a
            control for - or, with an explicit `plot_type`, `controls=`
            given as anything but a list. Also if `scale=` is given but the
            chosen plot has no color, or together with an explicit
            `plot_type`.
    """
    scale = kwargs.pop("scale", None)
    requested_controls = kwargs.get("controls")
    data, data_config, controls = product_view(data, data_config, requested_controls)
    if controls is not None:
        kwargs["controls"] = controls

    if plot_type is not None:
        if scale is not None:
            raise ValueError(
                "With an explicit plot_type, pass the Scale in its ColorSpec, "
                "e.g. color=ColorSpec(u, scale=scale), not as scale=."
            )
        if controls is not None and not isinstance(controls, list):
            raise ValueError(
                f"With an explicit plot_type, controls has to be a list of "
                f"ControlSpec instances, not {controls!r}."
            )
        return plot_type(data, data_config, **kwargs)

    explicit_controls = kwargs.get("controls")
    geometry_axes = drawn_geometry_axes(
        data_config, explicit_controls if isinstance(explicit_controls, list) else []
    )
    if isinstance(geometry_axes, list):
        raise ValueError(
            f"{data_config} has multiple GeometryAxes without a control - "
            "auto_plot can't pick one. Construct a Plot explicitly."
        )
    if geometry_axes is not None:
        plot = _auto_geometry_plot(data, data_config, geometry_axes, scale, **kwargs)
    else:
        plot = _auto_flat_plot(data, data_config, **kwargs)
    _check_controls_used(requested_controls, plot)
    color = getattr(plot, "color", None)
    if scale is not None and (color is None or color.scale is not scale):
        raise ValueError(
            f"scale= is not used, since the chosen {type(plot).__name__} has "
            "no color."
        )
    return plot


def _check_controls_used(controls, plot: Plot) -> None:
    """Raises if a dict entry or a single ControlSpec instance passed as
    `controls=` did not end up as one of `plot`'s controls, instead of
    silently ignoring it. A ControlSpec class only names the type for
    whatever axes are left over, so it is never unused."""

    def bound_to(control, key) -> bool:
        target = control.variable_or_axes
        return target is key or (isinstance(key, Variable) and target == key)

    taken = [c.name for c in plot.controls]
    hint = (
        "To bind a control to a Variable, e.g. the time of a space-time "
        "domain, pass a list such as controls=[TimeSpec(T)]."
    )
    if isinstance(controls, dict):
        unused = [k for k in controls if not any(bound_to(c, k) for c in plot.controls)]
        if unused:
            raise ValueError(
                f"controls has entries for {unused}, but they are not axes of "
                "the data left over for a control, so they would be ignored. "
                f"Controls in use: {taken}. {hint}"
            )
    elif isinstance(controls, ControlSpec):
        if not any(c is controls for c in plot.controls):
            raise ValueError(
                f"controls={controls.name} is not used, since no axis of the "
                f"data is left over for it. Controls in use: {taken}. {hint}"
            )


def is_curve_like(plot: Plot) -> bool:
    """Whether `plot` draws on one shared 2D x/y axes the way a LinePlot/
    PathPlot does, so several of them compose safely in one Overlay.

    A PointCloudPlot qualifies only for a 1D domain (drawn on a plain 2D
    x/y=0 axes) - a 2D/3D point cloud's axes are spatial coordinates, not
    domain-vs-value, and don't share meaning with a curve's axes; mesh/grid
    field plots are excluded for the same reason.
    """
    if isinstance(plot, (LinePlot, PathPlot)):
        return True
    if isinstance(plot, PointCloudPlot):
        return plot.coordinate_dim == 1
    return False


def _require_named_variable(
    data_config: DataConfiguration, feature_axes: FeatureAxes | None
) -> Variable:
    if feature_axes is None:
        raise ValueError(
            f"{data_config} has no FeatureAxes - there is no field data to "
            "plot. Construct a Plot explicitly."
        )
    variable = feature_axes.variables
    if variable.name is None or variable.dim is None:
        raise ValueError(
            f"{data_config}'s FeatureAxes was built from a raw shape, not a "
            "named Variable - auto_plot needs a Variable to label axes/color "
            "by. Construct a Plot explicitly."
        )
    return variable


def _is_auto_expanded(variable: Variable) -> bool:
    """True for a Variable's own multi-component expansion (e.g. dim=3 ->
    children named x_1/x_2/x_3) - one physical quantity, safe to bundle as
    a single color/vector role. False once a child's name breaks that
    pattern, meaning the children are genuinely distinct variables composed
    together (e.g. TEMPERATURE * PRESSURE), not components of one."""
    if variable.is_leaf:
        return True
    return all(
        child.name == f"{variable.name}_{i + 1}" for i, child in enumerate(variable.children)
    )


def _distinct_quantities(variable: Variable) -> list[Variable]:
    """Splits `variable` at the seams between genuinely different
    variables, leaving each auto-expanded multi-component quantity intact.
    A plain scalar or auto-expanded vector returns just itself."""
    if _is_auto_expanded(variable):
        return [variable]
    quantities = []
    for child in variable.children:
        quantities.extend(_distinct_quantities(child))
    return quantities


def _auto_quantity(
    data_config: DataConfiguration, variable: Variable
) -> "Variable | SelectorSpec":
    """The single quantity to plot, or - when the FeatureAxes bundles
    several distinct ones - a SelectorSpec letting the user switch between
    them, as long as they all share one dim (so the same Plot role stays
    valid regardless of which is selected)."""
    quantities = _distinct_quantities(variable)
    if len(quantities) == 1:
        return quantities[0]
    dims = {q.dim for q in quantities}
    if len(dims) == 1:
        return SelectorSpec(quantities)
    names = ", ".join(q.name for q in quantities)
    raise ValueError(
        f"{data_config}'s FeatureAxes bundles multiple distinct variables "
        f"({names}) with different dims ({sorted(dims)}) - auto_plot can't "
        "build one Plot role for all of them. Construct a Plot explicitly, "
        "naming one of them."
    )


def _other_axes(data_config: DataConfiguration, *consumed: Axes | None) -> list[Axes]:
    """Every axis in `data_config` besides the ones already spoken for
    (geometry, feature) and any EllipsisAxes wildcard."""
    return [
        axes
        for axes in data_config.axes
        if axes not in consumed and not isinstance(axes, EllipsisAxes)
    ]


def _resolve_control(spec, axis: Axes):
    """Resolves one control - a ControlSpec subclass or instance - for
    `axis`.

    A class is instantiated with the axis filled in. An instance gets its
    still-unset `variable_or_axes` filled in and is then reused as-is (by
    identity) - the mechanism that lets one instance, passed to several
    auto_plot() calls, end up shared across them (one widget, moving them
    together), the same way a shared Scale or SelectorSpec already works.
    """
    if isinstance(spec, type):
        # By keyword, not position: FixedSpec inherits ControlSpec.__init__
        # verbatim (init_state first), unlike SliderSpec/FacetSpec (both put
        # variable_or_axes first) - the keyword is the only thing every
        # ControlSpec subclass's signature agrees on.
        return spec(variable_or_axes=axis)
    if spec.variable_or_axes is None:
        spec.variable_or_axes = axis
    elif spec.variable_or_axes is not axis:
        raise ValueError(
            f"{spec} is already resolved for {spec.variable_or_axes!r} - "
            f"can't also resolve it for {axis!r}. Use a separate instance "
            "per axis, or a {axis: control} dict."
        )
    return spec


def _default_sliders(
    axes: list[Axes], existing_controls: list, controls=SliderSpec
) -> list[ControlSpec]:
    """One resolved control per axis not already covered by a caller-
    supplied control.

    `controls` (default SliderSpec) is a ControlSpec class or instance
    applied to every uncovered axis, or a `{axis: class-or-instance}` dict
    for several unresolved axes at once. A bare (non-dict) instance only
    makes sense for exactly one uncovered axis - it can't simultaneously
    represent several.

    Bounds/initial state left unresolved (None) are filled in from the data
    by DataPlot.__init__, same as any explicitly-constructed ControlSpec
    with unspecified bounds.
    """
    covered = {c.variable_or_axes for c in existing_controls}
    uncovered = [axis for axis in axes if axis not in covered]
    if isinstance(controls, dict):
        conflicting = covered & controls.keys()
        if conflicting:
            raise ValueError(
                f"controls has entries for axes already covered by an "
                f"explicit control: {conflicting}."
            )
        return [
            _resolve_control(controls.get(axis, SliderSpec), axis) for axis in uncovered
        ]
    if not isinstance(controls, type) and len(uncovered) > 1:
        raise ValueError(
            f"One ControlSpec instance ({controls}) can't resolve "
            f"{len(uncovered)} different axes ({uncovered}) - pass a class "
            "(a new instance per axis) or a {axis: control} dict instead."
        )
    return [_resolve_control(controls, axis) for axis in uncovered]


def _split_controls_default(kwargs: dict):
    """Pulls a class/instance/dict `controls=` (what to use for surplus
    axes) out of `kwargs`, leaving a plain list (or nothing) to forward to
    the Plot constructor as-is. A list means the caller already fully
    resolved their own controls."""
    controls = kwargs.get("controls")
    if controls is None or isinstance(controls, list):
        return SliderSpec, kwargs
    return controls, {k: v for k, v in kwargs.items() if k != "controls"}


def _with_extra_controls(kwargs: dict, axes: list[Axes], controls=SliderSpec) -> dict:
    generated = _default_sliders(axes, kwargs.get("controls") or [], controls)
    if not generated:
        return kwargs
    return dict(kwargs, controls=list(kwargs.get("controls") or []) + generated)


def _auto_geometry_plot(
    data,
    data_config: DataConfiguration,
    geometry_axes: GeometryAxes,
    scale: Scale | None = None,
    **kwargs,
) -> Plot:
    feature_axes = data_config.feature_axes
    variable = _require_named_variable(data_config, feature_axes)
    quantity = _auto_quantity(data_config, variable)
    geometry = geometry_axes.geometry
    default_control, kwargs = _split_controls_default(kwargs)
    # GeometryAxes besides the drawn one are already stepped through by a control.
    all_geometry_axes = [a for a in data_config.axes if isinstance(a, GeometryAxes)]
    kwargs = _with_extra_controls(
        kwargs, _other_axes(data_config, *all_geometry_axes, feature_axes), default_control
    )

    # The plotted quantity IS this geometry's own coordinate Variable (e.g.
    # a PointSampler's own output, or an operator-learning model's
    # coordinate input) - there's no separate quantity to color/height by,
    # so show the points themselves instead of dispatching on `quantity` as
    # if it were one.
    if geometry.variable is not None and quantity == geometry.variable:
        if quantity.dim in (1, 2, 3):
            return PointCloudPlot(data, data_config, color=None, **kwargs)
        raise ValueError(
            f"{quantity.name} matches this geometry's own coordinate "
            f"Variable but has dim={quantity.dim} - only dim 1-3 can be "
            "shown as a point cloud. Construct a Plot explicitly if this "
            "is intentional."
        )

    # Checked structurally (`.mesh` populated), not by isinstance(MeshGeometry)
    # - a SampledGeometry only has a mesh once mesh-mode sampling gave it real
    # cell connectivity, so this correctly falls through to the point-cloud
    # branch below outside of that, without auto_plot needing to know
    # SampledGeometry exists at all.
    if getattr(geometry, "mesh", None) is not None:
        # A 1D "mesh" (points connected in a line, e.g. a sampled Interval)
        # isn't a drawable 2D/3D surface at all - MeshFieldPlot/
        # MeshVectorPlot explicitly reject it. It's a LinePlot: value vs.
        # position along the domain, x=geometry_axes itself (LinePlot reads
        # real coordinates from a 1D geometry's own discretization_points -
        # see LinePlot._geometry_x_values()).
        if geometry.dim == 1:
            if quantity.dim == 1:
                return LinePlot(data, data_config, x=geometry_axes, y=quantity, **kwargs)
            raise ValueError(
                f"{quantity.name} has dim={quantity.dim} - a 1D mesh only "
                "supports a scalar quantity as a LinePlot. Construct a Plot "
                "explicitly if this is intentional."
            )
        if quantity.dim == 1:
            return MeshFieldPlot(
                data, data_config, color=ColorSpec(quantity, scale=scale), **kwargs
            )
        if quantity.dim == geometry.dim:
            return MeshVectorPlot(
                data, data_config, vector=VectorSpec(quantity), **kwargs
            )
        raise ValueError(
            f"{quantity.name} has dim={quantity.dim}, but the mesh is "
            f"{geometry.dim}D - expected dim=1 for a MeshFieldPlot or "
            f"dim={geometry.dim} for a MeshVectorPlot. Construct one "
            "explicitly if this is intentional."
        )

    if isinstance(geometry, GridGeometry):
        # Dispatch on coord_dim (discretization_points' own component count),
        # not on how many structural grid axes there are - a 2-axis
        # index-coordinate grid and a 2-axis grid embedded in 3D space need
        # different Plot families regardless of both having grid_dims==2.
        leaves = geometry.variable.leaves
        if len(leaves) > 2:
            # Surplus structural axes beyond the last two get a default
            # SliderSpec each, the same mechanism _other_axes/
            # _with_extra_controls already applies to non-geometry axes -
            # the last two leaves stay as the drawn grid.
            kwargs = _with_extra_controls(kwargs, leaves[:-2], default_control)
            leaves = leaves[-2:]
        grid_dims = len(leaves)
        coord_dim = geometry.discretization_points.shape[-1]

        if grid_dims == 1:
            if quantity.dim == 1:
                return LinePlot(data, data_config, x=leaves[0], y=quantity, **kwargs)
            raise ValueError(
                f"{quantity.name} has dim={quantity.dim} - a 1D grid only "
                "supports a scalar quantity as a LinePlot. Construct a "
                "PathPlot explicitly for a vector quantity."
            )

        if coord_dim == 2:
            if quantity.dim == 1:
                return HeatmapPlot(
                    data, data_config, x=leaves[0], y=leaves[1],
                    color=ColorSpec(quantity, scale=scale), **kwargs
                )
            if quantity.dim == 2:
                return QuiverPlot(
                    data, data_config, vector=VectorSpec(quantity), **kwargs
                )
            raise ValueError(
                f"{quantity.name} has dim={quantity.dim} - expected dim=1 for "
                "a HeatmapPlot or dim=2 for a QuiverPlot. Construct one "
                "explicitly if this is intentional."
            )

        if coord_dim == 3:
            if quantity.dim == 1:
                return EmbeddedGridPlot(
                    data, data_config, color=ColorSpec(quantity, scale=scale), **kwargs
                )
            if quantity.dim == 3:
                return QuiverPlot(
                    data, data_config, vector=VectorSpec(quantity), **kwargs
                )
            raise ValueError(
                f"{quantity.name} has dim={quantity.dim} - expected dim=1 for "
                "an EmbeddedGridPlot or dim=3 for a QuiverPlot. Construct one "
                "explicitly if this is intentional."
            )

        raise ValueError(
            f"{type(geometry).__name__}'s discretization_points has "
            f"{coord_dim} coordinate components - auto_plot only supports 2 "
            "or 3. Construct a Plot explicitly."
        )

    # Any other DiscreteGeometry still has discretization_points (just no
    # mesh/grid structure to them) - e.g. a SampledGeometry outside mesh
    # mode. QuiverPlot itself has no GridGeometry-specific requirement (only
    # 3D discretization_points), so it's reused as-is for the vector case.
    if (
        isinstance(geometry, DiscreteGeometry)
        and geometry.discretization_points is not None
    ):
        # 1D points (e.g. a PointCloud along a line) aren't drawable by
        # PointCloudPlot (2D/3D only) - a LinePlot, same as the 1D-mesh
        # case above, reading real coordinates via
        # LinePlot._geometry_x_values().
        if geometry.discretization_points.shape[-1] == 1:
            if quantity.dim == 1:
                return LinePlot(data, data_config, x=geometry_axes, y=quantity, **kwargs)
            raise ValueError(
                f"{quantity.name} has dim={quantity.dim} - a 1D point set "
                "only supports a scalar quantity as a LinePlot. Construct a "
                "Plot explicitly if this is intentional."
            )
        if quantity.dim == 1:
            return PointCloudPlot(
                data, data_config, color=ColorSpec(quantity, scale=scale), **kwargs
            )
        if quantity.dim == 3:
            return QuiverPlot(data, data_config, vector=VectorSpec(quantity), **kwargs)
        raise ValueError(
            f"{quantity.name} has dim={quantity.dim} - expected dim=1 for a "
            "PointCloudPlot or dim=3 for a QuiverPlot. Construct one "
            "explicitly if this is intentional."
        )

    raise ValueError(
        f"{type(geometry).__name__} has no known discretization - auto_plot "
        "can't infer point positions for a continuous geometry. Discretize "
        "it first (e.g. via create_mesh(), or a PointSampler run in mesh "
        "mode), or construct a Plot explicitly."
    )


def _auto_flat_plot(data, data_config: DataConfiguration, **kwargs) -> Plot:
    """Unlike the geometry branch, two distinct scalar quantities are not
    ambiguous here - x/y is exactly what a ScatterPlot needs them for, so
    this (unlike _auto_geometry_plot) only reaches for a SelectorSpec once
    ScatterPlot's 2-quantity case no longer applies (3+ distinct scalars)."""
    feature_axes = data_config.feature_axes
    variable = _require_named_variable(data_config, feature_axes)
    quantities = _distinct_quantities(variable)
    other_axes = _other_axes(data_config, feature_axes)
    all_scalar = all(q.dim == 1 for q in quantities)
    default_control, kwargs = _split_controls_default(kwargs)

    if len(quantities) == 1:
        leaves = quantities[0].leaves
        if len(leaves) == 1 and other_axes:
            domain, *rest = other_axes
            return LinePlot(
                data,
                data_config,
                x=domain,
                y=quantities[0],
                **_with_extra_controls(kwargs, rest, default_control),
            )
        if len(leaves) == 2:
            return ScatterPlot(data, data_config, x=leaves[0], y=leaves[1], **kwargs)

    if len(quantities) == 2 and all_scalar:
        # ScatterPlot flattens every remaining axis into its implicit
        # samples axis already - no control needed for `other_axes` here.
        return ScatterPlot(data, data_config, x=quantities[0], y=quantities[1], **kwargs)

    if len(quantities) >= 3 and all_scalar and other_axes:
        domain, *rest = other_axes
        return LinePlot(
            data,
            data_config,
            x=domain,
            y=SelectorSpec(quantities),
            **_with_extra_controls(kwargs, rest, default_control),
        )

    if len(quantities) > 1:
        names = ", ".join(q.name for q in quantities)
        raise ValueError(
            f"{data_config}'s FeatureAxes bundles multiple distinct "
            f"variables ({names}) that auto_plot can't combine into one "
            "plot automatically. Construct a Plot explicitly, e.g. a "
            "ScatterPlot naming two of them, or a LinePlot with an explicit "
            "SelectorSpec."
        )

    raise ValueError(
        f"Can't auto-select a plot for {data_config} - construct one "
        "explicitly (e.g. LinePlot/ScatterPlot/BarPlot/PathPlot with "
        "explicit roles)."
    )
