"""Automatic selection of a Plot type and its roles from the data axes."""

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
    """Build a Plot for `data`, choosing its type and roles from the axes.

    Data on a product geometry (e.g. a meshed space-time domain, or the
    output of a ProductSampler) is first restructured by
    :func:`~qewton.visualization.product_view.product_view`, with or
    without `plot_type`.

    With ``plot_type=None``, the plot type is selected as follows:

    - Field data on a meshed GeometryAxes becomes a
      :class:`~qewton.visualization.plots.data.mesh.MeshFieldPlot` for a
      scalar Variable, or a
      :class:`~qewton.visualization.plots.data.mesh.MeshVectorPlot` for a
      vector with one component per mesh dimension. A 1D mesh gives a
      :class:`~qewton.visualization.plots.data.curve.LinePlot`.
    - Field data on a
      :class:`~qewton.geometries.discrete.grid_geometry.GridGeometry` is
      dispatched on the coordinate count of its ``discretization_points``:
      2 gives a :class:`~qewton.visualization.plots.data.grid.HeatmapPlot`
      (scalar) or a 2-component
      :class:`~qewton.visualization.plots.data.grid.QuiverPlot`; 3 gives
      an :class:`~qewton.visualization.plots.data.grid.EmbeddedGridPlot`
      (scalar) or a 3-component QuiverPlot. Grid axes beyond the last two
      get a default control each; a grid with one axis gives a LinePlot.
    - Field data on any other discrete point set becomes a
      :class:`~qewton.visualization.plots.data.points.PointCloudPlot`
      (scalar) or a QuiverPlot (one component per coordinate); 1D points
      give a LinePlot. If the plotted Variable is the geometry's own
      coordinate Variable, the points themselves are drawn as an uncolored
      PointCloudPlot.
    - Without a GeometryAxes, a FeatureAxes with one scalar leaf becomes a
      LinePlot over the first remaining axis; two scalar leaves or two
      distinct scalar Variables become a
      :class:`~qewton.visualization.plots.data.samples.ScatterPlot`; three
      or more distinct scalar Variables become a LinePlot with a
      :class:`~qewton.visualization.plots.spec.SelectorSpec` on `y`.
    - With a GeometryAxes, several distinct Variables of one dimension in
      the FeatureAxes are offered through a SelectorSpec.
    - Every remaining axis gets a default control (a
      :class:`~qewton.visualization.plots.spec.SliderSpec` unless
      ``controls`` names another), except for a ScatterPlot, which
      flattens them into its samples.
    - Anything else raises ``ValueError`` instead of guessing.

    With an explicit `plot_type`, this calls
    ``plot_type(data, data_config, **kwargs)`` without any selection, so
    `kwargs` must supply that type's required roles.

    Args:
        data (array-like): The values, with one dimension per axis of
            `data_config`.
        data_config: Describes the axes of `data`. Plot type and roles are
            inferred from it.
        plot_type: The Plot class to construct, or None to select one
            automatically.
        **kwargs: Forwarded to the Plot constructor. Two keys are handled
            here:

            - ``controls``: A list of resolved ControlSpec instances, a
              ControlSpec class or unbound instance to use for every axis
              left over for a control, or a ``{axis: control}`` dict. A
              ControlSpec bound to a factor's Variable, e.g.
              ``[TimeSpec(T)]``, splits that factor off a product
              geometry. With an explicit `plot_type`, it must be a list.
            - ``scale``: A
              :class:`~qewton.visualization.plots.spec.Scale` for the
              ColorSpec created here, e.g. ``Scale(vmin=0.0, vmax=1.0)``
              for one fixed color range over every control state. Not
              allowed with an explicit `plot_type`.

    Returns:
        The constructed plot.

    Raises:
        ValueError: If no plot type can be selected unambiguously; if part
            of ``controls`` would be ignored (a dict key or single
            ControlSpec instance no axis takes a control for); if
            ``controls`` is not a list with an explicit `plot_type`; or if
            ``scale`` is given with an explicit `plot_type` or for a plot
            without color.
    """
    scale = kwargs.pop("scale", None)
    requested_controls = kwargs.get("controls")
    data, data_config, controls = product_view(
        data, data_config, requested_controls, _vector_dim(data_config, plot_type, kwargs)
    )
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


def _product_hint(geometry) -> str:
    """A hint for point sets whose coordinates combine several variables,
    e.g. (x, alpha) - most likely a product domain passed as one flat point
    cloud, which can't be split into its factors."""
    variable = getattr(geometry, "variable", None)
    if variable is None or variable.is_leaf or len(variable.children) < 2:
        return ""
    names = [child.name for child in variable.children]
    return (
        f" If these points are a product domain of {names}, pass one "
        "GeometryAxes per factor, e.g. DataConfiguration(GeometryAxes("
        "A_int.discretize(alphas)), GeometryAxes(Omega.discretize(x_grid)), "
        "FeatureAxes(u)), so a factor can get a slider."
    )


def _vector_dim(data_config: DataConfiguration, plot_type, kwargs) -> int | None:
    """Number of components of the vector the plot will draw, or None for a
    scalar. Taken from `vector=` with an explicit `plot_type`, else from a
    single auto-expanded multi-component FeatureAxes Variable."""
    if plot_type is not None:
        vector = kwargs.get("vector")
        dim = getattr(vector, "n_dimensions", None) or getattr(vector, "dim", None)
        return dim if dim in (2, 3) else None
    feature_axes = data_config.feature_axes
    variable = feature_axes.variables if feature_axes is not None else None
    if variable is None or variable.dim not in (2, 3) or variable.name is None:
        return None
    return variable.dim if _is_auto_expanded(variable) else None


def _check_controls_used(controls, plot: Plot) -> None:
    """Raise if a dict entry or a single ControlSpec instance passed as
    `controls=` did not end up as one of `plot`'s controls. A ControlSpec
    class only names the type for leftover axes, so it is never unused."""

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
    """Check whether `plot` draws domain against value on shared 2D axes.

    Such plots can be combined in one
    :class:`~qewton.visualization.layout.Overlay`.

    Args:
        plot: The plot to check.

    Returns:
        True for a
        :class:`~qewton.visualization.plots.data.curve.LinePlot`, a
        :class:`~qewton.visualization.plots.data.curve.PathPlot`, and a
        :class:`~qewton.visualization.plots.data.points.PointCloudPlot`
        over a 1D domain; False otherwise.
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
    """True for a leaf or a Variable's own multi-component expansion (e.g.
    dim=3 -> children named x_1/x_2/x_3), False for distinct variables
    composed together (e.g. TEMPERATURE * PRESSURE)."""
    if variable.is_leaf:
        return True
    return all(
        child.name == f"{variable.name}_{i + 1}" for i, child in enumerate(variable.children)
    )


def _distinct_quantities(variable: Variable) -> list[Variable]:
    """Split `variable` into its distinct quantities, keeping each
    auto-expanded multi-component quantity intact."""
    if _is_auto_expanded(variable):
        return [variable]
    quantities = []
    for child in variable.children:
        quantities.extend(_distinct_quantities(child))
    return quantities


def _auto_quantity(
    data_config: DataConfiguration, variable: Variable
) -> "Variable | SelectorSpec":
    """The single quantity to plot, or a SelectorSpec over several distinct
    quantities if they all share one dim (so one Plot role fits each)."""
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
    """Every axis in `data_config` except `consumed` and EllipsisAxes."""
    return [
        axes
        for axes in data_config.axes
        if axes not in consumed and not isinstance(axes, EllipsisAxes)
    ]


def _resolve_control(spec, axis: Axes):
    """Resolve one control - a ControlSpec class or instance - for `axis`.

    A class is instantiated for the axis. An instance gets its unset
    `variable_or_axes` filled in and is returned as-is, so one instance
    passed to several auto_plot() calls is shared between their plots.
    """
    if isinstance(spec, type):
        # By keyword: ControlSpec subclasses differ in positional order.
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
    """One resolved control per axis not covered by `existing_controls`.

    `controls` is a ControlSpec class or instance applied to every
    uncovered axis (an instance only for exactly one), or an
    `{axis: class-or-instance}` dict. Unset bounds and initial states are
    filled in from the data by DataPlot.
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
    """Split a class/instance/dict `controls=` (the default for leftover
    axes) off `kwargs`; a list stays in `kwargs` and is forwarded as-is."""
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

    # The plotted quantity is this geometry's own coordinate Variable (e.g.
    # a PointSampler's output): there is nothing to color by, so show the
    # points themselves.
    if geometry.variable is not None and quantity == geometry.variable:
        if quantity.dim in (1, 2, 3):
            return PointCloudPlot(data, data_config, color=None, **kwargs)
        raise ValueError(
            f"{quantity.name} matches this geometry's own coordinate "
            f"Variable but has dim={quantity.dim} - only dim 1-3 can be "
            "shown as a point cloud. Construct a Plot explicitly if this "
            "is intentional."
        )

    # Checked structurally (`.mesh` populated), not by isinstance: a
    # SampledGeometry only has a mesh after mesh-mode sampling, and falls
    # through to the point-cloud branch below otherwise.
    if getattr(geometry, "mesh", None) is not None:
        # A 1D mesh (e.g. a sampled Interval) is drawn as a LinePlot over
        # the geometry's own coordinates (LinePlot._geometry_x_values()).
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
        # Dispatch on coord_dim (discretization_points' component count),
        # not on the number of structural grid axes: a 2-axis grid can be
        # flat or embedded in 3D space.
        leaves = geometry.variable.leaves
        if len(leaves) > 2:
            # Every grid axis but the last two gets a default control; the
            # last two are drawn.
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

    # Any other DiscreteGeometry with discretization_points but no mesh or
    # grid structure, e.g. a SampledGeometry outside mesh mode.
    if (
        isinstance(geometry, DiscreteGeometry)
        and geometry.discretization_points is not None
    ):
        # PointCloudPlot draws 2D/3D points only; 1D points become a
        # LinePlot, as for a 1D mesh.
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
        # QuiverPlot needs one vector component per coordinate.
        coord_dim = geometry.discretization_points.shape[-1]
        if quantity.dim == coord_dim:
            return QuiverPlot(data, data_config, vector=VectorSpec(quantity), **kwargs)
        raise ValueError(
            f"{quantity.name} has dim={quantity.dim} - expected dim=1 for a "
            f"PointCloudPlot or dim={coord_dim} for a QuiverPlot on these "
            f"{coord_dim}D points. Construct one explicitly if this is "
            f"intentional.{_product_hint(geometry)}"
        )

    raise ValueError(
        f"{type(geometry).__name__} has no known discretization - auto_plot "
        "can't infer point positions for a continuous geometry. Discretize "
        "it first (e.g. via create_mesh(), or a PointSampler run in mesh "
        "mode), or construct a Plot explicitly."
    )


def _auto_flat_plot(data, data_config: DataConfiguration, **kwargs) -> Plot:
    """Plot selection without a drawn GeometryAxes. Two distinct scalar
    quantities give a ScatterPlot; a SelectorSpec is only used for three or
    more."""
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
        # ScatterPlot flattens every remaining axis into its samples axis,
        # so `other_axes` need no control.
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
