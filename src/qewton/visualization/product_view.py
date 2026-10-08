"""Restructures data on product geometries (e.g. space-time domains) for
plotting: factors named by a control get an axis of their own, the
remaining factors are merged into the one geometry that is drawn."""

import numpy as np

from qewton.config.axes import EllipsisAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.product_mesh_geometry import ProductMeshGeometry
from qewton.visualization.plots.spec import ControlSpec, SliderSpec

#: Highest dimension a drawn geometry can have.
MAX_DRAWN_DIM = 3


def _as_numpy(data, backend) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(backend.to_numpy(data))


def _factors_of(geometry_axes: GeometryAxes) -> list | None:
    """The meshed factor geometries behind a GeometryAxes: the factors of a
    product mesh, the geometry itself if it has a mesh, otherwise None."""
    if len(geometry_axes.shape) != 1:
        return None
    geometry = geometry_axes.geometry
    factors = getattr(geometry, "factors", None)
    if factors is not None:
        return factors
    if getattr(geometry, "mesh", None) is not None:
        return [geometry]
    return None


def _rebuild_factor(factor) -> MeshGeometry:
    """A cpu copy of a factor's mesh as a standalone MeshGeometry."""
    vertices = _as_numpy(factor.mesh.vertices, factor.backend)
    cells = _as_numpy(factor.mesh.cells, factor.backend)
    if cells.ndim < 2 or cells.shape[0] == 0:
        cells = []
    return MeshGeometry(
        variable=factor.variable,
        mesh=Mesh(vertices=vertices, cells=cells, backend=factor.backend),
        backend=factor.backend,
    )


def _explicit_controls(controls) -> list:
    if isinstance(controls, list):
        return controls
    if isinstance(controls, ControlSpec):
        return [controls]
    return []


def _split_from_controls(
    controls, variables: list[Variable], factor_of_axes: dict
) -> dict[int, GeometryAxes | None]:
    """Factor indices a control is bound to - by the factor's Variable, or
    by a GeometryAxes holding only that factor - mapped to that GeometryAxes
    (None if bound to the Variable)."""
    split = {}
    for control in _explicit_controls(controls):
        target = control.variable_or_axes
        if isinstance(target, GeometryAxes):
            if id(target) in factor_of_axes:
                split[factor_of_axes[id(target)]] = target
            continue
        if not isinstance(target, Variable):
            continue
        for idx, variable in enumerate(variables):
            if target == variable:
                if variable.dim != 1:
                    raise ValueError(
                        f"A control can only step through a 1-dimensional factor, "
                        f"but {variable.name} has dim={variable.dim}."
                    )
                split[idx] = None
                break
            if target in variable.leaves:
                raise ValueError(
                    f"{target.name} is only a component of the factor "
                    f"{variable.name}. A control has to name a whole factor of "
                    "the product geometry."
                )
    return split


def _default_split(variables: list[Variable], split: dict) -> list[int]:
    """The 1-dimensional factors to split off by default, if the drawn
    geometry would exceed MAX_DRAWN_DIM dimensions and the choice is
    unambiguous."""
    drawn = [i for i in range(len(variables)) if i not in split]
    excess = sum(variables[i].dim for i in drawn) - MAX_DRAWN_DIM
    if excess <= 0:
        return []
    candidates = [i for i in drawn if variables[i].dim == 1]
    names = [variables[i].name for i in drawn]
    if len(candidates) < excess:
        raise ValueError(
            f"The product of {names} has more than {MAX_DRAWN_DIM} dimensions and "
            "can not be drawn, since too few of its factors are 1-dimensional "
            "and could be controlled."
        )
    if len(candidates) > excess:
        raise ValueError(
            f"The product of {names} has more than {MAX_DRAWN_DIM} dimensions and "
            f"can not be drawn as a whole. Pass a control for {excess} of the "
            f"1-dimensional factors {[variables[i].name for i in candidates]}."
        )
    return candidates


def _default_control(controls, variable: Variable, n_defaults: int) -> ControlSpec:
    """The control for a factor split off by default: built from a
    ControlSpec class, or an unbound ControlSpec instance, passed as
    `controls` - a SliderSpec otherwise."""
    if isinstance(controls, type) and issubclass(controls, ControlSpec):
        return controls(variable_or_axes=variable)
    if (
        isinstance(controls, ControlSpec)
        and controls.variable_or_axes is None
        and n_defaults == 1
    ):
        controls.variable_or_axes = variable
        return controls
    return SliderSpec(variable_or_axes=variable)


def product_view(data, data_config: DataConfiguration, controls=None):
    """Restructures data on product geometries for plotting.

    Applies to data whose GeometryAxes refer to product meshes (e.g. the
    vertices of a meshed space-time domain), or to several meshed
    GeometryAxes (e.g. the output of a ProductSampler). Each factor of
    these products that one of the `controls` is bound to gets a
    GeometryAxes of its own, which the control steps through. The other
    factors are merged into one GeometryAxes, which is drawn.

    Without a control, all factors are drawn together. If that exceeds 3
    dimensions, the 1-dimensional factors are split off, as long as this
    choice is unambiguous, and get a default control: built from a
    ControlSpec class or unbound instance passed as `controls`, a
    SliderSpec otherwise.

    Args:
        data: The values, with one dimension per axis of `data_config`.
        data_config (DataConfiguration): Describes `data`.
        controls: The controls passed to the plot (a list, a ControlSpec
            class or instance, a dict, or None). Only ControlSpec instances
            bound to a factor's Variable, or to a GeometryAxes holding a
            single factor, select factors.

    Returns:
        tuple: `(data, data_config, controls)` - unchanged if the data has
            no product structure. `controls` becomes a list if a single
            ControlSpec instance selects a factor or default controls are
            added; a ControlSpec class, unbound instance or dict passed as
            `controls` then no longer resolves other remaining axes, which
            get a SliderSpec instead.

    Raises:
        ValueError: If a control names a factor that has more than one
            dimension or only a component of a factor, if no factor is
            left to draw, or if more than 3 dimensions would be drawn and
            the factors to split off are ambiguous.
    """
    axes = list(data_config.axes)
    geometry_idx = [i for i, a in enumerate(axes) if isinstance(a, GeometryAxes)]
    if not geometry_idx or any(isinstance(a, EllipsisAxes) for a in axes):
        return data, data_config, controls
    factors_per_axes = [_factors_of(axes[i]) for i in geometry_idx]
    if any(f is None for f in factors_per_axes):
        return data, data_config, controls
    is_product = len(geometry_idx) > 1 or len(factors_per_axes[0]) > 1
    if not is_product:
        return data, data_config, controls

    factors = [f for per_axes in factors_per_axes for f in per_axes]
    variables = [f.variable for f in factors]
    factor_of_axes, offset = {}, 0
    for i, per_axes in zip(geometry_idx, factors_per_axes):
        if len(per_axes) == 1:
            factor_of_axes[id(axes[i])] = offset
        offset += len(per_axes)
    selected = _split_from_controls(controls, variables, factor_of_axes)
    defaults = _default_split(variables, selected)
    split = {**selected, **{i: None for i in defaults}}
    drawn = [i for i in range(len(factors)) if i not in split]
    if not drawn:
        raise ValueError(
            "Every factor of the product geometry has a control, so nothing is "
            "left to draw."
        )
    if not split and len(geometry_idx) == 1:
        # Drawn as a whole - the product mesh can be used as it is.
        return data, data_config, controls

    data = np.asarray(data)
    if sum(len(a.shape) for a in axes) != data.ndim:
        return data, data_config, controls

    # Expand every geometry dimension into one dimension per factor.
    expanded_shape, positions, factor_positions = [], [], []
    for i, axis in enumerate(axes):
        start = len(expanded_shape)
        if i in geometry_idx:
            sizes = [len(f.mesh.vertices) for f in factors_per_axes[geometry_idx.index(i)]]
            factor_positions.extend(range(start, start + len(sizes)))
            expanded_shape.extend(sizes)
        else:
            n_dims = len(axis.shape)
            offset = sum(len(a.shape) for a in axes[:i])
            expanded_shape.extend(data.shape[offset : offset + n_dims])
        positions.append(list(range(start, len(expanded_shape))))
    data = data.reshape(expanded_shape)

    split_order = sorted(split)
    lead = [i for i in range(len(axes)) if i < geometry_idx[0]]
    rest = [i for i in range(len(axes)) if i > geometry_idx[0] and i not in geometry_idx]
    perm = (
        [p for i in lead for p in positions[i]]
        + [factor_positions[i] for i in split_order]
        + [factor_positions[i] for i in drawn]
        + [p for i in rest for p in positions[i]]
    )
    data = data.transpose(perm)
    n_lead = sum(len(positions[i]) for i in lead) + len(split_order)
    drawn_size = int(np.prod(data.shape[n_lead : n_lead + len(drawn)]))
    data = data.reshape(
        data.shape[:n_lead] + (drawn_size,) + data.shape[n_lead + len(drawn) :]
    )

    rebuilt = {i: _rebuild_factor(factors[i]) for i in range(len(factors))}
    factor_axes = [
        split[i] if split[i] is not None else GeometryAxes(rebuilt[i])
        for i in split_order
    ]
    drawn_geometry = (
        rebuilt[drawn[0]]
        if len(drawn) == 1
        else ProductMeshGeometry(
            [rebuilt[i] for i in drawn], backend=rebuilt[drawn[0]].backend
        )
    )
    new_config = DataConfiguration(
        *[axes[i] for i in lead],
        *factor_axes,
        GeometryAxes(drawn_geometry),
        *[axes[i] for i in rest],
        dtype=data_config.dtype,
    )
    default_controls = [
        _default_control(controls, variables[i], len(defaults)) for i in defaults
    ]
    if isinstance(controls, list):
        explicit = controls
    elif isinstance(controls, ControlSpec) and selected:
        explicit = [controls]
    else:
        explicit = None
    if default_controls:
        controls = (explicit or []) + default_controls
    elif explicit is not None:
        controls = explicit
    return data, new_config, controls
