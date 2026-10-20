"""Restructures data on product geometries (e.g. space-time domains) for
plotting: factors named by a control get an axis of their own, the
remaining factors are merged into the one geometry that is drawn."""

import numpy as np

from qewton.config.axes import EllipsisAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.product_mesh_geometry import (
    ProductDiscreteGeometry,
    ProductMeshGeometry,
)
from qewton.visualization.plots.spec import ControlSpec, SliderSpec

#: Highest dimension a drawn geometry can have.
MAX_DRAWN_DIM = 3


def _as_numpy(data, backend) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    return np.asarray(backend.to_numpy(data))


def _factors_of(geometry_axes: GeometryAxes) -> list | None:
    """The factor geometries behind a GeometryAxes: the factors of a
    product (meshed or a product of point sets), else the geometry itself
    if it has a mesh or discrete points, otherwise None."""
    geometry = geometry_axes.geometry
    factors = getattr(geometry, "factors", None)
    if factors is not None:
        return factors
    if getattr(geometry, "mesh", None) is not None and len(geometry_axes.shape) == 1:
        return [geometry]
    if getattr(geometry, "discretization_points", None) is not None:
        return [geometry]
    return None


def _is_meshed(factor) -> bool:
    return getattr(factor, "mesh", None) is not None


def _factor_shape(factor) -> tuple[int, ...]:
    """The data dimensions a factor spans: one per mesh vertex for a meshed
    factor, else its own structural shape, e.g. (33, 33) for a grid."""
    if _is_meshed(factor):
        return (len(factor.mesh.vertices),)
    return tuple(factor.shape)


def _rebuild_factor(factor):
    """A cpu copy of a factor - a standalone MeshGeometry for a meshed
    factor, a DiscreteGeometry with the same points otherwise."""
    if not _is_meshed(factor):
        return DiscreteGeometry(
            shape=tuple(factor.shape),
            variable=factor.variable,
            dim=factor.variable.dim,
            discretization_points=_as_numpy(factor.discretization_points, factor.backend),
            backend=factor.backend,
        )
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


def _default_split(
    variables: list[Variable], split: dict, vector_dim: int | None = None
) -> list[int]:
    """The 1-dimensional factors to split off by default, if the choice is
    unambiguous: enough of them that a vector with `vector_dim` components
    matches the drawn geometry (e.g. a 2D velocity on space x time), or
    else that the drawn geometry has at most MAX_DRAWN_DIM dimensions."""
    drawn = [i for i in range(len(variables)) if i not in split]
    drawn_dim = sum(variables[i].dim for i in drawn)
    candidates = [i for i in drawn if variables[i].dim == 1]
    names = [variables[i].name for i in drawn]
    if vector_dim is not None and len(candidates) >= drawn_dim - vector_dim > 0:
        excess = drawn_dim - vector_dim
        if len(candidates) > excess:
            raise ValueError(
                f"A {vector_dim}-component vector can not be drawn on the "
                f"product of {names} as a whole. Pass a control for {excess} of "
                f"the 1-dimensional factors {[variables[i].name for i in candidates]}."
            )
        return candidates
    excess = drawn_dim - MAX_DRAWN_DIM
    if excess <= 0:
        return []
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


def product_view(
    data, data_config: DataConfiguration, controls=None, vector_dim: int | None = None
):
    """Restructure data on product geometries for plotting.

    Applies to data whose GeometryAxes refer to product geometries (e.g.
    the vertices of a meshed space-time domain), or to several discrete
    GeometryAxes (e.g. the output of a ProductSampler). Each factor that
    one of the `controls` is bound to gets a GeometryAxes of its own, which
    the control steps through. The other factors are merged into one
    GeometryAxes, which is drawn.

    Without a control, all factors are drawn together. If that exceeds
    :data:`MAX_DRAWN_DIM` dimensions, or if a vector with `vector_dim`
    components is plotted and the drawn factors have more dimensions than
    it (e.g. a 2D velocity on a space-time domain), the 1-dimensional
    factors are split off, as long as this choice is unambiguous. They get
    a default control: built from a ControlSpec class or unbound instance
    passed as `controls`, a
    :class:`~qewton.visualization.plots.spec.SliderSpec` otherwise.

    Args:
        data (array-like): The values, with one dimension per axis of
            `data_config`.
        data_config: Describes `data`.
        controls (list | type[ControlSpec] | ControlSpec | dict | None): The
            controls passed to the plot. Only ControlSpec instances bound
            to a factor's Variable, or to a GeometryAxes holding a single
            factor, select factors.
        vector_dim: Number of components of the plotted vector, or None
            for a scalar or no vector at all.

    Returns:
        tuple: A ``(data, data_config, controls)`` triple:

        - ``data`` (numpy.ndarray or array-like): The restructured values;
          unchanged without product structure.
        - ``data_config`` (DataConfiguration): Describes the restructured
          values.
        - ``controls`` (list, ControlSpec class or instance, dict, or
          None): A list if a single ControlSpec instance selects a factor
          or default controls are added; then a ControlSpec class, unbound
          instance or dict passed as `controls` no longer applies to the
          remaining axes, which get a SliderSpec instead. Unchanged
          otherwise.

    Raises:
        ValueError: If a control names a factor with more than one
            dimension or only a component of a factor, if no factor is
            left to draw, or if the factors to split off by default are
            ambiguous or too few.
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
    defaults = _default_split(variables, selected, vector_dim)
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

    # Expand every geometry axis into the dimensions of its factors - one
    # per meshed factor, a factor's own structural shape otherwise.
    expanded_shape, positions, factor_positions = [], [], []
    for i, axis in enumerate(axes):
        start = len(expanded_shape)
        if i in geometry_idx:
            for f in factors_per_axes[geometry_idx.index(i)]:
                shape = _factor_shape(f)
                factor_positions.append(
                    list(range(len(expanded_shape), len(expanded_shape) + len(shape)))
                )
                expanded_shape.extend(shape)
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
        + [p for i in split_order for p in factor_positions[i]]
        + [p for i in drawn for p in factor_positions[i]]
        + [p for i in rest for p in positions[i]]
    )
    data = data.transpose(perm)
    n_lead = sum(len(positions[i]) for i in lead) + sum(
        len(factor_positions[i]) for i in split_order
    )
    n_drawn = sum(len(factor_positions[i]) for i in drawn)

    rebuilt = {i: _rebuild_factor(factors[i]) for i in range(len(factors))}
    factor_axes = [
        split[i] if split[i] is not None else GeometryAxes(rebuilt[i])
        for i in split_order
    ]
    backend = rebuilt[drawn[0]].backend
    if len(drawn) == 1:
        drawn_geometry = rebuilt[drawn[0]]
    elif all(_is_meshed(rebuilt[i]) for i in drawn):
        drawn_geometry = ProductMeshGeometry([rebuilt[i] for i in drawn], backend=backend)
    else:
        drawn_geometry = ProductDiscreteGeometry(
            [rebuilt[i] for i in drawn], backend=backend
        )
    if isinstance(drawn_geometry, ProductMeshGeometry):
        # A product mesh numbers its vertices along one axis.
        drawn_size = int(np.prod(data.shape[n_lead : n_lead + n_drawn]))
        data = data.reshape(
            data.shape[:n_lead] + (drawn_size,) + data.shape[n_lead + n_drawn :]
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
