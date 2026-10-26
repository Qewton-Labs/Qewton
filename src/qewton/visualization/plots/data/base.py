"""Base class :class:`DataPlot` and shared helpers for plots of data
described by a DataConfiguration."""

import numpy as np

from qewton.backends import resolve_backend
from qewton.config.axes import GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.result import VectorResult
from qewton.visualization.plots.spec import (
    ColorSpec,
    ControlSpec,
    PlotSpec,
    drawn_geometry_axes,
)


class CoordTransform:
    """Transforms a plot's evaluated point positions before they are drawn.

    Notes:
        Subclasses must implement :meth:`apply`.
    """

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        """Return the transformed positions.

        Args:
            vertices: Point positions, shape ``(n_points, dim)``.

        Returns:
            numpy.ndarray: The transformed positions.
        """
        raise NotImplementedError


class IdentityCoord(CoordTransform):
    """A CoordTransform that leaves point positions unchanged."""

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        """Return `vertices` unchanged."""
        return vertices


class DataPlot(Plot):
    """Base class for a Plot whose values come from `data` described by a
    DataConfiguration.

    The input family underlying the grid, mesh, point, curve and sample
    plot families. At construction, every control is resolved against the
    size of its data dimension (see
    :meth:`~qewton.visualization.plots.spec.ControlSpec.resolve`) and, for
    the dimension of a one-dimensional geometry (e.g. time), gets that
    geometry's coordinates.

    Args:
        data (array_like): The data to plot, indexable like an array of any
            supported backend.
        data_config: The
            :class:`~qewton.config.data_configurations.DataConfiguration`
            describing the dimensions of `data`.
        title (str, optional): Panel heading.
        label (str, optional): Legend entry text.
        theme (Theme, optional): Theme used to draw this plot.
        controls: Control specs reducing or partitioning `data`.
        coord_transform: Transform applied to point positions before drawing.
            Defaults to :class:`IdentityCoord`.

    Attributes:
        data (array_like): The data to plot.
        data_config (DataConfiguration): The configuration describing `data`.
        coord_transform (CoordTransform): Transform applied to point positions.
        drawn_geometry_axes (GeometryAxes or list of GeometryAxes or None): The
            GeometryAxes this plot draws: all GeometryAxes of `data_config`
            except those a control steps through. See
            :func:`~qewton.visualization.plots.spec.drawn_geometry_axes`.

    Notes:
        Subclasses must implement
        :meth:`~qewton.visualization.plots.base.Plot.evaluate` (usually
        starting with :meth:`apply_controls`) and
        :meth:`~qewton.visualization.plots.base.Plot.create_artist`.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        title=None,
        label=None,
        theme=None,
        controls: list[ControlSpec] | None = None,
        coord_transform: CoordTransform | None = None,
    ) -> None:
        super().__init__(title=title, label=label, theme=theme, controls=controls)
        self.data = data
        self.data_config = data_config
        self.coord_transform = coord_transform or IdentityCoord()

        for spec in self.controls:
            axis_slc, entry_slc = PlotSpec.get_slice(
                spec.variable_or_axes, self.data_config
            )
            if entry_slc is None:
                size = data.shape[PlotSpec.as_single_dim(axis_slc)]
            else:
                size = entry_slc.stop - entry_slc.start
            spec.resolve(range(size))
            if entry_slc is None:
                spec.set_coordinates(self._dim_coordinates(axis_slc, size))

    def _dim_coordinates(self, axis_slc, size: int) -> np.ndarray | None:
        """Coordinates along a data dimension, if it is the only dimension
        of a GeometryAxes over a 1D geometry (e.g. the time axis of
        space-time data) and the sizes match - None otherwise."""
        dim = PlotSpec.as_single_dim(axis_slc)
        dim = dim if dim >= 0 else self.data.ndim + dim
        offset = 0
        for axes in self.data_config.axes:
            if offset == dim:
                break
            offset += len(axes.shape)
        else:
            return None
        if not isinstance(axes, GeometryAxes) or len(axes.shape) != 1:
            return None
        geometry = axes.geometry
        if geometry.dim != 1:
            return None
        mesh = getattr(geometry, "mesh", None)
        points = mesh.vertices if mesh is not None else geometry.discretization_points
        if points is None:
            return None
        if not isinstance(points, np.ndarray):
            points = geometry.backend.to_numpy(points)
        points = np.asarray(points).reshape(-1)
        return points if len(points) == size else None

    def _resolve_controls(self) -> list[tuple[ControlSpec, int]]:
        """(spec, dim index in the original self.data) for each control spec,
        sorted by dim index descending, so collapsing one dimension never
        shifts the indices still to be processed."""
        resolved = []
        for spec in self.controls:
            axis_slc, entry_slc = PlotSpec.get_slice(
                spec.variable_or_axes, self.data_config
            )
            assert (
                entry_slc is None
            ), f"It is not yet supported to use controls on variables:\
                {spec.variable_or_axes}."
            axis_slc = PlotSpec.as_single_dim(axis_slc)
            real_idx = axis_slc if axis_slc >= 0 else len(self.data.shape) + axis_slc
            resolved.append((spec, real_idx))

        resolved.sort(key=lambda item: item[1], reverse=True)
        return resolved

    def apply_controls(self):
        """Slice `data` down to the current control states.

        Each control removes the dimension it acts on, indexed at its
        current state.

        Returns:
            tuple: ``(sliced, index_map, slice_map)``, where

            - ``sliced`` (array_like) is the reduced data.
            - ``index_map`` (callable) maps a dimension index of the original
              `data` to the corresponding index in `sliced`.
            - ``slice_map`` (callable) maps a slice tuple computed against the
              original `data_config` (e.g. from
              ``DataConfiguration.get_variable_slice``) onto `sliced` by
              dropping the entries of removed dimensions. Raises
              NotImplementedError for a tuple containing Ellipsis.
        """
        resolved = self._resolve_controls()

        sliced = self.data[:]
        removed_indices = []
        for spec, real_idx in resolved:
            indexer = [slice(None)] * sliced.ndim
            indexer[real_idx] = spec.state
            sliced = sliced[tuple(indexer)]
            removed_indices.append(real_idx)

        def index_map(original_idx: int) -> int:
            shift = sum(1 for r in removed_indices if r < original_idx)
            return original_idx - shift

        def slice_map(slc: tuple) -> tuple:
            if Ellipsis in slc:
                raise NotImplementedError(
                    "slice_map does not yet support slices containing Ellipsis."
                )
            return tuple(s for i, s in enumerate(slc) if i not in removed_indices)

        return sliced, index_map, slice_map

    def reduce_coordinates(self, points, geometry_dims: tuple[int, int]):
        """Apply the control reduction of :meth:`apply_controls` to geometry
        coordinates.

        Controls on a geometry dimension reduce `points` the same way they
        reduce `data`, so drawn positions stay in sync with drawn values.

        Args:
            points (array_like): Array indexed by the geometry's dimensions
                plus a trailing coordinate-component axis that is never
                reduced, e.g. ``discretization_points`` with shape ``(...,
                n_components)``.
            geometry_dims: ``(start, stop)`` range the geometry's dimensions
                occupy in `data`.

        Returns:
            array_like: The reduced `points`.
        """
        start, _ = geometry_dims
        reduced = points
        for spec, real_idx in self._resolve_controls():  # already sorted descending
            local = real_idx - start
            if 0 <= local < reduced.ndim - 1:
                indexer = [slice(None)] * reduced.ndim
                indexer[local] = spec.state
                reduced = reduced[tuple(indexer)]
        return reduced

    @property
    def drawn_geometry_axes(self):
        return drawn_geometry_axes(self.data_config, self.controls)

    def _geometry_dims(self) -> tuple[int, int]:
        """(start, stop) range in self.data that this plot's drawn
        GeometryAxes occupies, for use with reduce_coordinates()."""
        geom_axes = self.drawn_geometry_axes
        axis_slc, entry_slc = PlotSpec.get_slice(geom_axes, self.data_config)
        assert entry_slc is None
        if isinstance(axis_slc, slice):
            return axis_slc.start, axis_slc.stop
        real_idx = axis_slc if axis_slc >= 0 else self.data.ndim + axis_slc
        return real_idx, real_idx + 1

    def _count_controls_on_geometry_dims(self) -> int:
        """How many of this plot's controls reduce a dimension that belongs
        to the GeometryAxes, rather than a batch or feature dimension."""
        start, stop = self._geometry_dims()
        return sum(
            1 for _, real_idx in self._resolve_controls() if start <= real_idx < stop
        )

    def _resolve_structural_dim(self, spec) -> int:
        """Resolves a PlotSpec that must refer to exactly one full array
        dimension (not a channel slice) to its index in self.data. Raises
        ValueError otherwise."""
        axis_slc, entry_slc = PlotSpec.get_slice(spec.variable_or_axes, self.data_config)
        if entry_slc is not None:
            raise ValueError(
                f"{spec.variable_or_axes} refers to a channel slice, not a "
                f"own dimension - not allowed here for {type(self).__name__}."
            )
        if isinstance(axis_slc, slice):
            length = axis_slc.stop - axis_slc.start
            if length != 1:
                raise ValueError(
                    f"{spec.variable_or_axes} spans {length} dimensions - "
                    "must refer to exactly one dimension."
                )
            real_idx = axis_slc.start
        else:
            real_idx = axis_slc
        return real_idx if real_idx >= 0 else self.data.ndim + real_idx

    @staticmethod
    def component_count(spec, data_config: DataConfiguration) -> int:
        """Return the number of feature components a spec resolves to.

        Args:
            spec (PlotSpec): The spec to inspect.
            data_config: The configuration of the data.

        Returns:
            int: The Variable's dimension, or the length of the last entry of
                the spec's slice; 1 means scalar.
        """
        var = spec.variable_or_axes
        if isinstance(var, Variable):
            return var.dim
        slc = data_config.get_variable_slice(var)
        last = slc[-1] if isinstance(slc, tuple) else slc
        return (last.stop - last.start) if isinstance(last, slice) else 1

    def require_scalar(self, spec, role: str):
        """Check that `spec` resolves to a scalar.

        Args:
            spec (PlotSpec): The spec to check.
            role: Name of the role, used in the error message.

        Raises:
            ValueError: If `spec` has more than one component.
        """
        n = self.component_count(spec, self.data_config)
        if n != 1:
            raise ValueError(
                f"{role} must be scalar (dim=1), got dim={n}. "
                "Select a single component, or use a vector-aware plot type."
            )


class VectorFieldPlotMixin:
    """Shared vector and color handling for arrow-field plots.

    Used by :class:`~qewton.visualization.plots.data.grid.QuiverPlot`
    (grid points) and
    :class:`~qewton.visualization.plots.data.mesh.MeshVectorPlot` (mesh
    vertices), which differ only in where the arrow positions come from.

    The `color` argument of those plots resolves as:

    - None (default): colors by vector magnitude, i.e. a ColorSpec on the
      vector's own Variable. Pass ``ColorSpec(vector_variable, cmap=...,
      scale=...)`` to keep magnitude coloring with a custom colormap or a
      shared Scale.
    - a scalar ColorSpec or Variable: colors each arrow by that field.
    - False: flat, uncolored arrows in the theme's ``vector_color``.

    Arrows whose vector or color is NaN are dropped. With
    ``VectorSpec(scale=None)``, the longest vector in the whole data (all
    control states) is drawn `auto_scale_fill` times the typical spacing
    between drawn positions.

    Attributes:
        colors_by_magnitude (bool): Whether arrows are colored by the vector
            magnitude.
        auto_scale_fill (float): Longest arrow length under autoscaling, as a
            fraction of the typical spacing between drawn arrows.

    Notes:
        Must be combined with a
        :class:`~qewton.visualization.plots.data.base.DataPlot`. The plot must
        set ``self.vector`` (a
        :class:`~qewton.visualization.plots.spec.VectorSpec`) before calling
        ``_init_vector_color()``, and builds its result with
        ``_evaluate_arrows()``.
    """

    def _init_vector_color(self, color: ColorSpec | Variable | bool | None):
        if color is False:
            self.color = None
        elif color is None:
            self.color = ColorSpec(
                self.vector.embedded_selector_spec or self.vector.variable_or_axes
            )
        else:
            self.color = color if isinstance(color, ColorSpec) else ColorSpec(color)
            if not self.colors_by_magnitude:
                self.require_scalar(self.color, "color")

    @property
    def colors_by_magnitude(self) -> bool:
        return (
            self.color is not None
            and self.color.variable_or_axes is self.vector.variable_or_axes
        )

    def _evaluate_arrows(self, data, slice_map, positions: np.ndarray) -> VectorResult:
        """Extracts one vector (and color value) per position from the
        already control-reduced `data`, then applies the VectorSpec's
        normalize/subsample/scale. Arrows whose vector or color is NaN are
        dropped."""
        to_numpy = resolve_backend(data).to_numpy
        positions = np.asarray(resolve_backend(positions).to_numpy(positions))
        n_components = positions.shape[-1]
        slc = self.data_config.get_variable_slice(self.vector.variable_or_axes)
        vectors = np.asarray(to_numpy(data[slice_map(slc)])).reshape(-1, n_components)
        if len(vectors) != len(positions):
            raise ValueError(
                f"{self.vector.name} yields {len(vectors)} vectors but the plot "
                f"has {len(positions)} positions. Unresolved batch dimensions? "
                "Add a SliderSpec or FixedSpec for them."
            )

        magnitude = np.linalg.norm(vectors, axis=1)
        color = None
        if self.colors_by_magnitude:
            # Magnitude of the raw field, before normalize/scale.
            color = magnitude
        elif self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color = np.asarray(to_numpy(data[slice_map(slc)])).reshape(-1)
            if len(color) != len(positions):
                raise ValueError(
                    f"{self.color.name} yields {len(color)} values but the plot "
                    f"has {len(positions)} positions. Unresolved batch "
                    "dimensions? Add a SliderSpec or FixedSpec for them."
                )

        if self.vector.normalize:
            safe = np.where(magnitude > 0, magnitude, 1.0)[:, None]
            vectors = vectors / safe

        keep = np.zeros(len(positions), dtype=bool)
        keep[:: self.vector.subsample] = True
        keep &= ~np.isnan(vectors).any(axis=1)
        if color is not None:
            keep &= ~np.isnan(color)

        scale = self.vector.scale
        if scale is None:
            scale = self._auto_scale(positions[keep], n_components)
        vectors = vectors * scale

        return VectorResult(
            positions=positions[keep],
            vectors=vectors[keep],
            magnitude=magnitude[keep],
            color=color[keep] if color is not None else None,
        )

    auto_scale_fill = 0.8

    def _auto_scale(self, positions: np.ndarray, n_components: int) -> float:
        """VectorSpec(scale=None)'s factor: the longest vector in the whole
        data (every control state) maps to `auto_scale_fill` times the
        typical spacing of the drawn positions. 1.0 if either is zero."""
        if self.vector.normalize:
            reference = 1.0
        else:
            slc = self.data_config.get_variable_slice(self.vector.variable_or_axes)
            full = np.asarray(resolve_backend(self.data).to_numpy(self.data[slc]))
            norms = np.linalg.norm(full.reshape(-1, n_components), axis=1)
            reference = float(np.nanmax(norms)) if np.isfinite(norms).any() else 0.0
        spacing = _typical_spacing(positions)
        if reference <= 0 or spacing <= 0:
            return 1.0
        return self.auto_scale_fill * spacing / reference


def _typical_spacing(positions: np.ndarray, n_probe: int = 128) -> float:
    """Median nearest-neighbor distance among `positions`, estimated from up
    to `n_probe` deterministically chosen probe points against all points.
    0 for fewer than two distinct points."""
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    rng = np.random.default_rng(0)
    probes = positions[rng.choice(len(positions), min(n_probe, len(positions)), replace=False)]
    nearest = []
    for probe in probes:
        dist = np.linalg.norm(positions - probe, axis=1)
        dist = dist[dist > 0]  # the probe itself, and exact duplicates
        if len(dist):
            nearest.append(dist.min())
    return float(np.median(nearest)) if nearest else 0.0
