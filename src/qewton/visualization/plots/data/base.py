import numpy as np

from qewton.backends import resolve_backend
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.result import VectorResult
from qewton.visualization.plots.spec import ColorSpec, ControlSpec, PlotSpec


class CoordTransform:
    """Transforms a plot's evaluated point positions before they're drawn."""

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityCoord(CoordTransform):
    """A CoordTransform that leaves point positions unchanged."""

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        return vertices


class DataPlot(Plot):
    """Base class for a Plot whose values come from `data` + a
    `DataConfiguration` - the most common input family, underlying the grid,
    mesh, curve and scatter plot families."""

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

    def _resolve_controls(self) -> list[tuple[ControlSpec, int]]:
        """(spec, dim index in the original self.data) for each control spec,
        sorted by dim index descending - so collapsing one dimension during
        apply_controls()/reduce_coordinates() never shifts the indices still
        to be processed. Shared by both, since a control on a geometry
        dimension must reduce data and geometry coordinates the same way."""
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
        """Slices `self.data` down to the current Fixed/Slider/Facet control
        states, one dimension removed per control.

        Returns:
            tuple: `(sliced, index_map, slice_map)` - the reduced array,
                plus two helpers to translate positions computed against the
                original (unreduced) `self.data`/`self.data_config` into
                positions in `sliced`: `index_map(original_idx)` for a
                single dimension index, `slice_map(slc)` for a slice tuple
                (e.g. from `DataConfiguration.get_variable_slice`).
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
            """Maps a slice tuple computed against the original
            `data_config` (one entry per original dimension, e.g. from
            `get_variable_slice`) onto `sliced`'s already-reduced
            dimensions, by dropping the entries the controls consumed."""
            if Ellipsis in slc:
                raise NotImplementedError(
                    "slice_map does not yet support slices containing Ellipsis."
                )
            return tuple(s for i, s in enumerate(slc) if i not in removed_indices)

        return sliced, index_map, slice_map

    def reduce_coordinates(self, points, geometry_dims: tuple[int, int]):
        """Applies the same control reduction as apply_controls() to an array
        indexed by the geometry's own dimensions (e.g. discretization_points).

        `apply_controls()` only slices `self.data`; a geometry's coordinates
        live in a separate array and are otherwise never touched, so a control
        on a geometry dimension would desynchronize values from coordinates
        (data reduced, coordinates still describing every state). A control
        that reduces a geometry dimension must reduce the geometry coordinates
        the same way.

        Args:
            points: Array indexed by the geometry's dimensions, plus a
                trailing coordinate-component axis that is never reduced
                (e.g. discretization_points, shape (..., n_components)).
            geometry_dims: (start, stop) range the geometry's dimensions
                occupy in self.data - see _geometry_dims().
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

    def _geometry_dims(self) -> tuple[int, int]:
        """(start, stop) range in self.data that this plot's GeometryAxes
        occupies, for use with reduce_coordinates()."""
        geom_axes = self.data_config.geometry_axes
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
        dimension (not a channel slice) to its index in self.data. Shared by
        every plot family with a structural axis role (StructuredGridPlot's
        x/y, LinePlot's x)."""
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
        """Number of feature components a spec resolves to (1 means scalar).

        Used to reject vector variables where a scalar is required, which would
        otherwise silently render only the first component. Shared by every
        plot family that can take either a scalar or vector spec (MeshPlot's
        color/z, EmbeddedGridPlot's color).
        """
        var = spec.variable_or_axes
        if isinstance(var, Variable):
            return var.dim
        slc = data_config.get_variable_slice(var)
        last = slc[-1] if isinstance(slc, tuple) else slc
        return (last.stop - last.start) if isinstance(last, slice) else 1

    def require_scalar(self, spec, role: str):
        n = self.component_count(spec, self.data_config)
        if n != 1:
            raise ValueError(
                f"{role} must be scalar (dim=1), got dim={n}. "
                "Select a single component, or use a vector-aware plot type."
            )


class VectorFieldPlotMixin:
    """Shared vector and color handling for arrow-field plots - QuiverPlot
    (grid points) and MeshVectorPlot (mesh vertices) differ only in where
    the arrow positions come from.

    Expects `self.vector` (a VectorSpec) to be set before
    `_init_vector_color()` runs. The `color` role then resolves as:

    - None (default): colors by vector magnitude, i.e. a ColorSpec on the
      vector's own Variable - pass `ColorSpec(vector_variable, cmap=...,
      scale=...)` explicitly to keep magnitude coloring with a custom
      colormap or a shared Scale.
    - a scalar ColorSpec/Variable: colors each arrow by that field instead,
      the same way SurfacePlot decouples its color from its height.
    - False: flat, uncolored arrows in the theme's `vector_color`.
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
        normalize/scale/subsample.

        Arrows whose vector or color is NaN (e.g. grid points a
        MeshInterpolationNode's point_filter excluded) are dropped entirely
        rather than drawn as degenerate zero-length arrows.
        """
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
            # The magnitude of the raw field, before normalize/scale - those
            # only change how long an arrow is drawn, not what it encodes.
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

        # Decimate after normalize, not before: subsampling only changes
        # which arrows are drawn, never what a drawn arrow means.
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

    #: Longest arrow length under autoscaling, as a fraction of the typical
    #: spacing between drawn arrows - a bit below 1, so neighboring arrows
    #: (and 3D cones, whose width grows with their length) don't touch.
    auto_scale_fill = 0.8

    def _auto_scale(self, positions: np.ndarray, n_components: int) -> float:
        """VectorSpec(scale=None)'s factor: the longest vector in the WHOLE
        data (every control state, not just the drawn one) maps to
        `auto_scale_fill` times the typical spacing of the drawn positions -
        so arrow lengths stay comparable while a slider moves."""
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
    to `n_probe` probe points (deterministically chosen) against all of
    them - O(n_probe * n) instead of a full O(n^2) distance matrix, and
    robust to how the points are arranged (a grid, a planar slice embedded
    in 3D, or mesh vertices). 0 for fewer than two distinct points."""
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
