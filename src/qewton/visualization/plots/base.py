import itertools

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.spec import PlotSpec, ControlSpec, SliderSpec


class CoordTransform:
    """transform points AFTER evaluate()"""

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityCoord(CoordTransform):
    def apply(self, vertices: np.ndarray) -> np.ndarray:
        return vertices


class Plot:
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        title=None,
        theme=None,
        controls: list[ControlSpec] | None = None,
        coord_transform: CoordTransform | None = None,
    ) -> None:
        self.data = data
        self.data_config = data_config
        self._title = title
        self._theme = theme
        self.controls = controls or []
        self.coord_transform = coord_transform or IdentityCoord()

        for spec in self.controls:
            spec.resolve(self.data_config, self.data)

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        if self._theme is None:
            self._theme = value

    def preprocess(self, data):
        return data

    @property
    def title(self):
        return self._title

    def evaluate(self):
        raise NotImplementedError

    def create_artist(self, backend_figure, renderer):
        # uses self.data_config.evaluate_data(self.data, self.data_config)
        # to make it use the current state of the PlotAxis
        raise NotImplementedError

    def color_values(self):
        """Values to train this plot's shared color Scale, or None.

        None both when the plot has no ColorSpec and when its ColorSpec has
        no Scale attached - the common case, letting Figure.draw() skip the
        extra evaluate() call entirely. Generic across plot families because
        every evaluate() returns a result dataclass with a `.color` field.

        For a plot with SliderSpec controls, trains on every slider state, not
        just whichever is current when Figure.draw() runs - otherwise the
        colorbar would silently reset to whatever's on screen at redraw time
        (jumping around during scrubbing) instead of staying fixed. FixedSpec
        controls are left alone: they never move, so only their fixed state
        is ever drawn.
        """
        spec = getattr(self, "color", None)
        if spec is None or spec.scale is None:
            return None

        sliders = [c for c in self.controls if isinstance(c, SliderSpec)]
        if not sliders:
            return getattr(self.evaluate(), "color", None)

        originals = [s.state for s in sliders]
        try:
            all_values = []
            state_ranges = (range(s.minimum, s.maximum + 1, s.step) for s in sliders)
            for states in itertools.product(*state_ranges):
                for slider, state in zip(sliders, states):
                    slider.state = state
                values = getattr(self.evaluate(), "color", None)
                if values is not None:
                    all_values.append(np.asarray(values).reshape(-1))
            return np.concatenate(all_values) if all_values else None
        finally:
            for slider, original in zip(sliders, originals):
                slider.state = original

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
            if isinstance(axis_slc, slice):
                # A length-1 slice unambiguously names one dimension - e.g. a
                # control on one child variable of a multi-dim GeometryAxes
                # (the same mechanism HeatmapPlot's x/y already resolve
                # through), not a multi-axis control.
                length = axis_slc.stop - axis_slc.start
                assert length == 1, (
                    f"No multi-axis support for controls yet: {spec.variable_or_axes}."
                )
                axis_slc = axis_slc.start
            assert isinstance(
                axis_slc, int
            ), f"No multi-axis support for controls yet: {spec.variable_or_axes}."
            real_idx = axis_slc if axis_slc >= 0 else len(self.data.shape) + axis_slc
            resolved.append((spec, real_idx))

        resolved.sort(key=lambda item: item[1], reverse=True)
        return resolved

    def apply_controls(self):
        """Wendet Fixed/Slider/Facet-States an. Gibt zusaetzlich eine index_map
        zurueck, um andere, gegen das URSPRUENGLICHE `data` berechnete reale
        Dimensionsindizes (z.B. aus PlotSpec.get_slice) auf ihre Position im
        bereits reduzierten Array umzurechnen."""
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
            """Rechnet ein gegen das URSPRUENGLICHE `data_config` berechnetes
            Slice-Tupel (ein Eintrag pro urspruenglicher Dimension, z.B. aus
            get_variable_slice) auf die bereits durch die Controls reduzierten
            Dimensionen von `sliced` um, indem die von den Controls
            konsumierten Eintraege entfernt werden."""
            if Ellipsis in slc:
                raise NotImplementedError(
                    "slice_map unterstuetzt noch keine Slices mit Ellipsis."
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
