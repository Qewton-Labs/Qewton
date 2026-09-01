import numpy as np

from qewton.config.axes import Axes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import GridResult, ParametricGridResult, VectorResult
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec, VectorSpec


class StructuredGridPlot(DataPlot):
    """Base class for plots over a regular (x, y) meshgrid - heatmaps,
    surfaces, images. `x`/`y` each select one structural dimension of
    `data`; an optional `z`/`color` supplies the values drawn at each grid
    point."""

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable | Axes,
        z: AxisSpec | Variable | Axes | None = None,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, controls=controls)

        self.x = x if isinstance(x, AxisSpec) else AxisSpec(x)
        self.y = y if isinstance(y, AxisSpec) else AxisSpec(y)

        if isinstance(z, AxisSpec):
            self.z = z
        elif z is not None:
            self.z = AxisSpec(z)
        else:
            self.z = None

        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color
            else None
        )

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()

        # 1) X/Y structured - resolve to original self.data, then map to the
        #    already reduced `data`.
        x_idx = self._resolve_structural_dim(self.x)
        y_idx = self._resolve_structural_dim(self.y)

        if x_idx == y_idx:
            raise ValueError(
                f"{type(self).__name__}: x ({self.x.variable_or_axes}) and "
                f"y ({self.y.variable_or_axes}) refer to the same dimension. "
                "You might use a ScatterPlot or MeshPlot instead."
            )

        x_dim = index_map(x_idx)
        y_dim = index_map(y_idx)

        # 2) Color/Values BEFORE transpose - get_variable_slice returns a
        #    slice tuple for the original data_config, so map to the already
        #    reduced dimensions of `data` before applying it.
        values = data
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color = data[slice_map(slc)]
        else:
            color = None

        if self.z is not None:
            slc = self.data_config.get_variable_slice(self.z.variable_or_axes)
            values = values[slice_map(slc)]

        # 3) X/Y-Dimensions at the beginning (y, x, ...remaining dims)
        oriented = np.moveaxis(values, [y_dim, x_dim], [0, 1])
        if color is not None:
            color = np.moveaxis(color, [y_dim, x_dim], [0, 1])

        return GridResult(values=oriented, color=color)


class ImagePlot(StructuredGridPlot):
    """A 2D array drawn as an image, with `x`/`y` as pixel rows/columns."""

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable | Axes,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, controls=controls)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ImageArtist.create(backend_figure, self, row=row, col=col)


class SurfacePlot(StructuredGridPlot):
    """A 2D array drawn as a 3D surface, height given by `z` (or the data
    itself if `z` is omitted) and optionally colored by `color`."""

    @property
    def embedding_dim(self) -> int:
        return 3

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.SurfaceArtist.create(backend_figure, self, row=row, col=col)


class HeatmapPlot(StructuredGridPlot):
    """A 2D array drawn as a flat, colored grid (a heatmap)."""

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable | Axes,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, color=color, controls=controls)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.HeatmapArtist.create(backend_figure, self, row=row, col=col)


class EmbeddedGridPlot(DataPlot):
    """A structured grid embedded in 3D space, colored by a scalar field.

    Unlike HeatmapPlot, coordinates are not axis indices but explicit 3D
    points from `geometry.discretization_points`, so the grid is drawn at
    its true position and orientation in space - any parametric surface
    grid (e.g. a sphere parametrization or a curved shell), static or
    control-dependent.

    The geometry may have more than two dimensions: controls reduce it, so a
    `(k, N1, N2)` stack of slices with a SliderSpec/FixedSpec on the first
    dimension leaves an `(N1, N2)` surface to draw. Both the color data and
    the geometry's own coordinates are reduced the same way, so the drawn
    position moves correctly with control state instead of staying fixed
    while only the color changes.

    Positions come entirely from the geometry, so `color` is the only spec -
    there is no x/y role for the user to assign, unlike StructuredGridPlot.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        color: ColorSpec | Variable,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geom_axes = data_config.geometry_axes
        assert isinstance(
            geom_axes, GeometryAxes
        ), "Currently only DataConfigurations with a single GeometryAxes are supported."
        geometry = geom_axes.geometry
        points = geometry.discretization_points
        if points is None or points.shape[-1] != 3:
            raise ValueError(
                f"{type(self).__name__} requires discretization_points with 3 "
                "coordinate components. A grid with only index coordinates "
                "should use HeatmapPlot instead."
            )

        # Controls and geometry shape are both known here, so validate the
        # reduced dimensionality now rather than deferring to evaluate().
        n_reduced = self._count_controls_on_geometry_dims()
        remaining = len(geometry.shape) - n_reduced
        if remaining != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly 2 grid dimensions left "
                f"after controls, got {remaining} (geometry shape "
                f"{geometry.shape}, {n_reduced} reduced by controls). "
                "Add a SliderSpec or FixedSpec for the extra dimensions."
            )

        self.color = color if isinstance(color, ColorSpec) else ColorSpec(color)
        self.require_scalar(self.color, "color")

    @property
    def embedding_dim(self) -> int:
        return 3

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
        values = np.asarray(data[slice_map(slc)])

        geometry = self.data_config.geometry_axes.geometry
        points = self.reduce_coordinates(
            geometry.discretization_points, self._geometry_dims()
        )  # -> (N1, N2, 3)

        expected = points.shape[:2]
        if values.shape[:2] != expected:
            raise ValueError(
                f"{self.color.name} has grid shape {values.shape[:2]}, expected "
                f"{expected}. Unresolved batch dimensions? Add a SliderSpec or "
                "FixedSpec for them."
            )
        values = values.reshape(expected)
        return ParametricGridResult(
            values=values,
            x=points[..., 0],
            y=points[..., 1],
            z=points[..., 2],
            color=values,
        )

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.ParametricSurfaceArtist.create(
            backend_figure, self, row=row, col=col
        )


class QuiverPlot(DataPlot):
    """A vector field on a structured grid, drawn as arrows - the grid
    counterpart to MeshVectorPlot (mesh vertices) and EmbeddedGridPlot
    (scalar color on a grid). Draws any structured grid whose coordinates
    are explicit 2D or 3D points, e.g. a mesh field resampled onto a
    VolumeGridGeometry via a MeshInterpolationNode.

    Positions come from `geometry.discretization_points`, reduced by the
    same mechanism EmbeddedGridPlot uses, so a control on an extra grid
    dimension moves the drawn arrow positions correctly, not just the
    vectors.

    Grid points a MeshInterpolationNode's `point_filter` excluded (outside
    the source mesh) come back NaN in every vector component - those points
    are dropped entirely rather than drawn as degenerate zero-length arrows.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        vector: VectorSpec | Variable,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geometry = data_config.geometry_axes.geometry
        points = geometry.discretization_points
        if points is None or points.shape[-1] not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} requires discretization_points with 2 "
                "or 3 coordinate components."
            )
        self._coord_dim = points.shape[-1]

        self.vector = vector if isinstance(vector, VectorSpec) else VectorSpec(vector)
        n_components = self.component_count(self.vector, data_config)
        if n_components != self._coord_dim:
            raise ValueError(
                f"vector has {n_components} components, {type(self).__name__} "
                f"needs exactly {self._coord_dim} (the grid is embedded in "
                f"{self._coord_dim}D space)."
            )

    @property
    def embedding_dim(self) -> int:
        return self._coord_dim

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        slc = self.data_config.get_variable_slice(self.vector.variable_or_axes)
        vectors = np.asarray(data[slice_map(slc)]).reshape(-1, self._coord_dim)

        geometry = self.data_config.geometry_axes.geometry
        points = self.reduce_coordinates(
            geometry.discretization_points, self._geometry_dims()
        ).reshape(-1, self._coord_dim)
        if len(points) != len(vectors):
            raise ValueError(
                f"{self.vector.name} yields {len(vectors)} vectors but the grid "
                f"has {len(points)} points. Unresolved batch dimensions? Add a "
                "SliderSpec or FixedSpec for them."
            )

        magnitude = np.linalg.norm(vectors, axis=1)
        if self.vector.normalize:
            safe = np.where(magnitude > 0, magnitude, 1.0)[:, None]
            vectors = vectors / safe
        vectors = vectors * self.vector.scale

        step = self.vector.subsample
        if step > 1:
            points, vectors, magnitude = points[::step], vectors[::step], magnitude[::step]

        # Points a MeshInterpolationNode's point_filter excluded come back NaN
        # in every component - drop them rather than draw zero-length arrows.
        valid = ~np.isnan(vectors).any(axis=1)
        points, vectors, magnitude = points[valid], vectors[valid], magnitude[valid]

        return VectorResult(positions=points, vectors=vectors, magnitude=magnitude)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return (
            renderer.ArrowField2DArtist.create(backend_figure, self, row=row, col=col)
            if self._coord_dim == 2
            else renderer.ArrowField3DArtist.create(backend_figure, self, row=row, col=col)
        )
