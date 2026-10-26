"""Data plots on structured grids: heatmaps, surfaces, images, embedded
grids and quiver plots."""

import numpy as np

from qewton.config.axes import Axes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot, VectorFieldPlotMixin
from qewton.visualization.plots.result import (
    GridResult,
    ParametricGridResult,
)
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec, VectorSpec


class StructuredGridPlot(DataPlot):
    """Base class for plots over a regular (x, y) grid of data dimensions.

    Positions are implicit grid indices. Evaluates to a
    :class:`~qewton.visualization.plots.result.GridResult` with the y
    dimension first and the x dimension second.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`.
        x: The data dimension spanning the grid's x axis; must refer to
            exactly one full dimension, different from `y`.
        y: The data dimension spanning the grid's y axis; must refer to
            exactly one full dimension, different from `x`.
        z: The values drawn at each grid point (e.g. surface height). If None,
            the whole control-reduced data is used.
        color: The values coloring each grid point, or None.
        controls: Control specs reducing `data`.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        ValueError: On :meth:`evaluate`, if `x` and `y` refer to the same
            dimension or to a channel slice instead of a dimension.

    Attributes:
        x (AxisSpec): The grid's x axis.
        y (AxisSpec): The grid's y axis.
        z (AxisSpec or None): The value role.
        color (ColorSpec or None): The color role.

    Notes:
        Subclasses must implement
        :meth:`~qewton.visualization.plots.base.Plot.create_artist`.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable | Axes,
        z: AxisSpec | Variable | Axes | None = None,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

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
        """Evaluate the grid in the current control state.

        Returns:
            GridResult: Values (and color) with the y dimension first and the x
                dimension second.
        """
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
    """A 2D array drawn as an image, with `x`/`y` as pixel columns/rows.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`.
        x: The data dimension spanning the image's x axis.
        y: The data dimension spanning the image's y axis.
        controls: Control specs reducing `data`.
    """

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
        """Create the renderer's ``ImageArtist`` for this plot."""
        return renderer.ImageArtist.create(backend_figure, self, row=row, col=col)


class SurfacePlot(StructuredGridPlot):
    """A 2D array drawn as a 3D surface.

    The height is given by `z` (or the data itself if `z` is omitted); the
    surface is optionally colored by `color`. Takes the parameters of
    :class:`StructuredGridPlot`.
    """

    @property
    def embedding_dim(self) -> int:
        return 3

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's ``SurfaceArtist`` for this plot."""
        return renderer.SurfaceArtist.create(backend_figure, self, row=row, col=col)


class HeatmapPlot(StructuredGridPlot):
    """A 2D array drawn as a flat, colored grid.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`.
        x: The data dimension spanning the grid's x axis.
        y: The data dimension spanning the grid's y axis.
        color: The values coloring each cell. If None, the data itself is used.
        controls: Control specs reducing `data`.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,
        y: AxisSpec | Variable | Axes,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(
            data, data_config, x, y, color=color, controls=controls, **kwargs
        )

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's ``HeatmapArtist`` for this plot."""
        return renderer.HeatmapArtist.create(backend_figure, self, row=row, col=col)


class EmbeddedGridPlot(DataPlot):
    """A structured grid embedded in 3D space, colored by a scalar field.

    Positions are the explicit 3D points of the drawn geometry's
    ``discretization_points``, so any parametric surface grid (e.g. a
    sphere parametrization or a curved shell) is drawn at its true
    position. The geometry may have more than two dimensions if controls
    reduce it to two; color data and coordinates are reduced the same way.
    Evaluates to a
    :class:`~qewton.visualization.plots.result.ParametricGridResult`.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`; must have exactly one
            drawn GeometryAxes.
        color: The scalar field coloring the surface.
        controls: Control specs reducing `data`.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        AssertionError: If there is not exactly one drawn GeometryAxes.
        ValueError: If the geometry has no 3-component
            ``discretization_points``, two grid dimensions do not remain after
            controls, `color` is not scalar, or (on :meth:`evaluate`) the color
            data does not match the grid shape.

    Attributes:
        color (ColorSpec): The color role.
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

        geom_axes = self.drawn_geometry_axes
        assert isinstance(
            geom_axes, GeometryAxes
        ), "Exactly one GeometryAxes without a control is supported."
        geometry = geom_axes.geometry
        points = geometry.discretization_points
        if points is None or points.shape[-1] != 3:
            raise ValueError(
                f"{type(self).__name__} requires discretization_points with 3 "
                "coordinate components. A grid with only index coordinates "
                "should use HeatmapPlot instead."
            )

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
        """Evaluate color and positions in the current control state.

        Returns:
            ParametricGridResult: Grid values and 3D coordinates, each of shape
                ``(N1, N2)``.
        """
        data, index_map, slice_map = self.apply_controls()
        slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
        values = np.asarray(data[slice_map(slc)])

        geometry = self.drawn_geometry_axes.geometry
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
        """Create the renderer's ``ParametricSurfaceArtist`` for this plot."""
        return renderer.ParametricSurfaceArtist.create(
            backend_figure, self, row=row, col=col
        )


class QuiverPlot(VectorFieldPlotMixin, DataPlot):
    """A vector field on a structured grid, drawn as arrows.

    Positions are the explicit 2D or 3D points of the drawn geometry's
    ``discretization_points``, reduced by controls like the data (as in
    :class:`EmbeddedGridPlot`). Color handling, NaN dropping and
    autoscaling follow
    :class:`~qewton.visualization.plots.data.base.VectorFieldPlotMixin`.
    Evaluates to a
    :class:`~qewton.visualization.plots.result.VectorResult`.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`.
        vector: The vector field; its number of components must equal the
            number of coordinate components of the grid.
        color: None to color by magnitude, a scalar ColorSpec or Variable to
            color by that field, or False for uncolored arrows.
        controls: Control specs reducing `data`.
        n_bins: Number of distinct colors the arrows are drawn in.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        ValueError: If the geometry has no 2- or 3-component
            ``discretization_points``, or the vector's component count does not
            match.

    Attributes:
        vector (VectorSpec): The vector role.
        color (ColorSpec or None): The color role, None for uncolored arrows.
        n_bins (int): Number of distinct arrow colors.
        embedding_dim (int): 2 or 3, the number of coordinate components of the
            grid.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        vector: VectorSpec | Variable,
        color: ColorSpec | Variable | bool | None = None,
        controls: list[ControlSpec] | None = None,
        n_bins: int = 32,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geometry = self.drawn_geometry_axes.geometry
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
        self._init_vector_color(color)
        self.n_bins = n_bins

    @property
    def embedding_dim(self) -> int:
        return self._coord_dim

    def evaluate(self):
        """Evaluate the arrows in the current control state.

        Returns:
            VectorResult: Arrow positions, vectors, magnitudes and colors.
        """
        data, index_map, slice_map = self.apply_controls()
        geometry = self.drawn_geometry_axes.geometry
        points = self.reduce_coordinates(
            geometry.discretization_points, self._geometry_dims()
        ).reshape(-1, self._coord_dim)
        return self._evaluate_arrows(data, slice_map, points)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's 2D or 3D arrow-field artist for this plot."""
        return (
            renderer.ArrowField2DArtist.create(backend_figure, self, row=row, col=col)
            if self._coord_dim == 2
            else renderer.ArrowField3DArtist.create(
                backend_figure, self, row=row, col=col
            )
        )
