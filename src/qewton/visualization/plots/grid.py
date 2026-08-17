import numpy as np

from qewton.config.axes import Axes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.result import GridResult
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec


class StructuredGridPlot(Plot):
    """Heatmap, Surface, Contour - using meshgrids"""

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
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
                f"{type(self).__name__}: x ({self.x.variable_or_axes}) und "
                f"y ({self.y.variable_or_axes}) refer to the same dimension. "
                "You might use an PointPlot or MeshPlot instead."
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

        # could be moved to resolve also
        # self.x.coordinates = self._coordinates_for(self.x)
        # self.y.coordinates = self._coordinates_for(self.y)
        return GridResult(values=oriented, color=color)


class ImagePlot(StructuredGridPlot):
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, controls=controls)

    def create_artist(self, backend_figure, renderer):
        return renderer.ImageArtist.create(backend_figure, self)


class SurfacePlot(StructuredGridPlot):
    def create_artist(self, backend_figure, renderer):
        return renderer.SurfaceArtist.create(backend_figure, self)


class HeatmapPlot(StructuredGridPlot):
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, color=color, controls=controls)

    def create_artist(self, backend_figure, renderer):
        return renderer.HeatmapArtist.create(backend_figure, self)


class EmbeddedGridPlot(Plot):
    """Structured grid embedded in 3D space, colored by a scalar field.

    Unlike HeatmapPlot, coordinates are not axis indices but explicit 3D
    points from geometry.discretization_points, so the grid is drawn at its
    true position and orientation in space. Not slice-specific - the same
    plot draws any parametric surface grid (sphere parametrization, curved
    shell), static or control-dependent.

    The geometry may have more than two dimensions: controls reduce it, so a
    (k, N1, N2) stack of slices with a SliderSpec/FixedSpec on the first
    dimension leaves a (N1, N2) surface to draw - both the color data and the
    geometry's own coordinates are reduced the same way (reduce_coordinates()),
    so the drawn position moves correctly with control state instead of
    staying fixed while only the color changes.

    Structurally the grid counterpart of MeshFieldPlot: positions come
    entirely from the geometry, so color is the only spec. There is no x/y
    role to assign (the grid dimensions are fixed by the geometry, not chosen
    by the user), so this extends Plot directly rather than
    StructuredGridPlot, whose x/y roles this plot has no use for.
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
        return GridResult(
            values=values,
            x=points[..., 0],
            y=points[..., 1],
            z=points[..., 2],
            color=values,
        )

    def create_artist(self, backend_figure, renderer):
        return renderer.ParametricSurfaceArtist.create(backend_figure, self)
