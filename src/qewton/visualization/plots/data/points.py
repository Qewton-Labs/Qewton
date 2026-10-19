"""Data plot of unstructured point clouds."""

import numpy as np

from qewton.config.axes import GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import PointCloudResult
from qewton.visualization.plots.spec import ColorSpec, ControlSpec
from qewton.backends import resolve_backend


class PointCloudPlot(DataPlot):
    """Unstructured points in space (1D, 2D or 3D), optionally colored by a
    scalar field.

    Positions are the drawn geometry's ``discretization_points``, or its
    ``mesh.vertices`` if those are unset, with no assumption about their
    arrangement. A 1D point cloud is drawn on plain 2D axes at y=0.
    Evaluates to a
    :class:`~qewton.visualization.plots.result.PointCloudResult`.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`; must have exactly one
            drawn GeometryAxes.
        color: The scalar field coloring the points, or None.
        controls: Control specs reducing `data`.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        AssertionError: If there is not exactly one drawn GeometryAxes.
        ValueError: If the geometry has no points with 1, 2 or 3 components,
            `color` is not scalar, or (on :meth:`evaluate`) the number of color
            values does not match the number of points.

    Attributes:
        color (ColorSpec or None): The color role.
        coordinate_dim (int): Number of coordinate components of the points (1,
            2 or 3).
        embedding_dim (int): 3 for 3D points, else 2.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geom_axes = self.drawn_geometry_axes
        assert isinstance(
            geom_axes, GeometryAxes
        ), "Exactly one GeometryAxes without a control is supported."
        points = self._resolve_points(geom_axes.geometry)
        if points is None or points.shape[-1] not in (1, 2, 3):
            raise ValueError(
                f"{type(self).__name__} requires discretization_points with 1, "
                "2 or 3 coordinate components."
            )

        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color is not None
            else None
        )
        if self.color is not None:
            self.require_scalar(self.color, "color")

    @staticmethod
    def _resolve_points(geometry):
        points = geometry.discretization_points
        if points is None:
            mesh = getattr(geometry, "mesh", None)
            if mesh is not None:
                points = mesh.vertices
        return points

    @property
    def coordinate_dim(self) -> int:
        points = self._resolve_points(self.drawn_geometry_axes.geometry)
        return points.shape[-1]

    @property
    def embedding_dim(self) -> int:
        return 3 if self.coordinate_dim == 3 else 2

    def evaluate(self):
        """Evaluate positions and colors in the current control state.

        Returns:
            PointCloudResult: Point positions and one color value per point (or
                None).
        """
        data, index_map, slice_map = self.apply_controls()

        geometry = self.drawn_geometry_axes.geometry
        raw_points = self._resolve_points(geometry)
        points = (
            raw_points
            if isinstance(raw_points, np.ndarray)
            else np.asarray(geometry.backend.to_numpy(raw_points))
        )
        positions = points.reshape(-1, points.shape[-1])

        data_backend = resolve_backend(data)
        color = None
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color = np.asarray(data_backend.to_numpy(data[slice_map(slc)])).reshape(-1)
            if len(positions) != len(color):
                raise ValueError(
                    f"{self.color.name} yields {len(color)} values but the "
                    f"geometry has {len(positions)} points. Unresolved batch "
                    "dimensions? Add a SliderSpec or FixedSpec for them."
                )
        return PointCloudResult(positions=positions, color=color)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's 2D or 3D point-cloud artist for this
        plot."""
        return (
            renderer.PointCloud3DArtist.create(backend_figure, self, row=row, col=col)
            if self.embedding_dim == 3
            else renderer.PointCloud2DArtist.create(
                backend_figure, self, row=row, col=col
            )
        )
