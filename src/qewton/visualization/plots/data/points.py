import numpy as np

from qewton.config.axes import GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import PointCloudResult
from qewton.visualization.plots.spec import ColorSpec, ControlSpec
from qewton.backends import resolve_backend


class PointCloudPlot(DataPlot):
    """Arbitrary, unstructured points in space (1D, 2D or 3D), optionally
    colored by a scalar field - the point-cloud counterpart to
    MeshFieldPlot (which additionally needs cell connectivity) and
    EmbeddedGridPlot (which additionally needs a grid shape). Positions
    come from `geometry.discretization_points` (or `geometry.mesh.vertices`
    if unset), with no assumption about how they're arranged - the fallback
    `auto_plot` reaches for once a DiscreteGeometry has neither a mesh nor
    a grid shape (e.g. a plain SampledGeometry outside mesh mode), or when
    there's no separate quantity to color by at all (the geometry's own
    points, e.g. a PointSampler's own output). A 1D point cloud draws on a
    plain 2D x/y axes, baselined at y=0.
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

        geom_axes = data_config.geometry_axes
        assert isinstance(
            geom_axes, GeometryAxes
        ), "Currently only DataConfigurations with a single GeometryAxes are supported."
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
        """The geometry's own number of coordinate components (1, 2, or 3) -
        distinct from `embedding_dim`, which is the chart's own
        dimensionality (a 1D point cloud still draws on a plain 2D chart,
        baselined at y=0)."""
        points = self._resolve_points(self.data_config.geometry_axes.geometry)
        return points.shape[-1]

    @property
    def embedding_dim(self) -> int:
        return 3 if self.coordinate_dim == 3 else 2

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()

        geometry = self.data_config.geometry_axes.geometry
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
        return (
            renderer.PointCloud3DArtist.create(backend_figure, self, row=row, col=col)
            if self.embedding_dim == 3
            else renderer.PointCloud2DArtist.create(
                backend_figure, self, row=row, col=col
            )
        )
