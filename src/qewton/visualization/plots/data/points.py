import numpy as np

from qewton.config.axes import GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import PointCloudResult
from qewton.visualization.plots.spec import ColorSpec, ControlSpec


class PointCloudPlot(DataPlot):
    """A scalar field colored at arbitrary, unstructured points in space -
    the point-cloud counterpart to MeshFieldPlot (which additionally needs
    cell connectivity) and EmbeddedGridPlot (which additionally needs a
    grid shape). Positions come from `geometry.discretization_points`, with
    no assumption about how they're arranged - the fallback `auto_plot`
    reaches for once a DiscreteGeometry has neither a mesh nor a grid shape
    (e.g. a plain SampledGeometry outside mesh mode).
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
        if points is None or points.shape[-1] not in (2, 3):
            raise ValueError(
                f"{type(self).__name__} requires discretization_points with 2 "
                "or 3 coordinate components."
            )

        self.color = color if isinstance(color, ColorSpec) else ColorSpec(color)
        self.require_scalar(self.color, "color")

    @property
    def embedding_dim(self) -> int:
        points = self.data_config.geometry_axes.geometry.discretization_points
        return points.shape[-1]

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
        color = np.asarray(data[slice_map(slc)]).reshape(-1)

        geometry = self.data_config.geometry_axes.geometry
        points = np.asarray(geometry.discretization_points)
        positions = points.reshape(-1, points.shape[-1])
        if len(positions) != len(color):
            raise ValueError(
                f"{self.color.name} yields {len(color)} values but the "
                f"geometry has {len(positions)} points. Unresolved batch "
                "dimensions? Add a SliderSpec or FixedSpec for them."
            )
        return PointCloudResult(positions=positions, color=color)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return (
            renderer.PointCloud2DArtist.create(backend_figure, self, row=row, col=col)
            if self.embedding_dim == 2
            else renderer.PointCloud3DArtist.create(backend_figure, self, row=row, col=col)
        )
