import numpy as np
from plotly import graph_objects as go

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    PlotlyArtist,
    _apply_scale,
    _cycled_color,
    _spatial_variable,
    _to_numpy,
)


def _point_cloud_name(plot) -> str | None:
    """The legend entry for a point cloud trace - the colored quantity's
    own math_name when there is one, else the geometry's own coordinate
    Variable, math-wrapped to match axis titles."""
    if plot.color is not None:
        return plot.color.math_name
    geometry = plot.data_config.geometry_axes.geometry
    return f"${geometry.variable.name}$" if geometry.variable is not None else None


class PointCloud2DArtist(PlotlyArtist):
    """Points at arbitrary (x, y) positions from a geometry's own
    discretization_points, colored by a scalar field - the geometry-
    positioned counterpart to ScatterArtist (which draws named x/y roles
    instead of a geometry's own coordinates). A 1D point cloud is drawn the
    same way, baselined at y=0."""

    def __init__(self, idx, row=None, col=None):
        super().__init__(idx)
        self.row, self.col = row, col

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        trace = cls._trace(plot, plot.evaluate(), backend_figure, row, col)
        backend_figure.add_trace(trace, row=row, col=col)
        geometry = plot.data_config.geometry_axes.geometry
        if plot.coordinate_dim == 1:
            x_name = axis_names_from_variable(_spatial_variable(geometry), 1)[0]
            y_name = ""
        else:
            x_name, y_name = axis_names_from_variable(_spatial_variable(geometry), 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        return cls(len(backend_figure.data) - 1, row, col)

    @staticmethod
    def _trace(plot, result, backend_figure=None, row=None, col=None):
        positions = _to_numpy(result.positions)
        y = positions[:, 1] if positions.shape[-1] > 1 else np.zeros(positions.shape[0])
        marker = dict(size=plot.theme.marker_size)
        if result.color is not None:
            cmap = plot.color.cmap or plot.theme.default_cmap
            marker.update(color=_to_numpy(result.color), colorscale=cmap)
            marker.update(
                _apply_scale(plot.color.scale, backend_figure=backend_figure, row=row, col=col)
            )
        else:
            marker.update(color=_cycled_color(plot))
        return go.Scatter(
            x=positions[:, 0],
            y=y,
            mode="markers",
            name=plot.label or _point_cloud_name(plot),
            marker=marker,
            opacity=plot.theme.opacity_default,
        )

    def update(self, backend_figure, plot):
        new = self._trace(plot, plot.evaluate(), backend_figure, self.row, self.col)
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y = new.x, new.y
        trace.marker = new.marker


class PointCloud3DArtist(PlotlyArtist):
    """3D counterpart to PointCloud2DArtist, via go.Scatter3d."""

    def __init__(self, idx, row=None, col=None):
        super().__init__(idx)
        self.row, self.col = row, col

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        trace = cls._trace(plot, plot.evaluate(), backend_figure, row, col)
        backend_figure.add_trace(trace, row=row, col=col)
        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name, z_name = axis_names_from_variable(_spatial_variable(geometry), 3)
        backend_figure.update_scenes(
            row=row, col=col,
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name),
        )
        return cls(len(backend_figure.data) - 1, row, col)

    @staticmethod
    def _trace(plot, result, backend_figure=None, row=None, col=None):
        positions = _to_numpy(result.positions)
        marker = dict(size=plot.theme.marker_size)
        if result.color is not None:
            cmap = plot.color.cmap or plot.theme.default_cmap
            marker.update(color=_to_numpy(result.color), colorscale=cmap)
            marker.update(
                _apply_scale(plot.color.scale, backend_figure=backend_figure, row=row, col=col)
            )
        else:
            marker.update(color=_cycled_color(plot))
        return go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            name=plot.label or _point_cloud_name(plot),
            marker=marker,
            opacity=plot.theme.opacity_default,
        )

    def update(self, backend_figure, plot):
        new = self._trace(plot, plot.evaluate(), backend_figure, self.row, self.col)
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y, trace.z = new.x, new.y, new.z
        trace.marker = new.marker
