from plotly import graph_objects as go

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    PlotlyArtist,
    _apply_scale,
    _cycled_color,
    _spatial_variable,
    _to_numpy,
)


class PointCloud2DArtist(PlotlyArtist):
    """Points at arbitrary (x, y) positions from a geometry's own
    discretization_points, colored by a scalar field - the geometry-
    positioned counterpart to ScatterArtist (which draws named x/y roles
    instead of a geometry's own coordinates)."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        trace = cls._trace(plot, plot.evaluate())
        backend_figure.add_trace(trace, row=row, col=col)
        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name = axis_names_from_variable(_spatial_variable(geometry), 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        return cls(len(backend_figure.data) - 1)

    @staticmethod
    def _trace(plot, result):
        positions = _to_numpy(result.positions)
        marker = dict(size=plot.theme.marker_size)
        if result.color is not None:
            cmap = plot.color.cmap or plot.theme.default_cmap
            marker.update(color=_to_numpy(result.color), colorscale=cmap)
            marker.update(_apply_scale(plot.color.scale))
        else:
            marker.update(color=_cycled_color(plot))
        return go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers",
            marker=marker,
            opacity=plot.theme.opacity_default,
        )

    def update(self, backend_figure, plot):
        new = self._trace(plot, plot.evaluate())
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y = new.x, new.y
        trace.marker = new.marker


class PointCloud3DArtist(PlotlyArtist):
    """3D counterpart to PointCloud2DArtist, via go.Scatter3d."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        trace = cls._trace(plot, plot.evaluate())
        backend_figure.add_trace(trace, row=row, col=col)
        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name, z_name = axis_names_from_variable(_spatial_variable(geometry), 3)
        backend_figure.update_scenes(
            row=row, col=col,
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name),
        )
        return cls(len(backend_figure.data) - 1)

    @staticmethod
    def _trace(plot, result):
        positions = _to_numpy(result.positions)
        marker = dict(size=plot.theme.marker_size)
        if result.color is not None:
            cmap = plot.color.cmap or plot.theme.default_cmap
            marker.update(color=_to_numpy(result.color), colorscale=cmap)
            marker.update(_apply_scale(plot.color.scale))
        else:
            marker.update(color=_cycled_color(plot))
        return go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=marker,
            opacity=plot.theme.opacity_default,
        )

    def update(self, backend_figure, plot):
        new = self._trace(plot, plot.evaluate())
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y, trace.z = new.x, new.y, new.z
        trace.marker = new.marker
