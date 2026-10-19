"""Plotly artists for point clouds."""

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
    """Legend name: the color's math_name, else the geometry's variable
    name in math mode."""
    if plot.color is not None:
        return plot.color.math_name
    geometry = plot.drawn_geometry_axes.geometry
    return f"${geometry.variable.name}$" if geometry.variable is not None else None


class PointCloud2DArtist(PlotlyArtist):
    """Draws a 1D or 2D point cloud plot.

    Used for :class:`~qewton.visualization.plots.data.points.PointCloudPlot`.
    Points sit at the geometry's own positions, colored by the color field
    or, without one, by the theme's cycled palette color. 1D points are
    drawn at y = 0. Axis titles come from the geometry's coordinate
    variable.

    Args:
        idx (int): Index of the scatter trace.
        row (int, optional): Facet grid row, used to place the colorbar on
            update.
        col (int, optional): Facet grid column, used to place the colorbar
            on update.

    Attributes:
        row (int or None): Facet grid row.
        col (int or None): Facet grid column.
    """

    def __init__(self, idx, row=None, col=None):
        super().__init__(idx)
        self.row, self.col = row, col

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        trace = cls._trace(plot, plot.evaluate(), backend_figure, row, col)
        backend_figure.add_trace(trace, row=row, col=col)
        geometry = plot.drawn_geometry_axes.geometry
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
            cmap = plot.theme.resolve_cmap(plot.color.cmap)
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

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        new = self._trace(plot, plot.evaluate(), backend_figure, self.row, self.col)
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y = new.x, new.y
        trace.marker = new.marker


class PointCloud3DArtist(PlotlyArtist):
    """Draws a 3D point cloud plot as ``go.Scatter3d``.

    Used for :class:`~qewton.visualization.plots.data.points.PointCloudPlot`.
    Same coloring and axis titles as :class:`PointCloud2DArtist`.

    Args:
        idx (int): Index of the scatter trace.
        row (int, optional): Facet grid row, used to place the colorbar on
            update.
        col (int, optional): Facet grid column, used to place the colorbar
            on update.

    Attributes:
        row (int or None): Facet grid row.
        col (int or None): Facet grid column.
    """

    def __init__(self, idx, row=None, col=None):
        super().__init__(idx)
        self.row, self.col = row, col

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        trace = cls._trace(plot, plot.evaluate(), backend_figure, row, col)
        backend_figure.add_trace(trace, row=row, col=col)
        geometry = plot.drawn_geometry_axes.geometry
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
            cmap = plot.theme.resolve_cmap(plot.color.cmap)
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

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        new = self._trace(plot, plot.evaluate(), backend_figure, self.row, self.col)
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y, trace.z = new.x, new.y, new.z
        trace.marker = new.marker
