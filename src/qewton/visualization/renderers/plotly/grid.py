"""Plotly artists for plots on structured grids."""

from plotly import graph_objects as go
import numpy as np

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    PlotlyArtist,
    _apply_scale,
    _mask_nan_color_as_gaps,
    _spatial_variable,
)


class ImageArtist(PlotlyArtist):
    """Draws an image plot as a raster image.

    Used for :class:`~qewton.visualization.plots.data.grid.ImagePlot`.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        image = plot.evaluate().values
        trace = go.Image(z=image)

        backend_figure.add_trace(trace, row=row, col=col)

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type="log" if plot.x.log_scale else "linear",
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type="log" if plot.y.log_scale else "linear",
            row=row,
            col=col,
        )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        image = plot.evaluate().values
        backend_figure.data[self.figure_idx].z = image


class HeatmapArtist(PlotlyArtist):
    """Draws a heatmap plot as a ``go.Heatmap``.

    Used for :class:`~qewton.visualization.plots.data.grid.HeatmapPlot`. The
    plot's color field, if present, replaces the values as heatmap data.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        data, color = result.values, result.color
        c = plot.color

        cmap = plot.theme.resolve_cmap(c.cmap if c is not None else None)
        if color is not None:
            data = color

        scale_kwargs = _apply_scale(
            c.scale if c is not None else None,
            "zmin",
            "zmax",
            backend_figure=backend_figure,
            row=row,
            col=col,
        )
        trace = go.Heatmap(z=data[..., 0], colorscale=cmap, **scale_kwargs)

        backend_figure.add_trace(trace, row=row, col=col)

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type="log" if plot.x.log_scale else "linear",
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type="log" if plot.y.log_scale else "linear",
            row=row,
            col=col,
        )

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        data, color = result.values, result.color
        if color is not None:
            data = color
        trace = backend_figure.data[self.figure_idx]
        trace.z = data[..., 0]
        if plot.color is not None and plot.color.scale is not None:
            trace.zmin, trace.zmax = plot.color.scale.range


class ParametricHeatmapArtist(PlotlyArtist):
    """Draws a heatmap at explicit 2D coordinates.

    Used for :class:`~qewton.visualization.plots.table.heatmap.TableHeatMap`.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        cmap = plot.theme.resolve_cmap(plot.color.cmap)
        scale_kwargs = _apply_scale(
            plot.color.scale, backend_figure=backend_figure, row=row, col=col
        )
        idx = len(backend_figure.data)
        backend_figure.add_trace(
            go.Heatmap(
                x=result.x,
                y=result.y,
                z=result.z[..., 0],
                colorscale=cmap,
                **scale_kwargs,
            ),
            row=row,
            col=col,
        )

        backend_figure.update_xaxes(
            title=plot.x.math_name,
            type="log" if plot.x.log_scale else "linear",
            row=row,
            col=col,
        )
        backend_figure.update_yaxes(
            title=plot.y.math_name,
            type="log" if plot.y.log_scale else "linear",
            row=row,
            col=col,
        )

        return cls(idx)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y, trace.z = result.x, result.y, result.z[..., 0]
        if plot.color.scale is not None:
            trace.zmin, trace.zmax = plot.color.scale.range


class SurfaceArtist(PlotlyArtist):
    """Draws a surface plot as a 3D height field.

    Used for :class:`~qewton.visualization.plots.data.grid.SurfacePlot`. The
    values are the height over an index grid; the color field, if present,
    sets the surface color.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        cmap = plot.theme.resolve_cmap(plot.color.cmap if plot.color is not None else None)

        result = plot.evaluate()
        data, color = result.values, result.color
        scale_kwargs = _apply_scale(
            plot.color.scale if plot.color is not None else None,
            backend_figure=backend_figure,
            row=row,
            col=col,
        )
        trace = go.Surface(
            z=data[..., 0], surfacecolor=color, colorscale=cmap, **scale_kwargs
        )
        backend_figure.add_trace(trace, row=row, col=col)

        # go.Surface lives in a `scene`, whose axes are set via
        # update_scenes, not update_xaxes/update_yaxes.
        scene_axes = dict(
            xaxis=dict(
                title=plot.x.math_name, type="log" if plot.x.log_scale else "linear"
            ),
            yaxis=dict(
                title=plot.y.math_name, type="log" if plot.y.log_scale else "linear"
            ),
        )
        if plot.z is not None:
            scene_axes["zaxis"] = dict(
                title=plot.z.math_name, type="log" if plot.z.log_scale else "linear"
            )
        backend_figure.update_scenes(row=row, col=col, **scene_axes)

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        data, color = result.values, result.color
        trace = backend_figure.data[self.figure_idx]
        trace.z = data[..., 0]
        trace.surfacecolor = color
        if plot.color is not None and plot.color.scale is not None:
            trace.cmin, trace.cmax = plot.color.scale.range


class ParametricSurfaceArtist(PlotlyArtist):
    """Draws a structured grid at explicit 3D coordinates.

    Used for :class:`~qewton.visualization.plots.data.grid.EmbeddedGridPlot`.
    The grid is a ``go.Surface`` with x/y/z from the geometry and the color
    field as surface color; points with NaN color are left out as holes.
    The scene axes span the full geometry, not only the drawn slice, so the
    view stays fixed while a control moves the slice.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        cmap = plot.theme.resolve_cmap(plot.color.cmap)
        scale_kwargs = _apply_scale(
            plot.color.scale, backend_figure=backend_figure, row=row, col=col
        )
        x, y, z = _mask_nan_color_as_gaps(result.x, result.y, result.z, result.color)

        idx = len(backend_figure.data)
        backend_figure.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=result.color,
                colorscale=cmap,
                **scale_kwargs,
            ),
            row=row,
            col=col,
        )

        backend_figure.update_scenes(
            aspectmode="data", row=row, col=col, **cls._scene_axis_kwargs(plot)
        )
        return cls(idx)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        x, y, z = _mask_nan_color_as_gaps(result.x, result.y, result.z, result.color)
        trace.x, trace.y, trace.z = x, y, z
        trace.surfacecolor = result.color
        if plot.color.scale is not None:
            trace.cmin, trace.cmax = plot.color.scale.range

    @staticmethod
    def _scene_axis_kwargs(plot) -> dict:
        """Scene axis ranges from the full geometry extent and titles from
        the geometry's coordinate Variable."""
        geometry = plot.drawn_geometry_axes.geometry
        points = geometry.discretization_points
        names = axis_names_from_variable(_spatial_variable(geometry), 3)
        return {
            axis: dict(
                title=names[i],
                range=[
                    float(np.nanmin(points[..., i])),
                    float(np.nanmax(points[..., i])),
                ],
            )
            for i, axis in enumerate(("xaxis", "yaxis", "zaxis"))
        }
