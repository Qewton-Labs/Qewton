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
    """Draws an ImagePlot as a raster image."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
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

    def update(self, backend_figure, plot):
        image = plot.evaluate().values
        backend_figure.data[self.figure_idx].z = image


class HeatmapArtist(PlotlyArtist):
    """Draws a HeatmapPlot as a flat, colored grid."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        data, color = result.values, result.color
        c = plot.color

        cmap = (
            c.cmap
            if c is not None and c.cmap is not None
            else plot.theme.default_cmap
        )
        if color is not None:
            data = color

        scale_kwargs = _apply_scale(c.scale if c is not None else None, "zmin", "zmax")
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

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        data, color = result.values, result.color
        if color is not None:
            data = color
        trace = backend_figure.data[self.figure_idx]
        trace.z = data[..., 0]
        if plot.color is not None and plot.color.scale is not None:
            trace.zmin, trace.zmax = plot.color.scale.range


class SurfaceArtist(PlotlyArtist):
    """Draws a SurfacePlot as a 3D height field over an index grid."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        cmap = plot.theme.default_cmap
        if plot.color is not None:
            if plot.color.cmap is not None:
                cmap = plot.color.cmap

        result = plot.evaluate()
        data, color = result.values, result.color
        scale_kwargs = _apply_scale(plot.color.scale if plot.color is not None else None)
        trace = go.Surface(
            z=data[..., 0], surfacecolor=color, colorscale=cmap, **scale_kwargs
        )
        backend_figure.add_trace(trace, row=row, col=col)

        # SurfaceArtist draws into a `scene` subplot (go.Surface), which has
        # no top-level xaxis/yaxis of its own - update_xaxes/update_yaxes
        # would silently target the wrong (nonexistent, for this cell)
        # cartesian axes, so x/y/z titles all go through update_scenes
        # together, not split across two different calls the way a 2D
        # HeatmapArtist can get away with.
        scene_axes = dict(
            xaxis=dict(title=plot.x.math_name, type="log" if plot.x.log_scale else "linear"),
            yaxis=dict(title=plot.y.math_name, type="log" if plot.y.log_scale else "linear"),
        )
        if plot.z is not None:
            scene_axes["zaxis"] = dict(
                title=plot.z.math_name, type="log" if plot.z.log_scale else "linear"
            )
        backend_figure.update_scenes(row=row, col=col, **scene_axes)

        return cls(len(backend_figure.data) - 1)

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        data, color = result.values, result.color
        trace = backend_figure.data[self.figure_idx]
        trace.z = data[..., 0]
        trace.surfacecolor = color
        if plot.color is not None and plot.color.scale is not None:
            trace.cmin, trace.cmax = plot.color.scale.range


class ParametricSurfaceArtist(PlotlyArtist):
    """Structured grid drawn at explicit 3D coordinates.

    go.Surface takes x/y/z as full 2D arrays, not just a height field over an
    index grid - an arbitrary embedded parametric surface. surfacecolor
    decouples the field from the geometry, unlike SurfaceArtist where z
    carries both.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        cmap = plot.color.cmap or plot.theme.default_cmap
        scale_kwargs = _apply_scale(plot.color.scale)
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

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        trace = backend_figure.data[self.figure_idx]
        x, y, z = _mask_nan_color_as_gaps(result.x, result.y, result.z, result.color)
        trace.x, trace.y, trace.z = x, y, z
        trace.surfacecolor = result.color
        if plot.color.scale is not None:
            trace.cmin, trace.cmax = plot.color.scale.range

    @staticmethod
    def _scene_axis_kwargs(plot) -> dict:
        """Explicit scene axis ranges from the FULL (unreduced) geometry
        extent, not just the currently drawn slice, plus a title per axis
        derived from the geometry's own coordinate Variable (EmbeddedGridPlot
        has no x=/y=/z= AxisSpec of its own to name them from - positions
        come straight from the geometry, see axis_names_from_variable()).

        Without the range part, Plotly auto-scales the scene to fit whichever
        slice is currently on screen - so as a slider moves the slice, the
        camera silently reframes around it every redraw and the view looks
        static even though the actual position moved.
        """
        geometry = plot.data_config.geometry_axes.geometry
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
