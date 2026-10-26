"""Plotly artists for fields on meshes."""

from plotly import graph_objects as go
import numpy as np

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    _detach_to_numpy,
    PlotlyArtist,
    _apply_scale,
    _bin_colors,
    _bin_values,
    _colorbar_carrier_trace,
    _edge_trace,
    _edge_trace_2d,
    _mesh_edges,
    _to_numpy,
    _triangle_fill_trace,
)


class SurfaceMeshArtist(PlotlyArtist):
    """Draws a 3D triangle mesh as ``go.Mesh3d``.

    Used for a 3D :class:`~qewton.visualization.plots.data.mesh.MeshFieldPlot`
    and for a :class:`~qewton.visualization.plots.data.mesh.MeshSurfacePlot`
    (2D mesh elevated by a data value). The color field is per vertex or
    per cell; without one, the mesh is drawn in ``theme.geometry_color``.
    With ``plot.show_edges`` a wireframe trace is added.

    Args:
        mesh_idx (int): Index of the mesh trace.
        edges_idx (int, optional): Index of the wireframe trace, if any.

    Attributes:
        edges_idx (int or None): Index of the wireframe trace, if any.
        trace_indices (list of int): The mesh trace and, if present, the
            wireframe trace.
    """

    def __init__(self, mesh_idx, edges_idx=None):
        super().__init__(mesh_idx)
        self.edges_idx = edges_idx

    @property
    def trace_indices(self) -> list[int]:
        return [self.figure_idx] + ([self.edges_idx] if self.edges_idx is not None else [])

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        vertices, cells = _to_numpy(result.vertices), _detach_to_numpy(result.cells)
        color = _to_numpy(result.color) if result.color is not None else None

        spec = getattr(plot, "color", None)
        cmap = plot.theme.resolve_cmap(spec.cmap if spec is not None else None)

        kwargs = dict(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=cells[:, 0],
            j=cells[:, 1],
            k=cells[:, 2],
            flatshading=False,
            lighting=dict(
                ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
            ),
            opacity=plot.theme.surface_opacity,
        )
        if color is not None:
            kwargs.update(
                intensity=color,
                intensitymode="vertex" if len(color) == len(vertices) else "cell",
                colorscale=cmap,
                **_apply_scale(
                    spec.scale if spec is not None else None,
                    backend_figure=backend_figure, row=row, col=col,
                ),
            )
        else:
            kwargs["color"] = plot.theme.geometry_color

        mesh_idx = len(backend_figure.data)
        backend_figure.add_trace(go.Mesh3d(**kwargs), row=row, col=col)

        edges_idx = None
        if plot.show_edges:
            edges_idx = len(backend_figure.data)
            backend_figure.add_trace(
                _edge_trace(
                    vertices, cells,
                    color=plot.theme.line_color, opacity=plot.theme.wireframe_opacity,
                ),
                row=row, col=col,
            )

        backend_figure.update_scenes(row=row, col=col, **cls._scene_axis_titles(plot))

        return cls(mesh_idx, edges_idx)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        vertices = _to_numpy(result.vertices)
        color = _to_numpy(result.color) if result.color is not None else None
        trace = backend_figure.data[self.figure_idx]
        # Only z can change (MeshSurfacePlot); x/y are fixed.
        trace.z = vertices[:, 2]
        if color is not None:
            trace.intensity = color
            if plot.color is not None and plot.color.scale is not None:
                trace.cmin, trace.cmax = plot.color.scale.range
        if self.edges_idx is not None:
            edge_trace = backend_figure.data[self.edges_idx]
            edge_trace.z = _edge_trace(vertices, _detach_to_numpy(plot.render_cells())).z

    @staticmethod
    def _scene_axis_titles(plot) -> dict:
        """Scene axis titles: x/y from the mesh's coordinate Variable; z
        from `plot.z` if the plot has one (MeshSurfacePlot), else the
        Variable's third component."""
        geometry = plot.drawn_geometry_axes.geometry
        z_spec = getattr(plot, "z", None)
        if z_spec is not None:
            x_name, y_name = axis_names_from_variable(geometry.variable, 2)
            z_name = z_spec.math_name
        else:
            x_name, y_name, z_name = axis_names_from_variable(geometry.variable, 3)
        return dict(
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name)
        )


class FilledMeshArtist(PlotlyArtist):
    """Draws a scalar field on a 2D triangulation as binned flat fills.

    Used for a 2D :class:`~qewton.visualization.plots.data.mesh.MeshFieldPlot`.
    Each triangle is colored by the mean of its vertex values. Triangles
    are grouped into ``plot.n_bins`` value bins, each drawn as one
    ``go.Scatter(fill="toself")`` trace (some possibly empty), plus an
    invisible trace carrying the continuous colorbar and, with
    ``plot.show_edges``, a wireframe trace. The y-axis is scale-anchored to
    x.

    Args:
        fill_indices (list of int): Indices of the fill traces, one per bin.
        colorbar_idx (int): Index of the colorbar carrier trace.
        edges_idx (int, optional): Index of the wireframe trace, if any.

    Attributes:
        fill_indices (list of int): Indices of the fill traces, one per bin.
        colorbar_idx (int): Index of the colorbar carrier trace.
        edges_idx (int or None): Index of the wireframe trace, if any.
        trace_indices (list of int): The fill traces and the colorbar carrier;
            the wireframe is not included because a 2D mesh's vertices never
            move.
    """

    def __init__(self, fill_indices, colorbar_idx, edges_idx=None):
        super().__init__(fill_indices[0] if fill_indices else colorbar_idx)
        self.fill_indices = fill_indices
        self.colorbar_idx = colorbar_idx
        self.edges_idx = edges_idx

    @property
    def trace_indices(self) -> list[int]:
        return self.fill_indices + [self.colorbar_idx]

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        vertices = _to_numpy(result.vertices)[:, :2]
        cells = _detach_to_numpy(result.cells)
        color = _to_numpy(result.color)
        cmap = plot.theme.resolve_cmap(plot.color.cmap)

        triangle_values = color[cells].mean(axis=1)
        bin_idx, centers, vmin, vmax = _bin_values(triangle_values, plot.n_bins, plot.color.scale)
        bin_colors = _bin_colors(cmap, centers, vmin, vmax)

        fill_indices = []
        for i in range(plot.n_bins):
            fill_indices.append(len(backend_figure.data))
            backend_figure.add_trace(
                _triangle_fill_trace(
                    vertices, cells[bin_idx == i], bin_colors[i],
                    opacity=plot.theme.surface_opacity,
                ),
                row=row,
                col=col,
            )

        edges_idx = None
        if plot.show_edges:
            edges_idx = len(backend_figure.data)
            backend_figure.add_trace(
                _edge_trace_2d(
                    vertices, _mesh_edges(cells),
                    color=plot.theme.line_color, width=0.5,
                    opacity=plot.theme.wireframe_opacity,
                ),
                row=row,
                col=col,
            )

        colorbar_idx = len(backend_figure.data)
        backend_figure.add_trace(
            _colorbar_carrier_trace(cmap, vmin, vmax, plot.color.scale, backend_figure, row, col),
            row=row,
            col=col,
        )

        geometry = plot.drawn_geometry_axes.geometry
        x_name, y_name = axis_names_from_variable(geometry.variable, 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        # scaleanchor="x" is exact only for a non-faceted figure, where the
        # cell's x-axis is named "x".
        backend_figure.update_yaxes(scaleanchor="x", row=row, col=col)

        return cls(fill_indices, colorbar_idx, edges_idx)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        vertices = _to_numpy(result.vertices)[:, :2]
        cells = _detach_to_numpy(result.cells)
        color = _to_numpy(result.color)
        cmap = plot.theme.resolve_cmap(plot.color.cmap)

        triangle_values = color[cells].mean(axis=1)
        bin_idx, centers, vmin, vmax = _bin_values(triangle_values, plot.n_bins, plot.color.scale)
        bin_colors = _bin_colors(cmap, centers, vmin, vmax)

        for i, trace_idx in enumerate(self.fill_indices):
            new = _triangle_fill_trace(vertices, cells[bin_idx == i], bin_colors[i])
            trace = backend_figure.data[trace_idx]
            trace.x, trace.y, trace.fillcolor = new.x, new.y, new.fillcolor
            trace.line.color = new.line.color

        if self.edges_idx is not None:
            new_edges = _edge_trace_2d(vertices, _mesh_edges(cells))
            edge_trace = backend_figure.data[self.edges_idx]
            edge_trace.x, edge_trace.y = new_edges.x, new_edges.y

        colorbar_trace = backend_figure.data[self.colorbar_idx]
        colorbar_trace.marker.cmin, colorbar_trace.marker.cmax = vmin, vmax


#: Width of a colored field curve, relative to the theme's line width.
_FIELD_CURVE_WIDTH_FACTOR = 3


def _segment_coordinates(vertices: np.ndarray, segments: np.ndarray) -> list[list]:
    """Per coordinate axis, the segment end points separated by None, so
    one line trace draws every segment independently."""
    columns = [[] for _ in range(vertices.shape[1])]
    for a, b in segments:
        for column, coordinate in zip(columns, zip(vertices[a], vertices[b])):
            column.extend([*coordinate, None])
    return columns


class CurveFieldArtist(PlotlyArtist):
    """Draws a scalar field on line segments as colored lines.

    Used for a :class:`~qewton.visualization.plots.data.mesh.MeshFieldPlot` on
    a curve (e.g. the boundary of a 2D domain). In 3D, one
    ``go.Scatter3d`` trace colors its line per vertex. In 2D, segments are
    colored by the mean of their end values and binned into
    ``plot.n_bins`` traces, plus an invisible colorbar carrier trace, as in
    :class:`FilledMeshArtist`.

    Args:
        line_indices (list of int): Indices of the line traces (one in 3D, one
            per bin in 2D).
        colorbar_idx (int, optional): Index of the colorbar carrier trace (2D
            only).

    Attributes:
        line_indices (list of int): Indices of the line traces.
        colorbar_idx (int or None): Index of the colorbar carrier trace.
        trace_indices (list of int): The line traces and, if present, the
            colorbar carrier.
    """

    def __init__(self, line_indices, colorbar_idx=None):
        super().__init__(line_indices[0])
        self.line_indices = line_indices
        self.colorbar_idx = colorbar_idx

    @property
    def trace_indices(self) -> list[int]:
        colorbar = [self.colorbar_idx] if self.colorbar_idx is not None else []
        return self.line_indices + colorbar

    @staticmethod
    def _evaluate(plot):
        result = plot.evaluate()
        return (
            _to_numpy(result.vertices),
            _detach_to_numpy(result.cells),
            _to_numpy(result.color),
        )

    @staticmethod
    def _binned_traces(plot, vertices, cells, color):
        """(traces, vmin, vmax): one 2D line trace per value bin."""
        cmap = plot.theme.resolve_cmap(plot.color.cmap)
        bin_idx, centers, vmin, vmax = _bin_values(
            color[cells].mean(axis=1), plot.n_bins, plot.color.scale
        )
        bin_colors = _bin_colors(cmap, centers, vmin, vmax)
        traces = []
        for i in range(plot.n_bins):
            xs, ys = _segment_coordinates(vertices[:, :2], cells[bin_idx == i])
            traces.append(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(
                        color=bin_colors[i],
                        width=plot.theme.line_width * _FIELD_CURVE_WIDTH_FACTOR,
                    ),
                    hoverinfo="skip", showlegend=False,
                )
            )
        return traces, vmin, vmax

    @staticmethod
    def _segment_colors(cells, color) -> list:
        """Per-point colors matching _segment_coordinates(); the None gap
        after each segment repeats its end value."""
        return [value for a, b in cells for value in (color[a], color[b], color[b])]

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        vertices, cells, color = cls._evaluate(plot)
        cmap = plot.theme.resolve_cmap(plot.color.cmap)

        if plot.dim == 3:
            xs, ys, zs = _segment_coordinates(vertices, cells)
            idx = len(backend_figure.data)
            backend_figure.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(
                        color=cls._segment_colors(cells, color),
                        colorscale=cmap,
                        width=plot.theme.line_width * _FIELD_CURVE_WIDTH_FACTOR,
                        **_apply_scale(
                            plot.color.scale,
                            backend_figure=backend_figure, row=row, col=col,
                        ),
                    ),
                    hoverinfo="skip", showlegend=False,
                ),
                row=row, col=col,
            )
            backend_figure.update_scenes(
                row=row, col=col, **SurfaceMeshArtist._scene_axis_titles(plot)
            )
            return cls([idx])

        traces, vmin, vmax = cls._binned_traces(plot, vertices, cells, color)
        line_indices = []
        for trace in traces:
            line_indices.append(len(backend_figure.data))
            backend_figure.add_trace(trace, row=row, col=col)
        colorbar_idx = len(backend_figure.data)
        backend_figure.add_trace(
            _colorbar_carrier_trace(
                cmap, vmin, vmax, plot.color.scale, backend_figure, row, col
            ),
            row=row, col=col,
        )

        geometry = plot.drawn_geometry_axes.geometry
        x_name, y_name = axis_names_from_variable(geometry.variable, 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        backend_figure.update_yaxes(scaleanchor="x", row=row, col=col)
        return cls(line_indices, colorbar_idx)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        vertices, cells, color = self._evaluate(plot)
        if plot.dim == 3:
            backend_figure.data[self.figure_idx].line.color = self._segment_colors(
                cells, color
            )
            return
        traces, vmin, vmax = self._binned_traces(plot, vertices, cells, color)
        for trace_idx, new in zip(self.line_indices, traces):
            trace = backend_figure.data[trace_idx]
            trace.x, trace.y, trace.line.color = new.x, new.y, new.line.color
        colorbar_trace = backend_figure.data[self.colorbar_idx]
        colorbar_trace.marker.cmin, colorbar_trace.marker.cmax = vmin, vmax
