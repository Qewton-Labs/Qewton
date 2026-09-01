from plotly import graph_objects as go
from plotly.colors import sample_colorscale
import numpy as np

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    _detach_to_numpy,
    PlotlyArtist,
    _apply_scale,
    _edge_trace,
    _edge_trace_2d,
    _mesh_edges,
    _to_numpy,
    _triangle_fill_trace,
)


class SurfaceMeshArtist(PlotlyArtist):
    """3D triangulated mesh with optional per-vertex coloring.

    Serves both MeshFieldPlot (3D surfaces) and MeshSurfacePlot (2D mesh
    elevated into 3D) - both return a MeshResult with 3D vertices.
    """

    def __init__(self, mesh_idx, edges_idx=None):
        super().__init__(mesh_idx)
        self.edges_idx = edges_idx

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        vertices, cells = _to_numpy(result.vertices), _detach_to_numpy(result.cells)
        color = _to_numpy(result.color) if result.color is not None else None

        spec = getattr(plot, "color", None)
        cmap = (spec.cmap if spec is not None and spec.cmap else None) or plot.theme.default_cmap

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
                **_apply_scale(spec.scale if spec is not None else None),
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

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        vertices = _to_numpy(result.vertices)
        color = _to_numpy(result.color) if result.color is not None else None
        trace = backend_figure.data[self.figure_idx]
        # z may change when a control moves (MeshSurfacePlot), x/y never do
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
        """x/y always come from the mesh's own ambient coordinate Variable
        (MeshFieldPlot and MeshSurfacePlot alike - the mesh itself never
        moves). z differs: MeshFieldPlot's z genuinely is the mesh's own
        3rd coordinate (part of the same Variable), but MeshSurfacePlot's z
        is a *data* value elevating a 2D mesh - it has its own named AxisSpec
        (`plot.z`) instead, unrelated to the mesh's 2D coordinate Variable."""
        geometry = plot.data_config.geometry_axes.geometry
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
    """Scalar field on a 2D triangulation, drawn as flat-shaded fills binned
    by value.

    Plotly has no 2D primitive with per-triangle Gouraud shading, and the
    alternative (go.Mesh3d at z=0) draws into a `scene` cell, which can
    never sit beside a genuinely cartesian cell (HeatmapPlot) in one row -
    exactly the comparison layout this exists for. Instead: one
    go.Scatter(fill="toself") trace per value bin, each batching every
    triangle whose value falls in it (one fillcolor per trace - the cost is
    flat, per-triangle shading instead of per-vertex interpolation).

    Always creates exactly `plot.n_bins` fill traces, some possibly empty,
    so a redraw whose bin occupancy shifted doesn't change the trace count.
    A separate invisible trace carries the continuous colorbar, since no
    single visible trace holds the full value range for Plotly to infer one
    from.
    """

    def __init__(self, fill_indices, colorbar_idx, edges_idx=None):
        super().__init__(fill_indices[0] if fill_indices else colorbar_idx)
        self.fill_indices = fill_indices
        self.colorbar_idx = colorbar_idx
        self.edges_idx = edges_idx

    @staticmethod
    def _bin(triangle_values, n_bins, scale):
        """(bin_idx, bin_centers, vmin, vmax) - vmin/vmax come from the
        shared Scale's trained range when one is set, else the triangle
        values themselves, matching _apply_scale's own convention."""
        if scale is not None and scale.range is not None:
            vmin, vmax = scale.range
        else:
            vmin = float(np.nanmin(triangle_values))
            vmax = float(np.nanmax(triangle_values))
        if vmin == vmax:
            vmax = vmin + 1e-9  # degenerate (uniform field) - avoid a zero-width range
        edges = np.linspace(vmin, vmax, n_bins + 1)
        bin_idx = np.clip(np.digitize(triangle_values, edges[1:-1]), 0, n_bins - 1)
        centers = (edges[:-1] + edges[1:]) / 2
        return bin_idx, centers, vmin, vmax

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        vertices = _to_numpy(result.vertices)[:, :2]
        cells = _detach_to_numpy(result.cells)
        color = _to_numpy(result.color)
        cmap = plot.color.cmap or plot.theme.default_cmap

        triangle_values = color[cells].mean(axis=1)
        bin_idx, centers, vmin, vmax = cls._bin(triangle_values, plot.n_bins, plot.color.scale)
        bin_colors = sample_colorscale(cmap, list(np.clip((centers - vmin) / (vmax - vmin), 0, 1)))

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
            cls._colorbar_carrier_trace(cmap, vmin, vmax, plot.color.scale),
            row=row,
            col=col,
        )

        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name = axis_names_from_variable(geometry.variable, 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        # Same known gap as GeometryArtist2D: exact only for the
        # non-faceted case (one cell's xaxis is literally named "x").
        backend_figure.update_yaxes(scaleanchor="x", row=row, col=col)

        return cls(fill_indices, colorbar_idx, edges_idx)

    @staticmethod
    def _colorbar_carrier_trace(cmap, vmin, vmax, scale):
        """An invisible single-point trace whose only purpose is showing a
        continuous colorbar for this artist's discretely-binned fills -
        _apply_scale's cmin/cmax (when a shared Scale set them) are
        overridden by the actual bin range, since a single dummy point
        gives Plotly nothing to infer a range from on its own."""
        marker = {
            "colorscale": cmap, "color": [vmin],
            **_apply_scale(scale),
            "cmin": vmin, "cmax": vmax,
        }
        return go.Scatter(
            x=[None], y=[None], mode="markers", marker=marker,
            showlegend=False, hoverinfo="skip",
        )

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        vertices = _to_numpy(result.vertices)[:, :2]
        cells = _detach_to_numpy(result.cells)
        color = _to_numpy(result.color)
        cmap = plot.color.cmap or plot.theme.default_cmap

        triangle_values = color[cells].mean(axis=1)
        bin_idx, centers, vmin, vmax = self._bin(triangle_values, plot.n_bins, plot.color.scale)
        bin_colors = sample_colorscale(cmap, list(np.clip((centers - vmin) / (vmax - vmin), 0, 1)))

        for i, trace_idx in enumerate(self.fill_indices):
            new = _triangle_fill_trace(vertices, cells[bin_idx == i], bin_colors[i])
            trace = backend_figure.data[trace_idx]
            trace.x, trace.y, trace.fillcolor = new.x, new.y, new.fillcolor

        if self.edges_idx is not None:
            new_edges = _edge_trace_2d(vertices, _mesh_edges(cells))
            edge_trace = backend_figure.data[self.edges_idx]
            edge_trace.x, edge_trace.y = new_edges.x, new_edges.y

        colorbar_trace = backend_figure.data[self.colorbar_idx]
        colorbar_trace.marker.cmin, colorbar_trace.marker.cmax = vmin, vmax
