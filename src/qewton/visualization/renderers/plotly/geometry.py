from plotly import graph_objects as go

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    _detach_to_numpy,
    PlotlyArtist,
    _edge_trace,
    _edge_trace_2d,
    _mesh_edges,
    _to_numpy,
    _triangle_fill_trace,
)


class GeometryArtist(PlotlyArtist):
    """Draws a 3D GeometryPlot as an uncolored surface mesh."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        vertices, cells = _to_numpy(result.vertices), _detach_to_numpy(result.cells)
        color = plot.theme.geometry_color

        mesh_idx = len(backend_figure.data)
        backend_figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=cells[:, 0],
                j=cells[:, 1],
                k=cells[:, 2],
                color=color,
                flatshading=True,
                opacity=plot.theme.surface_opacity,
            ),
            row=row,
            col=col,
        )

        if plot.show_edges:
            backend_figure.add_trace(
                _edge_trace(
                    vertices, cells,
                    color=plot.theme.line_color, opacity=plot.theme.wireframe_opacity,
                ),
                row=row, col=col,
            )

        x_name, y_name, z_name = axis_names_from_variable(plot.geometry.variable, 3)
        backend_figure.update_scenes(
            row=row, col=col,
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name),
        )

        return cls(mesh_idx)

    def update(self, backend_figure, plot):
        pass  # Geometry is static for now, later maybe parametric geometries


class GeometryArtist2D(PlotlyArtist):
    """Draws a 2D GeometryPlot as a filled triangulation with its boundary
    (and, optionally, interior edges) outlined on top."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        mesh = plot.interior_mesh or plot.boundary_mesh
        vertices, cells = _to_numpy(mesh.vertices), _detach_to_numpy(mesh.cells)
        boundary_vertices = _to_numpy(plot.boundary_mesh.vertices)
        boundary_cells = _detach_to_numpy(plot.boundary_mesh.cells)
        color = plot.theme.geometry_color

        fill_idx = len(backend_figure.data)
        backend_figure.add_trace(
            _triangle_fill_trace(vertices, cells, color, opacity=plot.theme.surface_opacity),
            row=row, col=col,
        )

        if plot.show_edges:
            backend_figure.add_trace(
                _edge_trace_2d(
                    vertices,
                    _mesh_edges(cells),
                    color=plot.theme.line_color,
                    width=0.5,
                    opacity=plot.theme.wireframe_opacity,
                ),
                row=row,
                col=col,
            )

        # boundary on top - unordered edges, holes included, no ordering
        # needed. Always fully opaque, unlike the interior wireframe above:
        # this is the domain's actual boundary, not a stylistic overlay.
        backend_figure.add_trace(
            _edge_trace_2d(
                boundary_vertices, boundary_cells, color=plot.theme.line_color
            ),
            row=row,
            col=col,
        )

        x_name, y_name = axis_names_from_variable(plot.geometry.variable, 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        # NOTE: scaleanchor="x" is exact for the non-faceted case (the only
        # xaxis is literally named "x"); in a faceted grid each cell has its
        # own xaxis (x2, x3, ...) so this anchors to the wrong one for cells
        # past the first. Cosmetic only (aspect-ratio lock) and not the
        # facet-motivating use case (EmbeddedGridPlot, 3D) - left as a known
        # gap rather than resolved per-cell axis naming here.
        backend_figure.update_yaxes(scaleanchor="x", row=row, col=col)

        return cls(fill_idx)

    def update(self, backend_figure, plot):
        pass
