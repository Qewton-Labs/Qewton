from plotly import graph_objects as go
import numpy as np

from qewton.visualization.renderers.plotly.common import PlotlyArtist, _edge_trace, _mesh_edges


class GeometryArtist(PlotlyArtist):
    def __init__(self, mesh_idx, edges_idx=None):
        super().__init__(mesh_idx)
        self.edges_idx = edges_idx

    @classmethod
    def create(cls, backend_figure, plot):
        result = plot.evaluate()
        vertices, cells = result.vertices, result.cells
        color = getattr(plot.theme, "geometry_color", "lightgray")

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
            )
        )

        edges_idx = None
        if plot.show_edges:
            edges_idx = len(backend_figure.data)
            backend_figure.add_trace(_edge_trace(vertices, cells))

        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        return cls(mesh_idx, edges_idx)

    def update(self, backend_figure, plot):
        pass  # Geometry is static for now, later maybe parametric geometries

    def remove(self, backend_figure):
        pass


class GeometryArtist2D(PlotlyArtist):
    def __init__(self, fill_idx, boundary_idx=None, interior_idx=None):
        super().__init__(fill_idx)
        self.boundary_idx = boundary_idx
        self.interior_idx = interior_idx

    @staticmethod
    def _triangle_fill_trace(
        vertices: np.ndarray, cells: np.ndarray, color: str
    ) -> go.Scatter:
        """Fills a 2D triangulation as one trace. Each triangle is a None-separated
        segment, which Plotly fills independently - so holes and disconnected
        components need no special handling."""
        xs, ys = [], []
        for tri in cells:
            pts = vertices[tri]
            xs.extend([pts[0, 0], pts[1, 0], pts[2, 0], pts[0, 0], None])
            ys.extend([pts[0, 1], pts[1, 1], pts[2, 1], pts[0, 1], None])
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(width=0),  # no interior edges - boundary drawn separately
            hoverinfo="skip",
            showlegend=False,
        )

    @staticmethod
    def _edge_trace_2d(
        vertices: np.ndarray,
        edges: np.ndarray,
        color: str = "black",
        width: float = 1.5,
    ) -> go.Scatter:
        """Draws unordered 2D edges. Works for boundary_faces directly - line
        segments need no traversal order, unlike filled polygons."""
        xs, ys = [], []
        for a, b in edges:
            xs.extend([vertices[a, 0], vertices[b, 0], None])
            ys.extend([vertices[a, 1], vertices[b, 1], None])
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        )

    @classmethod
    def create(cls, backend_figure, plot):
        mesh = plot.interior_mesh or plot.boundary_mesh
        color = getattr(plot.theme, "geometry_color", "lightgray")

        fill_idx = len(backend_figure.data)
        backend_figure.add_trace(
            cls._triangle_fill_trace(mesh.vertices, mesh.cells, color)
        )

        interior_idx = None
        if plot.show_edges:
            interior_idx = len(backend_figure.data)
            backend_figure.add_trace(
                cls._edge_trace_2d(
                    mesh.vertices,
                    _mesh_edges(mesh.cells),
                    color="gray",
                    width=0.5,
                )
            )

        # boundary on top - unordered edges, holes included, no ordering needed
        boundary_idx = len(backend_figure.data)
        backend_figure.add_trace(
            cls._edge_trace_2d(plot.boundary_mesh.vertices, plot.boundary_mesh.cells)
        )

        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)
        backend_figure.update_yaxes(scaleanchor="x")

        return cls(fill_idx, boundary_idx, interior_idx)

    def update(self, backend_figure, plot):
        pass

    def remove(self, backend_figure):
        pass
