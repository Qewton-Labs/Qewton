from plotly import graph_objects as go
import numpy as np

from qewton.visualization.renderers.plotly.common import PlotlyArtist, _apply_scale, _edge_trace


class SurfaceMeshArtist(PlotlyArtist):
    """3D triangulated mesh with optional per-vertex coloring.

    Serves both MeshFieldPlot (3D surfaces) and MeshSurfacePlot (2D mesh
    elevated into 3D) - both return a MeshResult with 3D vertices.
    """

    def __init__(self, mesh_idx, edges_idx=None):
        super().__init__(mesh_idx)
        self.edges_idx = edges_idx

    @classmethod
    def create(cls, backend_figure, plot):
        result = plot.evaluate()
        vertices, cells, color = result.vertices, result.cells, result.color

        spec = getattr(plot, "color", None)
        cmap = (spec.cmap if spec is not None and spec.cmap else None) or getattr(
            plot.theme, "default_cmap", "viridis"
        )

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
        )
        if color is not None:
            kwargs.update(
                intensity=color,
                intensitymode="vertex" if len(color) == len(vertices) else "cell",
                colorscale=cmap,
                **_apply_scale(spec.scale if spec is not None else None),
            )
        else:
            kwargs["color"] = getattr(plot.theme, "geometry_color", "lightgray")

        mesh_idx = len(backend_figure.data)
        backend_figure.add_trace(go.Mesh3d(**kwargs))

        edges_idx = None
        if plot.show_edges:
            edges_idx = len(backend_figure.data)
            backend_figure.add_trace(_edge_trace(vertices, cells))

        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        return cls(mesh_idx, edges_idx)

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        vertices, color = result.vertices, result.color
        trace = backend_figure.data[self.figure_idx]
        # z may change when a control moves (MeshSurfacePlot), x/y never do
        trace.z = vertices[:, 2]
        if color is not None:
            trace.intensity = color
            if plot.color is not None and plot.color.scale is not None:
                trace.cmin, trace.cmax = plot.color.scale.range
        if self.edges_idx is not None:
            edge_trace = backend_figure.data[self.edges_idx]
            edge_trace.z = _edge_trace(vertices, plot.render_cells()).z

    def remove(self, backend_figure):
        pass


class FilledMeshArtist(PlotlyArtist):
    """Scalar field on a 2D triangulation.

    Plotly has no 2D counterpart to Mesh3d, so the mesh is drawn as a flat
    Mesh3d at z=0 viewed from directly above. This gives real Gouraud shading
    across triangles, unlike fill='toself', which allows only one color per trace.
    """

    def __init__(self, mesh_idx, edges_idx=None):
        super().__init__(mesh_idx)
        self.edges_idx = edges_idx

    @classmethod
    def create(cls, backend_figure, plot):
        result = plot.evaluate()
        vertices, cells, color = result.vertices, result.cells, result.color
        zeros = np.zeros(len(vertices))
        cmap = (plot.color.cmap if plot.color.cmap else None) or getattr(
            plot.theme, "default_cmap", "viridis"
        )

        mesh_idx = len(backend_figure.data)
        backend_figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=zeros,
                i=cells[:, 0],
                j=cells[:, 1],
                k=cells[:, 2],
                intensity=color,
                intensitymode="vertex",
                colorscale=cmap,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                **_apply_scale(plot.color.scale),
            )
        )

        edges_idx = None
        if plot.show_edges:
            edges_idx = len(backend_figure.data)
            backend_figure.add_trace(
                _edge_trace(np.column_stack([vertices[:, :2], zeros]), cells)
            )

        backend_figure.update_layout(
            scene=dict(
                camera=dict(eye=dict(x=0, y=0, z=2.0), up=dict(x=0, y=1, z=0)),
                zaxis=dict(visible=False),
                aspectmode="data",
                dragmode="pan",
            )
        )
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        return cls(mesh_idx, edges_idx)

    def update(self, backend_figure, plot):
        color = plot.evaluate().color
        trace = backend_figure.data[self.figure_idx]
        trace.intensity = color
        if plot.color.scale is not None:
            trace.cmin, trace.cmax = plot.color.scale.range

    def remove(self, backend_figure):
        pass
