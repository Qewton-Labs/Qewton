from plotly import graph_objects as go
import numpy as np

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    _detach_to_numpy,
    PlotlyArtist,
    _apply_scale,
    _edge_trace,
    _to_numpy,
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
                _edge_trace(vertices, cells, color=plot.theme.line_color), row=row, col=col
            )

        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)
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

    def remove(self, backend_figure):
        pass

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
            z_name = z_spec.name
        else:
            x_name, y_name, z_name = axis_names_from_variable(geometry.variable, 3)
        return dict(
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name)
        )


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
    def create(cls, backend_figure, plot, row=None, col=None):
        result = plot.evaluate()
        vertices, cells = _to_numpy(result.vertices), _detach_to_numpy(result.cells)
        color = _to_numpy(result.color) if result.color is not None else None
        zeros = np.zeros(len(vertices))
        cmap = (plot.color.cmap if plot.color.cmap else None) or plot.theme.default_cmap

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
            ),
            row=row,
            col=col,
        )

        edges_idx = None
        if plot.show_edges:
            edges_idx = len(backend_figure.data)
            backend_figure.add_trace(
                _edge_trace(
                    np.column_stack([vertices[:, :2], zeros]), cells, color=plot.theme.line_color
                ),
                row=row,
                col=col,
            )

        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name = axis_names_from_variable(geometry.variable, 2)
        backend_figure.update_scenes(
            camera=dict(eye=dict(x=0, y=0, z=2.0), up=dict(x=0, y=1, z=0)),
            xaxis=dict(title=x_name),
            yaxis=dict(title=y_name),
            zaxis=dict(visible=False),
            aspectmode="data",
            dragmode="pan",
            row=row,
            col=col,
        )
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)

        return cls(mesh_idx, edges_idx)

    def update(self, backend_figure, plot):
        result = plot.evaluate()
        color = _to_numpy(result.color) if result.color is not None else None
        trace = backend_figure.data[self.figure_idx]
        trace.intensity = color
        if plot.color.scale is not None:
            trace.cmin, trace.cmax = plot.color.scale.range

    def remove(self, backend_figure):
        pass
