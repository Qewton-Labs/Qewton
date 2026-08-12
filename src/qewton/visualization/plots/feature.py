import numpy as np

from qewton.config.axes import Axes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.base import GridPlot3d, MeshPlot
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec, VectorSpec


class ImagePlot(GridPlot3d):
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, controls=controls)

    def create_artist(self, backend_figure, renderer):
        return renderer.ImageArtist.create(backend_figure, self)


class SurfacePlot(GridPlot3d):
    def create_artist(self, backend_figure, renderer):
        return renderer.SurfaceArtist.create(backend_figure, self)


class HeatmapPlot(GridPlot3d):
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, x, y, color=color, controls=controls)

    def create_artist(self, backend_figure, renderer):
        return renderer.HeatmapArtist.create(backend_figure, self)


class MeshFieldPlot(MeshPlot):
    """Scalar field colored on a mesh - the unstructured counterpart to HeatmapPlot.

    Works in 2D (colored triangulation) and 3D (colored surface of a body).
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        color: ColorSpec | Variable,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = True,
        **kwargs,
    ):
        super().__init__(
            data, data_config, controls=controls, show_edges=show_edges, **kwargs
        )
        self.color = color if isinstance(color, ColorSpec) else ColorSpec(color)
        self.require_scalar(self.color, "color")

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        color = self.scalar_at_vertices(self.color, data, slice_map)
        vertices = self.coord_transform.apply(self.mesh.vertices)
        return vertices, self.render_cells(), color

    def create_artist(self, backend_figure, renderer):
        return (
            renderer.MeshField2DArtist.create(backend_figure, self)
            if self.dim == 2
            else renderer.MeshArtist.create(backend_figure, self)
        )


class MeshSurfacePlot(MeshPlot):
    """Elevates a 2D triangulation into 3D using a data variable as height.

    The unstructured counterpart to SurfacePlot: x and y come from the mesh,
    z from the data, and color is optional and independent of z.
    """

    supported_dims = (2,)

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        z: AxisSpec | Variable,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = True,
        **kwargs,
    ):
        super().__init__(
            data, data_config, controls=controls, show_edges=show_edges, **kwargs
        )
        self.z = z if isinstance(z, AxisSpec) else AxisSpec(z)
        self.require_scalar(self.z, "z")
        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color is not None
            else ColorSpec(self.z.variable_or_axes)
        )
        if self.color is not None:
            self.require_scalar(self.color, "color")

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        z = self.scalar_at_vertices(self.z, data, slice_map)
        vertices = self.coord_transform.apply(np.column_stack([self.mesh.vertices, z]))
        color = (
            z
            if self.color.variable_or_axes is self.z.variable_or_axes
            else self.scalar_at_vertices(self.color, data, slice_map)
        )
        return vertices, self.mesh.cells, color

    def create_artist(self, backend_figure, renderer):
        return renderer.MeshArtist.create(backend_figure, self)


class MeshVectorPlot(MeshPlot):
    """Vector field sampled at mesh vertices, drawn as arrows.

    Cells are not needed for the arrows themselves - combine with a GeometryPlot
    or MeshFieldPlot in the same Figure to show the underlying domain.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        vector: VectorSpec | Variable,
        controls: list[ControlSpec] | None = None,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, show_edges=False, **kwargs)
        self.vector = vector if isinstance(vector, VectorSpec) else VectorSpec(vector)

        n_components = self.component_count(self.vector, data_config)
        if n_components != self.dim:
            raise ValueError(
                f"vector has {n_components} components but the mesh is {self.dim}D."
            )

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        slc = self.data_config.get_variable_slice(self.vector.variable_or_axes)
        vectors = np.asarray(data[slice_map(slc)]).reshape(-1, self.dim)
        if len(vectors) != self.n_vertices:
            raise ValueError(
                f"{self.vector.name} yields {len(vectors)} vectors but the mesh has "
                f"{self.n_vertices} vertices. Unresolved batch dimensions?"
            )

        magnitude = np.linalg.norm(vectors, axis=1)
        if self.vector.normalize:
            safe = np.where(magnitude > 0, magnitude, 1.0)[:, None]
            vectors = vectors / safe
        vectors = vectors * self.vector.scale

        positions = self.coord_transform.apply(self.mesh.vertices)
        return positions, vectors, magnitude

    def create_artist(self, backend_figure, renderer):
        return (
            renderer.Quiver2DArtist.create(backend_figure, self)
            if self.dim == 2
            else renderer.ConeArtist.create(backend_figure, self)
        )
