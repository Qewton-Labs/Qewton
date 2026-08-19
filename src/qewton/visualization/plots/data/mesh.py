import numpy as np

from qewton.config.axes import GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import MeshResult, VectorResult
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec, VectorSpec


class MeshPlot(DataPlot):
    """Base for plots on unstructured meshes (2D or 3D).

    Cells carry pure topology and never pass through spec resolution. Data
    variables are extracted per vertex, so the mesh dimension only constrains
    which concrete plot types are applicable.
    """

    #: Accepted vertex dimensions; subclasses narrow this where needed.
    supported_dims: tuple[int, ...] = (2, 3)

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = True,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geom_axes = data_config.geometry_axes
        assert isinstance(
            geom_axes, GeometryAxes
        ), "Currently only DataConfigurations with a single GeometryAxes are supported."
        if geom_axes is None or not isinstance(geom_axes.geometry, MeshGeometry):
            raise ValueError(
                f"{type(self).__name__} requires a GeometryAxes wrapping a MeshGeometry."
            )
        self.mesh = geom_axes.geometry.mesh
        self.dim = self.mesh.vertices.shape[1]
        if self.dim not in self.supported_dims:
            raise ValueError(
                f"{type(self).__name__} supports {self.supported_dims}D meshes, "
                f"got a {self.dim}D mesh."
            )
        self.show_edges = show_edges

    @property
    def n_vertices(self) -> int:
        return len(self.mesh.vertices)

    def render_cells(self) -> np.ndarray:
        """Cells to draw, as indices into the ORIGINAL vertex array.

        For volumetric meshes (tetrahedra in 3D) only the boundary is visible,
        so boundary_faces is used. Indices stay relative to mesh.vertices, which
        keeps per-vertex data aligned - unlike get_boundary_mesh(), which may
        reindex and is therefore only safe for data-free plots.
        """
        is_volumetric = self.mesh.cells.shape[1] == self.dim + 1
        return (
            self.mesh.boundary_faces
            if (is_volumetric and self.dim == 3)
            else self.mesh.cells
        )

    def scalar_at_vertices(self, spec, data, slice_map) -> np.ndarray:
        """Extract one scalar value per mesh vertex for the given spec."""
        slc = self.data_config.get_variable_slice(spec.variable_or_axes)
        values = np.asarray(data[slice_map(slc)])
        if values.size != self.n_vertices:
            raise ValueError(
                f"{spec.name} yields {values.size} values but the mesh has "
                f"{self.n_vertices} vertices. Unresolved batch dimensions? "
                "Add a SliderSpec or FixedSpec for them."
            )
        return values.reshape(-1)


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

    @property
    def embedding_dim(self) -> int:
        return 3  # both FilledMeshArtist and SurfaceMeshArtist draw a 3D-space mesh

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        color = self.scalar_at_vertices(self.color, data, slice_map)
        vertices = self.coord_transform.apply(self.mesh.vertices)
        return MeshResult(vertices=vertices, cells=self.render_cells(), color=color)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return (
            renderer.FilledMeshArtist.create(backend_figure, self, row=row, col=col)
            if self.dim == 2
            else renderer.SurfaceMeshArtist.create(backend_figure, self, row=row, col=col)
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
        return MeshResult(vertices=vertices, cells=self.mesh.cells, color=color)

    @property
    def embedding_dim(self) -> int:
        return 3

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.SurfaceMeshArtist.create(backend_figure, self, row=row, col=col)


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

        step = self.vector.subsample
        if step > 1:
            # Decimate after normalize/scale, not before: subsampling only
            # changes which arrows are drawn, never what a drawn arrow means.
            positions = positions[::step]
            vectors = vectors[::step]
            magnitude = magnitude[::step]

        return VectorResult(positions=positions, vectors=vectors, magnitude=magnitude)

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return (
            renderer.ArrowField2DArtist.create(backend_figure, self, row=row, col=col)
            if self.dim == 2
            else renderer.ArrowField3DArtist.create(backend_figure, self, row=row, col=col)
        )
