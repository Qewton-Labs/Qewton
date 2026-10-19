"""Data plots on unstructured meshes."""

import numpy as np

from qewton.config.axes import GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.plots.data.base import DataPlot, VectorFieldPlotMixin
from qewton.visualization.plots.result import MeshResult
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, ControlSpec, VectorSpec


class MeshPlot(DataPlot):
    """Base class for plots on unstructured meshes (2D or 3D).

    The mesh is taken from the geometry of the single drawn GeometryAxes.
    Cells carry pure topology; data is extracted per vertex.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`; must have exactly one
            drawn GeometryAxes whose geometry has a mesh.
        controls: Control specs reducing `data`.
        show_edges: Whether cell edges are drawn.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        AssertionError: If there is not exactly one drawn GeometryAxes.
        ValueError: If the geometry has no mesh, or the mesh dimension is not
            in `supported_dims`.

    Attributes:
        supported_dims (tuple of int): Accepted vertex dimensions (class
            attribute).
        mesh (Mesh): The drawn mesh.
        dim (int): Vertex dimension of the mesh.
        show_edges (bool): Whether cell edges are drawn.
        cell_dim (int): Dimension of the mesh cells: 1 for line segments, 2 for
            triangles, 3 for tetrahedra.
        n_vertices (int): Number of mesh vertices.

    Notes:
        Subclasses must implement
        :meth:`~qewton.visualization.plots.base.Plot.evaluate` and
        :meth:`~qewton.visualization.plots.base.Plot.create_artist`, and may
        narrow `supported_dims`.
    """

    supported_dims: tuple[int, ...] = (2, 3)

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = False,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geom_axes = self.drawn_geometry_axes
        assert isinstance(
            geom_axes, GeometryAxes
        ), "Exactly one GeometryAxes without a control is supported."
        if geom_axes is None or getattr(geom_axes.geometry, "mesh", None) is None:
            raise ValueError(
                f"{type(self).__name__} requires a GeometryAxes whose geometry "
                "currently has a mesh (vertices + cell connectivity)."
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
    def cell_dim(self) -> int:
        return self.mesh.cells.shape[1] - 1

    @property
    def n_vertices(self) -> int:
        return len(self.mesh.vertices)

    def render_cells(self) -> np.ndarray:
        """Return the cells to draw, as indices into ``mesh.vertices``.

        For a volumetric 3D mesh (tetrahedra), these are the mesh's
        ``boundary_faces``; otherwise its cells. Indices are not
        renumbered, so per-vertex data stays aligned.

        Returns:
            numpy.ndarray: Cell connectivity.
        """
        is_volumetric = self.mesh.cells.shape[1] == self.dim + 1
        return (
            self.mesh.boundary_faces
            if (is_volumetric and self.dim == 3)
            else self.mesh.cells
        )

    def scalar_at_vertices(self, spec, data, slice_map) -> np.ndarray:
        """Extract one scalar value per mesh vertex for `spec`.

        Args:
            spec (PlotSpec): The spec to extract.
            data (array_like): The control-reduced data.
            slice_map (callable): The slice map returned by
                :meth:`apply_controls`.

        Returns:
            numpy.ndarray: One value per vertex.

        Raises:
            ValueError: If the number of values does not match the number of
                vertices.
        """
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
    """Scalar field colored on a mesh.

    Works in 2D (colored triangulation) and 3D (colored surface of a body),
    and on curves made of line segments in 2D or 3D (e.g. the boundary of a
    domain), which are drawn as colored lines. Evaluates to a
    :class:`~qewton.visualization.plots.result.MeshResult`.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`.
        color: The scalar field coloring the mesh.
        controls: Control specs reducing `data`.
        show_edges: Whether cell edges are drawn.
        n_bins: Number of value bins for renderers that draw one color per
            trace, e.g. Plotly's 2D triangulations and curves.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        ValueError: If `color` is not scalar.

    Attributes:
        color (ColorSpec): The color role.
        n_bins (int): Number of value bins.
        embedding_dim (int): 2 for a 2D mesh, else 3.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        color: ColorSpec | Variable,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = False,
        n_bins: int = 64,
        **kwargs,
    ):
        super().__init__(
            data, data_config, controls=controls, show_edges=show_edges, **kwargs
        )
        self.color = color if isinstance(color, ColorSpec) else ColorSpec(color)
        self.require_scalar(self.color, "color")
        self.n_bins = n_bins

    @property
    def embedding_dim(self) -> int:
        return 2 if self.dim == 2 else 3

    def evaluate(self):
        """Evaluate the colored mesh in the current control state.

        Returns:
            MeshResult: Transformed vertices, the cells to draw, and one color
                per vertex.
        """
        data, index_map, slice_map = self.apply_controls()
        color = self.scalar_at_vertices(self.color, data, slice_map)
        vertices = self.coord_transform.apply(self.mesh.vertices)
        return MeshResult(vertices=vertices, cells=self.render_cells(), color=color)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's curve, filled-mesh (2D) or surface-mesh
        (3D) artist for this plot."""
        if self.cell_dim == 1:
            return renderer.CurveFieldArtist.create(backend_figure, self, row=row, col=col)
        return (
            renderer.FilledMeshArtist.create(backend_figure, self, row=row, col=col)
            if self.dim == 2
            else renderer.SurfaceMeshArtist.create(backend_figure, self, row=row, col=col)
        )


class MeshSurfacePlot(MeshPlot):
    """Elevates a 2D triangulation into 3D using a data variable as height.

    x and y come from the mesh, z from the data. Evaluates to a
    :class:`~qewton.visualization.plots.result.MeshResult`.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`; the mesh must be 2D.
        z: The scalar height.
        color: The scalar field coloring the surface. Defaults to `z`.
        controls: Control specs reducing `data`.
        show_edges: Whether cell edges are drawn.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        ValueError: If `z` or `color` is not scalar.

    Attributes:
        z (AxisSpec): The height role.
        color (ColorSpec): The color role.
    """

    supported_dims = (2,)

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        z: AxisSpec | Variable,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = False,
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
        """Evaluate the elevated mesh in the current control state.

        Returns:
            MeshResult: 3D vertices (mesh x, y plus height), cells, and one
                color per vertex.
        """
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
        """Create the renderer's ``SurfaceMeshArtist`` for this plot."""
        return renderer.SurfaceMeshArtist.create(backend_figure, self, row=row, col=col)


class MeshVectorPlot(VectorFieldPlotMixin, MeshPlot):
    """Vector field sampled at mesh vertices, drawn as arrows.

    Color handling, NaN dropping and autoscaling follow
    :class:`~qewton.visualization.plots.data.base.VectorFieldPlotMixin`.
    Cells are not drawn; combine with a GeometryPlot or
    :class:`MeshFieldPlot` in the same Figure to show the domain. Evaluates
    to a :class:`~qewton.visualization.plots.result.VectorResult`.

    Args:
        data (array_like): The data to plot.
        data_config: Configuration describing `data`.
        vector: The vector field; its number of components must equal the mesh
            dimension.
        color: None to color by magnitude, a scalar ColorSpec or Variable to
            color by that field, or False for uncolored arrows.
        controls: Control specs reducing `data`.
        n_bins: Number of distinct colors the arrows are drawn in.
        **kwargs: Further arguments of
            :class:`~qewton.visualization.plots.data.base.DataPlot`.

    Raises:
        ValueError: If the vector's component count does not match the mesh
            dimension.

    Attributes:
        vector (VectorSpec): The vector role.
        color (ColorSpec or None): The color role, None for uncolored arrows.
        n_bins (int): Number of distinct arrow colors.
        embedding_dim (int): The mesh dimension.
    """

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        vector: VectorSpec | Variable,
        color: ColorSpec | Variable | bool | None = None,
        controls: list[ControlSpec] | None = None,
        n_bins: int = 32,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, show_edges=False, **kwargs)
        self.vector = vector if isinstance(vector, VectorSpec) else VectorSpec(vector)

        n_components = self.component_count(self.vector, data_config)
        if n_components != self.dim:
            raise ValueError(
                f"vector has {n_components} components but the mesh is {self.dim}D."
            )
        self._init_vector_color(color)
        self.n_bins = n_bins

    def evaluate(self):
        """Evaluate the arrows in the current control state.

        Returns:
            VectorResult: Arrow positions, vectors, magnitudes and colors.
        """
        data, index_map, slice_map = self.apply_controls()
        positions = self.coord_transform.apply(self.mesh.vertices)
        return self._evaluate_arrows(data, slice_map, positions)

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's 2D or 3D arrow-field artist for this plot."""
        return (
            renderer.ArrowField2DArtist.create(backend_figure, self, row=row, col=col)
            if self.dim == 2
            else renderer.ArrowField3DArtist.create(backend_figure, self, row=row, col=col)
        )
