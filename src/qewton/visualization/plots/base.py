import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import Axes, GeometryAxes
from qewton.config.variables import Variable
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.visualization.plots.spec import PlotSpec, ColorSpec, ControlSpec, AxisSpec


class CoordTransform:
    """transform points AFTER evaluate()"""

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityCoord(CoordTransform):
    def apply(self, vertices: np.ndarray) -> np.ndarray:
        return vertices


class Plot:
    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        title=None,
        theme=None,
        controls: list[ControlSpec] | None = None,
        coord_transform: CoordTransform | None = None,
    ) -> None:
        self.data = data
        self.data_config = data_config
        self._title = title
        self._theme = theme
        self.controls = controls or []
        self.coord_transform = coord_transform or IdentityCoord()

        for spec in self.controls:
            spec.resolve(self.data_config, self.data)

    @property
    def theme(self):
        return self._theme

    @theme.setter
    def theme(self, value):
        if self._theme is None:
            self._theme = value

    def preprocess(self, data):
        return data

    @property
    def title(self):
        return self._title

    def evaluate(self):
        raise NotImplementedError

    def create_artist(self, backend_figure, renderer):
        # uses self.data_config.evaluate_data(self.data, self.data_config)
        # to make it use the current state of the PlotAxis
        raise NotImplementedError

    def apply_controls(self):
        """Wendet Fixed/Slider/Facet-States an. Gibt zusaetzlich eine index_map
        zurueck, um andere, gegen das URSPRUENGLICHE `data` berechnete reale
        Dimensionsindizes (z.B. aus PlotSpec.get_slice) auf ihre Position im
        bereits reduzierten Array umzurechnen."""
        resolved = []
        for spec in self.controls:
            axis_slc, entry_slc = PlotSpec.get_slice(
                spec.variable_or_axes, self.data_config
            )
            assert (
                entry_slc is None
            ), f"It is not yet supported to use controls on variables:\
                {spec.variable_or_axes}."
            assert isinstance(
                axis_slc, int
            ), f"No multi-axis support for controls yet: {spec.variable_or_axes}."
            real_idx = axis_slc if axis_slc >= 0 else len(self.data.shape) + axis_slc
            resolved.append((spec, real_idx))

        # start from last index
        resolved.sort(key=lambda item: item[1], reverse=True)

        sliced = self.data[:]
        removed_indices = []
        for spec, real_idx in resolved:
            indexer = [slice(None)] * sliced.ndim
            indexer[real_idx] = spec.state
            sliced = sliced[tuple(indexer)]
            removed_indices.append(real_idx)

        def index_map(original_idx: int) -> int:
            shift = sum(1 for r in removed_indices if r < original_idx)
            return original_idx - shift

        def slice_map(slc: tuple) -> tuple:
            """Rechnet ein gegen das URSPRUENGLICHE `data_config` berechnetes
            Slice-Tupel (ein Eintrag pro urspruenglicher Dimension, z.B. aus
            get_variable_slice) auf die bereits durch die Controls reduzierten
            Dimensionen von `sliced` um, indem die von den Controls
            konsumierten Eintraege entfernt werden."""
            if Ellipsis in slc:
                raise NotImplementedError(
                    "slice_map unterstuetzt noch keine Slices mit Ellipsis."
                )
            return tuple(s for i, s in enumerate(slc) if i not in removed_indices)

        return sliced, index_map, slice_map


class GridPlot3d(Plot):
    """Heatmap, Surface, Contour - using meshgrids"""

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        z: AxisSpec | Variable | Axes | None = None,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, controls=controls)

        self.x = x if isinstance(x, AxisSpec) else AxisSpec(x)
        self.y = y if isinstance(y, AxisSpec) else AxisSpec(y)

        if isinstance(z, AxisSpec):
            self.z = z
        elif z is not None:
            self.z = AxisSpec(z)
        else:
            self.z = None

        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color
            else None
        )

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()

        # 1) X/Y structured - resolve to original self.data, then map to the
        #    already reduced `data`.
        x_idx = self._resolve_structural_dim(self.x)
        y_idx = self._resolve_structural_dim(self.y)

        if x_idx == y_idx:
            raise ValueError(
                f"{type(self).__name__}: x ({self.x.variable_or_axes}) und "
                f"y ({self.y.variable_or_axes}) refer to the same dimension. "
                "You might use an PointPlot or MeshPlot instead."
            )

        x_dim = index_map(x_idx)
        y_dim = index_map(y_idx)

        # 2) Color/Values BEFORE transpose - get_variable_slice returns a
        #    slice tuple for the original data_config, so map to the already
        #    reduced dimensions of `data` before applying it.
        values = data
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color = data[slice_map(slc)]
        else:
            color = None

        if self.z is not None:
            slc = self.data_config.get_variable_slice(self.z.variable_or_axes)
            values = values[slice_map(slc)]

        # 3) X/Y-Dimensions at the beginning (y, x, ...remaining dims)
        oriented = np.moveaxis(values, [y_dim, x_dim], [0, 1])
        if color is not None:
            color = np.moveaxis(color, [y_dim, x_dim], [0, 1])

        # could be moved to resolve also
        # self.x.coordinates = self._coordinates_for(self.x)
        # self.y.coordinates = self._coordinates_for(self.y)
        return oriented, color

    def _resolve_structural_dim(self, spec) -> int:
        axis_slc, entry_slc = PlotSpec.get_slice(spec.variable_or_axes, self.data_config)
        if entry_slc is not None:
            raise ValueError(
                f"{spec.variable_or_axes} refers to a channel slice, not a "
                "own dimension - not allowed for x/y in GridPlot3d."
            )
        if isinstance(axis_slc, slice):
            length = axis_slc.stop - axis_slc.start
            if length != 1:
                raise ValueError(
                    f"{spec.variable_or_axes} spans {length} dimensions - "
                    "x/y must refer to exactly one dimension."
                )
            real_idx = axis_slc.start
        else:
            real_idx = axis_slc
        return real_idx if real_idx >= 0 else self.data.ndim + real_idx


class PointPlot(Plot):
    pass


class MeshPlot(Plot):
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

    @staticmethod
    def component_count(spec, data_config: DataConfiguration) -> int:
        """Number of feature components a spec resolves to (1 means scalar).

        Used to reject vector variables where a scalar is required, which would
        otherwise silently render only the first component.
        """
        var = spec.variable_or_axes
        if isinstance(var, Variable):
            return var.dim
        slc = data_config.get_variable_slice(var)
        last = slc[-1] if isinstance(slc, tuple) else slc
        return (last.stop - last.start) if isinstance(last, slice) else 1

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

    def require_scalar(self, spec, role: str):
        n = self.component_count(spec, self.data_config)
        if n != 1:
            raise ValueError(
                f"{role} must be scalar (dim=1), got dim={n}. "
                "Use MeshVectorPlot for vector fields, or select a single component."
            )
