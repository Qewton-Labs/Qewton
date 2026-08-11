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


class StructuredGridPlot(Plot):
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

        # 1) X/Y strukturell aufloesen - jeweils gegen das ORIGINAL self.data,
        #    danach durch index_map auf die Position im bereits reduzierten
        #    `data` umrechnen.
        x_idx = self._resolve_structural_dim(self.x)
        y_idx = self._resolve_structural_dim(self.y)

        if x_idx == y_idx:
            raise ValueError(
                f"{type(self).__name__}: x ({self.x.variable_or_axes}) und "
                f"y ({self.y.variable_or_axes}) loesen auf dieselbe Dimension auf. "
                "Das deutet auf eine unstrukturierte Geometrie (Punktwolke/Graph) hin - "
                "verwende dafuer UnstructuredPointPlot statt StructuredGridPlot."
            )

        x_dim = index_map(x_idx)
        y_dim = index_map(y_idx)

        # 2) Farbe/Werte VOR dem Transpose extrahieren - get_variable_slice
        #    liefert ein Slice-Tupel gegen das URSPRUENGLICHE data_config,
        #    daher ueber slice_map auf die bereits reduzierten Dimensionen
        #    von `data` umrechnen, bevor es angewendet wird.
        values = data
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color = data[slice_map(slc)]
        else:
            color = None

        if self.z is not None:
            slc = self.data_config.get_variable_slice(self.z.variable_or_axes)
            values = values[slice_map(slc)]

        # 3) X/Y-Dimensionen an den Anfang bringen (y, x, ...restliche Dims)
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
                f"{spec.variable_or_axes} loest in eine Kanal-Slice auf, nicht in "
                "eine eigene Dimension - fuer x/y bei StructuredGridPlot nicht zulaessig "
            )
        if isinstance(axis_slc, slice):
            length = axis_slc.stop - axis_slc.start
            if length != 1:
                raise ValueError(
                    f"{spec.variable_or_axes} spannt {length} Dimensionen - "
                    "x/y muessen genau eine Dimension referenzieren."
                )
            real_idx = axis_slc.start
        else:
            real_idx = axis_slc
        return real_idx if real_idx >= 0 else self.data.ndim + real_idx


class UnstructuredPointPlot(Plot):
    pass


class UnstructuredMeshPlot(Plot):
    """Fuer echte Mesh-Geometrien (2D/3D) MIT zugehoerigen Daten - PDE-Loesung,
    NN-Aktivierung, etc. 'cells' ist reine Topologie und laeuft nie durch ein
    AxisSpec-Mapping, 'color' ist eine ganz normale FeatureAxes-Variable wie
    bei jedem anderen Plot-Typ."""

    def __init__(
        self,
        data,
        data_config,
        color: ColorSpec | Variable | None = None,
        controls: list[ControlSpec] | None = None,
        show_edges: bool = True,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)

        geom_axes = data_config.get_axis(GeometryAxes)
        if geom_axes is None or not isinstance(geom_axes.geometry, MeshGeometry):
            raise ValueError(
                f"{type(self).__name__} requires a GeometryAxes wrapping a MeshGeometry."
            )
        self.mesh = geom_axes.geometry.mesh
        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color
            else None
        )
        self.show_edges = show_edges

    def evaluate(self):
        data, index_map, slice_map = self.apply_controls()
        color = None
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            color = data[slice_map(slc)]
        return self.coord_transform.apply(self.mesh.vertices), self.mesh.cells, color

    def create_artist(self, backend_figure, renderer):
        return renderer.MeshArtist.create(backend_figure, self)
