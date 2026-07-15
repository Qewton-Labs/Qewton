import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import Axes
from qewton.config.variables import Variable
from qewton.visualization.plots.spec import ColorSpec, ControlSpec, AxisSpec


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


class StructuredGridPlot(Plot):
    """Heatmap, Surface, Contour - using meshgrids"""

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        x: AxisSpec | Variable | Axes,  # TODO: in future we could also allow for slices
        y: AxisSpec | Variable | Axes,
        color: AxisSpec | Variable | Axes | None = None,
        controls: list[ControlSpec] | None = None,
    ):
        super().__init__(data, data_config, controls=controls)

        self.x = x if isinstance(x, AxisSpec) else AxisSpec(x)
        self.y = y if isinstance(y, AxisSpec) else AxisSpec(y)
        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color
            else None
        )

    def evaluate(self):
        data = apply_controls(self.data, self.controls, self.data_config)
        geom_axes = self.data_config.get_axis(type(None))  # -> GeometryAxes
        geometry = geom_axes.geometry

        result = {}
        for spec, key in ((self.x, "x"), (self.y, "y")):
            dim = geometry.dim_index_for_variable(spec.variable_or_axes)
            spec.coordinates = geometry.coordinates_for(spec.variable_or_axes)
            result[key] = np.moveaxis(
                data, dim, -1
            )  # Beispiel, an echte Extraktion anpassen

        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            result["color"] = data[slc]
        return result


class UnstructuredPointPlot(Plot):
    """Basis fuer Scatter/PointCloud/Graph-Node-Positionen."""

    def __init__(self, data, data_config, x, y, color=None, controls=None, **kwargs):
        super().__init__(data, data_config, controls=controls, **kwargs)
        geom_axes = data_config.get_axis(type(None))  # -> GeometryAxes
        if geom_axes is None or geom_axes.geometry.is_structured:
            raise ValueError(
                f"{type(self).__name__} requires an unstructured GeometryAxes."
            )
        self.x = x if isinstance(x, AxisSpec) else AxisSpec(x)
        self.y = y if isinstance(y, AxisSpec) else AxisSpec(y)
        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color
            else None
        )

    def evaluate(self):
        data = apply_controls(self.data, self.controls, self.data_config)
        geom_axes = self.data_config.get_axis(type(None))  # -> GeometryAxes
        geometry = geom_axes.geometry

        result = {
            "x": geometry.coordinates_for(self.x.variable_or_axes),
            "y": geometry.coordinates_for(self.y.variable_or_axes),
        }
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            result["color"] = data[slc]

        result["x"] = self.coord.apply(result["x"])
        result["y"] = self.coord.apply(result["y"])
        return result


class UnstructuredMeshPlot(Plot):
    """Fuer Mesh-Geometrien (2D/3D): Vertices + Zell-Topologie.
    'cells' ist eine dritte Art von Information neben Koordinate/Wert -
    reine Struktur, laeuft nie durch ein AxisSpec-Mapping."""

    def __init__(
        self,
        data,
        data_config,
        color=None,
        controls=None,
        show_edges: bool = True,
        **kwargs,
    ):
        super().__init__(data, data_config, controls=controls, **kwargs)
        geom_axes = data_config.get_axis(type(None))  # -> GeometryAxes
        # if not isinstance(geom_axes.geometry, MeshGeometry):
        #     raise ValueError(f"{type(self).__name__} requires a MeshGeometry.")
        self.mesh = geom_axes.geometry.mesh
        self.color = (
            (color if isinstance(color, ColorSpec) else ColorSpec(color))
            if color
            else None
        )
        self.show_edges = show_edges

    def evaluate(self):
        data = apply_controls(self.data, self.controls, self.data_config)
        result = {
            "vertices": self.coord.apply(self.mesh.vertices),
            "cells": self.mesh.cells,
        }
        if self.color is not None:
            slc = self.data_config.get_variable_slice(self.color.variable_or_axes)
            result["color"] = data[slc]
        return result
