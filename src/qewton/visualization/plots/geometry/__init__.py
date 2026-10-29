"""Geometry: a geometry drawn without data."""

import numpy as np

from qewton.geometries.base import DiscreteGeometry, Geometry
from qewton.visualization.axes import POSITION_CHANNELS
from qewton.visualization.plots.base import Plot, PointEncoding
from qewton.visualization.plots.result import (
    LineResult,
    MarkerResult,
    Result,
    SurfaceResult,
)
from qewton.visualization.references import component_titles, title_of
from qewton.visualization.scales import Scale


class GeometryPlot(Plot):
    """A geometry without data, in a uniform color: points as markers,
    curves as lines, surfaces filled, volumes by their boundary. It uses
    the geometry interface and Results of the data plots.

    Args:
        geometry: A discrete geometry, drawn as it is, or a geometry meshed
            with ``create_mesh(max_vertex_distance)``; with 2 or 3
            coordinates.
        max_vertex_distance: Mesh resolution of a geometry that is not
            discrete; None for its default.
        show_edges: Whether the cell edges of surfaces are drawn.
        title, label: As for :class:`~qewton.visualization.plots.base.Plot`;
            the label defaults to the geometry's variable.
        opacity: Opacity of every element; None for opaque.

    Raises:
        ValueError: If the geometry does not have 2 or 3 coordinates.

    Attributes:
        geometry (DiscreteGeometry): The drawn geometry.
        show_edges (bool): Whether the cell edges of surfaces are drawn.
        opacity (float or None): Opacity of every element.
    """

    def __init__(
        self,
        geometry: Geometry,
        *,
        max_vertex_distance: float | None = None,
        show_edges: bool = True,
        title: str | None = None,
        label: str | None = None,
        opacity: float | None = None,
    ) -> None:
        super().__init__(title=title, label=label)
        if not isinstance(geometry, DiscreteGeometry):
            geometry = geometry.create_mesh(max_vertex_distance)
        if geometry.dim not in (2, 3):
            raise ValueError(
                f"GeometryPlot draws geometries with 2 or 3 coordinates, "
                f"{geometry.variable} has {geometry.dim}."
            )
        self.geometry = geometry
        self.show_edges = show_edges
        self.opacity = opacity
        self._positions = geometry.coordinates()

    @property
    def dim(self) -> int:
        return self.geometry.dim

    @property
    def point_encoding(self) -> PointEncoding | None:
        return PointEncoding.COLOR if self.geometry.topological_dim >= 2 else None

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        return POSITION_CHANNELS[: self.dim]

    def axis_title(self, channel: str) -> str | None:
        """See :meth:`~qewton.visualization.plots.base.Plot.axis_title`."""
        titles = component_titles(self.geometry.variable, self.dim)
        return titles[POSITION_CHANNELS.index(channel)]

    def train_scales(self, axes: dict[str, Scale]) -> None:
        """See :meth:`~qewton.visualization.plots.base.Plot.train_scales`."""
        for k, channel in enumerate(POSITION_CHANNELS[: self.dim]):
            axes[channel].train(self._positions[:, k])

    def evaluate(self) -> list[Result]:
        """See :meth:`~qewton.visualization.plots.base.Plot.evaluate`."""
        result = self._build()
        result.label = self.label if self.label is not None else title_of(
            self.geometry.variable
        )
        result.color_index = self.color_index
        result.opacity = self.opacity if self.opacity is not None else 1.0
        return [result]

    def _build(self) -> Result:
        # The primitive of the geometry's cells: none, segments, triangles
        # (a volume by its boundary).
        topological_dim = self.geometry.topological_dim
        if topological_dim == 0:
            return MarkerResult(positions=self._positions)
        if topological_dim == 1:
            return LineResult(positions=self._positions, segments=self.geometry.cells)
        cells = (
            self.geometry.cells if topological_dim == 2 else self.geometry.boundary_cells
        )
        return SurfaceResult(
            positions=self._positions,
            cells=np.asarray(cells),
            show_edges=self.show_edges,
        )
