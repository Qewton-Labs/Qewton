"""Heatmap: a 2D domain drawn flat, its cells colored by a value."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.data.base import DataPlot, grid_values
from qewton.visualization.plots.base import PointEncoding
from qewton.visualization.plots.result import HeatmapResult, Result, SurfaceResult
from qewton.visualization.scales import ColorScale


class HeatmapPlot(DataPlot):
    """A 2D domain drawn flat, its cells colored by a value: a rectilinear
    grid as an image of cells, a mesh as filled triangles.

    Args:
        data, data_config, position, x, y, color, color_scale, controls,
            title, label, opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`; the
            positions are the domain's coordinates and `color` is required.
        show_edges: Whether the cell edges of a mesh are drawn.

    Attributes:
        show_edges (bool): Whether the cell edges of a mesh are drawn.
    """

    channels = ("x", "y", "color")
    required = ("color",)
    results = {(Structure.GRID, 2): HeatmapResult, (Structure.MESH, 2): SurfaceResult}
    dims = (2,)
    point_encoding = PointEncoding.COLOR
    domain_positions = True

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        *,
        position=None,
        x=None,
        y=None,
        color: Variable | Selector | None = None,
        color_scale: ColorScale | list[ColorScale] | None = None,
        controls: tuple[ControlAxis, ...] | list[ControlAxis] = (),
        title: str | None = None,
        label: str | None = None,
        opacity: float | None = None,
        show_edges: bool = False,
    ) -> None:
        super().__init__(
            data,
            data_config,
            position=position,
            x=x,
            y=y,
            color=color,
            color_scale=color_scale,
            controls=controls,
            title=title,
            label=label,
            opacity=opacity,
        )
        self.show_edges = show_edges

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`."""
        if self.domain.structure is Structure.GRID:
            x, y, values = grid_values(self.domain, color)
            return HeatmapResult(x=x, y=y, values=values)
        return SurfaceResult(
            positions=positions,
            cells=self.domain.cells,
            values=color,
            show_edges=self.show_edges,
        )
