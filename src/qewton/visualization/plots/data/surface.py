"""Surface: a surface drawn in 3D."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.base import PointEncoding
from qewton.visualization.plots.result import Result, SurfaceResult
from qewton.visualization.scales import ColorScale


class SurfacePlot(DataPlot):
    """A surface in 3D: a 2D domain lifted by a height from the values
    (``position=X, z=u``), a surface with 3D coordinates (e.g. the boundary
    of a body), or a volume drawn by its boundary. Bind ``color`` to color
    it, e.g. by the height.

    Args:
        data, data_config, position, x, y, z, color, color_scale, controls,
            title, label, opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`.
        show_edges: Whether the cell edges are drawn.

    Attributes:
        show_edges (bool): Whether the cell edges are drawn.
        cells (numpy.ndarray): The drawn triangles as point indices.
    """

    results = {
        (Structure.GRID, 2): SurfaceResult,
        (Structure.MESH, 2): SurfaceResult,
        (Structure.GRID, 3): SurfaceResult,
        (Structure.MESH, 3): SurfaceResult,
    }
    dims = (3,)
    point_encoding = PointEncoding.COLOR

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        *,
        position=None,
        x=None,
        y=None,
        z=None,
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
            z=z,
            color=color,
            color_scale=color_scale,
            controls=controls,
            title=title,
            label=label,
            opacity=opacity,
        )
        self.show_edges = show_edges
        self.cells = (
            self.domain.boundary_cells
            if self.domain.topological_dim == 3
            else self.domain.cells
        )

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`."""
        return SurfaceResult(
            positions=positions,
            cells=self.cells,
            values=color,
            show_edges=self.show_edges,
        )
