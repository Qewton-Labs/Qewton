"""Bar: a bar per point of a one-dimensional domain."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import BarResult, Result
from qewton.visualization.scales import ColorScale


class BarPlot(DataPlot):
    """A bar per point of a one-dimensional domain, from 0 to y: e.g. a
    value per batch element (``x=batch, y=u``).

    Args:
        data, data_config, position, x, y, color, color_scale, controls,
            title, label, opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`.
    """

    channels = ("x", "y", "color")
    results = {
        (Structure.GRID, 1): BarResult,
        (Structure.MESH, 1): BarResult,
        (Structure.POINTS, 0): BarResult,
    }
    dims = (2,)

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

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`."""
        return BarResult(positions=positions, values=color)
