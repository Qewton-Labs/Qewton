"""Line: a line through a one-dimensional domain."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import DiscreteGeometry, Structure
from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import LineResult, Result
from qewton.visualization.scales import ColorScale


class LinePlot(DataPlot):
    """A line through a one-dimensional domain in its order: along x for a
    graph (``x=T, y=u``) or through positions from the values for a path
    (``position=r``). A color varies along the line; a point set on a line
    is ordered by its coordinate.

    Args:
        data, data_config, position, x, y, z, color, color_scale, controls,
            title, label, opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`.
        width: Line width, or None for the theme's.
        dash: Dash pattern, e.g. ``"dash"``, or None for a solid line.

    Attributes:
        width (float or None): Line width.
        dash (str or None): Dash pattern.
    """

    results = {
        (Structure.GRID, 1): LineResult,
        (Structure.MESH, 1): LineResult,
        (Structure.POINTS, 0): LineResult,
    }

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
        width: float | None = None,
        dash: str | None = None,
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
        self.width = width
        self.dash = dash

    @classmethod
    def draws(cls, domain: DiscreteGeometry) -> bool:
        """As :meth:`~qewton.visualization.plots.data.base.DataPlot.draws`;
        a point set additionally needs one coordinate to be ordered by."""
        if domain.structure is Structure.POINTS and domain.dim != 1:
            return False
        return super().draws(domain)

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`."""
        return LineResult(
            positions=positions,
            segments=self._segments(),
            values=color,
            width=self.width,
            dash=self.dash,
        )

    def _segments(self) -> np.ndarray:
        if self.domain.structure is Structure.POINTS:
            order = np.argsort(self.domain.coordinates()[:, 0], kind="stable")
            return np.stack([order[:-1], order[1:]], axis=1)
        return self.domain.cells
