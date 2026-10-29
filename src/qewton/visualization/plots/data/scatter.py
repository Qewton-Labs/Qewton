"""Scatter: a marker per element."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.visualization.axes import POSITION_CHANNELS, ControlAxis, Selector
from qewton.visualization.plots.base import PointEncoding
from qewton.visualization.plots.data.base import DataPlot
from qewton.visualization.plots.result import MarkerResult, Result
from qewton.visualization.scales import ColorScale


class ScatterPlot(DataPlot):
    """A marker per element: the points of a domain, or samples with
    positions from the values. With one position component the markers
    lie on the x axis of a 2D cell, e.g. the points of a 1D domain.

    Args:
        data, data_config, position, x, y, z, color, color_scale, controls,
            title, label, opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`.
        size: Marker size, or None for the theme's.
        symbol: Marker symbol, or None for the default.

    Attributes:
        size (float or None): Marker size.
        symbol (str or None): Marker symbol.
    """

    results = {(s, d): MarkerResult for s in Structure for d in range(4)}
    dims = (1, 2, 3)

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
        size: float | None = None,
        symbol: str | None = None,
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
        self.size = size
        self.symbol = symbol

    @property
    def point_encoding(self) -> PointEncoding | None:
        """Colored points of a domain show their values by color."""
        positions = [b for b in self.bindings if b.channel in POSITION_CHANNELS]
        colored = len(positions) < len(self.bindings)
        return PointEncoding.COLOR if colored and all(b.in_domain for b in positions) else None

    @property
    def dim(self) -> int:
        return max(2, self.n_positions)

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`;
        one position component is drawn at y = 0."""
        if positions.shape[1] == 1:
            positions = np.concatenate([positions, np.zeros_like(positions)], axis=1)
        return MarkerResult(
            positions=positions, values=color, size=self.size, symbol=self.symbol
        )
