"""Image: a rectilinear 2D grid of colors."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.data.base import DataPlot, grid_values
from qewton.visualization.plots.base import PointEncoding
from qewton.visualization.plots.result import ImageResult, Result


class ImagePlot(DataPlot):
    """A rectilinear 2D grid of colors: the color channel is an RGB or
    RGBA feature with components in [0, 1], drawn as it is (no
    ColorScale); components outside are clipped.

    Args:
        data, data_config, position, x, y, controls, title, label,
            opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`; the
            positions are the domain's coordinates.
        color: A feature (or a Selector of features) with 3 or 4
            components.

    Raises:
        ValueError: As for DataPlot, and if a color reference does not have
            3 or 4 components.
    """

    channels = ("x", "y", "color")
    required = ("color",)
    results = {(Structure.GRID, 2): ImageResult}
    dims = (2,)
    point_encoding = PointEncoding.COLOR
    domain_positions = True
    color_mapped = False

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        *,
        position=None,
        x=None,
        y=None,
        color: Variable | Selector | None = None,
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
            controls=controls,
            title=title,
            label=label,
            opacity=opacity,
        )
        candidates = color.candidates if isinstance(color, Selector) else [color]
        for candidate in candidates:
            if candidate.dim not in (3, 4):
                raise ValueError(
                    f"ImagePlot needs RGB or RGBA colors; {candidate} has "
                    f"{candidate.dim} components."
                )

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`."""
        x, y, colors = grid_values(self.domain, np.clip(color, 0.0, 1.0))
        return ImageResult(x=x, y=y, colors=colors)
