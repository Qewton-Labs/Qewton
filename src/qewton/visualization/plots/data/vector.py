"""Vector: an arrow per point."""

import numpy as np

from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.base import Structure
from qewton.visualization.axes import ControlAxis, Selector
from qewton.visualization.plots.base import PointEncoding
from qewton.visualization.plots.data.base import DataPlot, n_components
from qewton.visualization.plots.result import ArrowResult, Result
from qewton.visualization.scales import ColorScale, Scale, VectorScale


class VectorPlot(DataPlot):
    """An arrow per point, from a vector feature with one component per
    position component. A VectorScale turns the vectors into arrows in
    data units, so every renderer draws the same arrows. Bind ``color``
    to color them, e.g. ``color=v`` for the magnitude.

    Args:
        data, data_config, position, x, y, z, color, color_scale, controls,
            title, label, opacity: As for
            :class:`~qewton.visualization.plots.data.base.DataPlot`.
        vector: The vector feature.
        vector_scale: Scale of the arrows; a new one if None. Share one
            instance for equal arrow lengths across plots.
        subsample: Draw every n-th arrow.

    Raises:
        ValueError: As for DataPlot, and if `vector` is not a feature with
            one component per position component, or `subsample` is not
            positive.

    Attributes:
        vector_scale (VectorScale): Scale of the arrows.
        subsample (int): Every n-th arrow is drawn.
    """

    results = {(s, d): ArrowResult for s in Structure for d in range(4)}
    point_encoding = PointEncoding.ARROW

    def __init__(
        self,
        data,
        data_config: DataConfiguration,
        *,
        vector: Variable,
        position=None,
        x=None,
        y=None,
        z=None,
        color: Variable | Selector | None = None,
        color_scale: ColorScale | list[ColorScale] | None = None,
        vector_scale: VectorScale | None = None,
        subsample: int = 1,
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
            z=z,
            color=color,
            color_scale=color_scale,
            controls=controls,
            title=title,
            label=label,
            opacity=opacity,
        )
        if isinstance(vector, Selector) or self._in_domain(vector):
            raise ValueError("vector= takes one feature (a Variable of the values).")
        if n_components(vector) != self.dim:
            raise ValueError(
                f"{vector} has {n_components(vector)} components for {self.dim} "
                "position components."
            )
        if subsample < 1:
            raise ValueError(f"subsample must be positive, got {subsample}.")
        self._vector = vector
        self.vector_scale = vector_scale if vector_scale is not None else VectorScale()
        self.subsample = subsample

    @property
    def scales(self) -> list[Scale]:
        return [*self.color_scales, self.vector_scale]

    def train_scales(self, axes: dict[str, Scale]) -> None:
        """See :meth:`~qewton.visualization.plots.base.Plot.train_scales`;
        also trains the vector scale on the magnitudes."""
        super().train_scales(axes)
        for bindings, overlay_states in self._trained_copies():
            _, _, features = self._collect(bindings, overlay_states)
            vectors = self.feature_values(self._vector, features)
            self.vector_scale.train(np.linalg.norm(vectors, axis=-1))

    def build(
        self, positions: np.ndarray, color: np.ndarray | None, features: np.ndarray
    ) -> Result:
        """See :meth:`~qewton.visualization.plots.data.base.DataPlot.build`."""
        vectors = self.vector_scale.arrows(
            self.feature_values(self._vector, features), _spacing(positions)
        )
        step = self.subsample
        return ArrowResult(
            positions=positions[::step],
            vectors=vectors[::step],
            values=color[::step] if color is not None else None,
        )


def _spacing(positions: np.ndarray) -> float:
    # Typical distance between points: the edge of the volume (area,
    # length) per point over the extents that are not flat.
    extent = np.ptp(positions, axis=0)
    extent = extent[extent > 0]
    if extent.size == 0:
        return 1.0
    return float((np.prod(extent) / len(positions)) ** (1 / extent.size))
