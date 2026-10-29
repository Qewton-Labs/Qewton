"""Scales: how data values are mapped to visual values."""

from enum import Enum, auto

import numpy as np


class Scale:
    """Maps data values to visual values (positions, colors, lengths).

    Its range is trained from every value bound to it, over all control
    states, before a figure is drawn; explicit bounds win. A shared instance
    links whatever uses it: one range for axes of several overlays, one
    colorbar for several plots.

    Args:
        vmin: Fixed lower bound.
        vmax: Fixed upper bound.
        log: Logarithmic mapping. Values that are not positive are ignored
            when training.
        symmetric: Range symmetric around 0, e.g. for errors.
        categorical: Values are categories, e.g. the index of a batch axis.

    Raises:
        ValueError: If `log` and `symmetric` are both set, or a fixed bound
            of a log scale is not positive.

    Attributes:
        vmin (float or None): Fixed lower bound.
        vmax (float or None): Fixed upper bound.
        log (bool): Logarithmic mapping.
        symmetric (bool): Range symmetric around 0.
        categorical (bool): Values are categories.
        range (tuple of float or None): ``(lo, hi)`` from the fixed bounds
            and the trained values, made symmetric if requested; None while
            neither is known.
    """

    def __init__(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        log: bool = False,
        symmetric: bool = False,
        categorical: bool = False,
    ) -> None:
        if log and symmetric:
            raise ValueError("A Scale cannot be logarithmic and symmetric around 0.")
        if log and any(b is not None and b <= 0 for b in (vmin, vmax)):
            raise ValueError(
                f"A logarithmic Scale needs positive bounds, got vmin={vmin}, vmax={vmax}."
            )
        self.vmin = vmin
        self.vmax = vmax
        self.log = log
        self.symmetric = symmetric
        self.categorical = categorical
        self._trained: tuple[float, float] | None = None

    def reset(self) -> None:
        """Forget the trained range, before the scale is trained anew."""
        self._trained = None

    def train(self, values) -> None:
        """Extend the trained range to cover `values`, ignoring NaN (and,
        for a log scale, values that are not positive).

        Args:
            values (array_like): Values bound to this scale.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if self.log:
            values = values[values > 0]
        if values.size == 0:
            return
        lo, hi = float(values.min()), float(values.max())
        if self._trained is not None:
            lo, hi = min(lo, self._trained[0]), max(hi, self._trained[1])
        self._trained = (lo, hi)

    @property
    def range(self) -> tuple[float, float] | None:
        trained = self._trained
        lo = self.vmin if self.vmin is not None else (trained[0] if trained else None)
        hi = self.vmax if self.vmax is not None else (trained[1] if trained else None)
        if lo is None or hi is None:
            return None
        if self.symmetric:
            bound = max(abs(lo), abs(hi))
            lo, hi = -bound, bound
        return lo, hi

    def transform(self, values) -> np.ndarray:
        """Map `values` into the scale's space: log10 for a log scale
        (NaN where not positive), unchanged otherwise. For renderers whose
        framework cannot map a log scale itself, e.g. for colors.

        Args:
            values (array_like): Values bound to this scale.

        Returns:
            numpy.ndarray: The transformed values, as floats.
        """
        values = np.asarray(values, dtype=float)
        if not self.log:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)


def log_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Ticks of a log scale: 1, 2 and 5 times powers of ten within
    the range, all leading digits if that gives fewer than three, only the
    powers of ten (thinned to at most eight) over many decades.

    Args:
        lo, hi: The range in log10 space.

    Returns:
        tuple: The tick positions in log10 space and their labels.
    """
    decades = range(int(np.floor(lo)), int(np.ceil(hi)) + 1)

    def within(mantissas):
        values = [m * 10.0**k for k in decades for m in mantissas]
        return [v for v in values if lo - 1e-9 <= np.log10(v) <= hi + 1e-9]

    values = within((1, 2, 5))
    if len(values) < 3:
        values = within(range(1, 10))
    if len(values) > 8:
        values = within((1,))
        values = values[:: int(np.ceil(len(values) / 8))]
    return [float(np.log10(v)) for v in values], [f"{v:g}" for v in values]


class Reduce(Enum):
    """How a value with several components is reduced to one for a scalar
    channel such as color.

    Attributes:
        MAGNITUDE: The Euclidean norm of the components.
    """

    MAGNITUDE = auto()


class ColorScale(Scale):
    """Maps values to colors; one colorbar is drawn per instance.

    Args:
        cmap (ColormapRole or QewtonColormap or str or list, optional): A
            colormap role (e.g. ``cmaps.diverging``), resolved against the
            figure's theme, a qewton colormap, or a renderer-native
            colormap. None uses the theme's sequential colormap.
        reduce: How a value with several components becomes one color
            value: :attr:`Reduce.MAGNITUDE` or the index of a component.
        vmin, vmax, log, symmetric, categorical: As for :class:`Scale`.

    Attributes:
        cmap (object or None): The colormap.
        reduce (Reduce or int): The reduction of several components.
    """

    def __init__(
        self,
        cmap=None,
        reduce: Reduce | int = Reduce.MAGNITUDE,
        vmin: float | None = None,
        vmax: float | None = None,
        log: bool = False,
        symmetric: bool = False,
        categorical: bool = False,
    ) -> None:
        super().__init__(vmin, vmax, log, symmetric, categorical)
        self.cmap = cmap
        self.reduce = reduce

    def reduce_components(self, values: np.ndarray) -> np.ndarray:
        """Reduce values of shape ``(n, k)`` to one value per element.

        Args:
            values: One row per element, one column per component.

        Returns:
            numpy.ndarray: Shape ``(n,)``.
        """
        if values.shape[-1] == 1:
            return values[..., 0]
        if self.reduce is Reduce.MAGNITUDE:
            return np.linalg.norm(values, axis=-1)
        return values[..., self.reduce]


class VectorScale(Scale):
    """Maps vectors to arrows: trained on the vector magnitudes, it scales
    the longest vector to `length` (untrained: the longest of those it
    maps).

    Args:
        length: Length of the longest arrow in data units; None for the
            typical point spacing of the drawn positions.
        normalize: Draw every arrow with the same length, showing only the
            direction.
        vmax: Fixed largest magnitude, drawn with `length`.

    Attributes:
        length (float or None): Length of the longest arrow.
        normalize (bool): Whether every arrow has the same length.
    """

    def __init__(
        self,
        length: float | None = None,
        normalize: bool = False,
        vmax: float | None = None,
    ) -> None:
        super().__init__(vmax=vmax)
        self.length = length
        self.normalize = normalize

    def arrows(self, vectors: np.ndarray, spacing: float) -> np.ndarray:
        """The arrows of `vectors`, in data units.

        Args:
            vectors: Shape ``(n, dim)``.
            spacing: Typical point spacing of the drawn positions; the
                length of the longest arrow if `length` is None.

        Returns:
            numpy.ndarray: Shape ``(n, dim)``.
        """
        length = self.length if self.length is not None else spacing
        magnitudes = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if self.normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(magnitudes > 0, vectors / magnitudes * length, 0.0)
        largest = self.range[1] if self.range is not None else magnitudes.max(initial=0)
        if largest <= 0:
            return np.zeros_like(vectors)
        return vectors / largest * length
