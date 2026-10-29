"""Qewton's colormaps as typed objects.

Available as ``qewton.viz.cmaps``. Two kinds can be passed as
``ColorScale(cmap=...)``:

- Roles - `sequential`, `diverging`, `cyclic`: the colormap of this kind
  that the current Theme uses. They are resolved at draw time against the
  Figure's theme
  (:meth:`~qewton.visualization.themes.base.Theme.resolve_cmap`), so
  switching the theme switches every plot that uses a role.
- Concrete colormaps - `meadow`, `dusk`, `redblue`, `orangeviolet_paper`,
  `phase`, `spectrum` and the single-hue `reds`, `oranges`, `yellows`,
  `greens`, `blues`, `violets`, `browns`: one fixed colorscale from
  :mod:`~qewton.visualization.themes.palette`, independent of the theme.
  Only these can be a Theme's ``*_cmap`` field.

A plain string or a colorscale list is not interpreted by Qewton; it is
passed through to the renderer as its own colormap (e.g. Plotly's
``"Viridis"``).
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.visualization.themes.palette import (
    QEWTON_DUSK,
    QEWTON_MEADOW,
    QEWTON_ORANGEVIOLET_PAPER,
    QEWTON_PHASE,
    QEWTON_REDBLUE,
    QEWTON_SEQUENTIAL,
    QEWTON_SPECTRUM,
)


class Colormap:
    """Base class for Qewton's colormap types.

    Notes:
        Has no behavior of its own. Subclasses define what the colormap
        resolves to in
        :meth:`~qewton.visualization.themes.base.Theme.resolve_cmap`,
        which handles :class:`ColormapRole` and :class:`QewtonColormap`
        only.
    """


@dataclass(frozen=True)
class ColormapRole(Colormap):
    """A kind of colormap, resolved against the Theme at draw time.

    Args:
        name: ``"sequential"``, ``"diverging"`` or ``"cyclic"``; the role
            resolves to the Theme field ``<name>_cmap``.
    """

    name: str

    def __repr__(self) -> str:
        """Return ``cmaps.<name>``, the role's module attribute."""
        return f"cmaps.{self.name}"


@dataclass(frozen=True)
class QewtonColormap(Colormap):
    """One concrete, immutable colorscale.

    Args:
        name: Name of the colormap.
        colorscale: ``(position, "#rrggbb")`` stops with positions from 0
            to 1.
    """

    name: str
    colorscale: tuple[tuple[float, str], ...]

    @classmethod
    def from_stops(cls, name: str, stops) -> "QewtonColormap":
        """Create a colormap from a list of stops.

        Args:
            name: Name of the colormap.
            stops (Iterable[tuple[float, str]]): ``(position, color)``
                pairs, e.g. ``[position, "#rrggbb"]`` lists as produced in
                :mod:`~qewton.visualization.themes.palette`.

        Returns:
            The colormap with the stops converted to tuples.
        """
        return cls(name, tuple((float(pos), color) for pos, color in stops))

    def __repr__(self) -> str:
        """Return ``cmaps.<name>``, the colormap's module attribute."""
        return f"cmaps.{self.name}"


# Roles
sequential = ColormapRole("sequential")
diverging = ColormapRole("diverging")
cyclic = ColormapRole("cyclic")

ROLES = (sequential, diverging, cyclic)

# Concrete multi-hue colormaps
meadow = QewtonColormap.from_stops("meadow", QEWTON_MEADOW)  # viridis-like
dusk = QewtonColormap.from_stops("dusk", QEWTON_DUSK)  # plasma-like, ends in white
redblue = QewtonColormap.from_stops("redblue", QEWTON_REDBLUE)  # diverging, white pivot
orangeviolet_paper = QewtonColormap.from_stops(
    "orangeviolet_paper", QEWTON_ORANGEVIOLET_PAPER
)  # diverging, warm paper-neutral pivot
phase = QewtonColormap.from_stops("phase", QEWTON_PHASE)  # cyclic
# NOT perceptually uniform - only for many distinguishable bands, never for
# data with a magnitude.
spectrum = QewtonColormap.from_stops("spectrum", QEWTON_SPECTRUM)

# Concrete single-hue sequential colormaps (white -> family's darkest shade)
reds = QewtonColormap.from_stops("reds", QEWTON_SEQUENTIAL["red"])
oranges = QewtonColormap.from_stops("oranges", QEWTON_SEQUENTIAL["orange"])
yellows = QewtonColormap.from_stops("yellows", QEWTON_SEQUENTIAL["yellow"])
greens = QewtonColormap.from_stops("greens", QEWTON_SEQUENTIAL["green"])
blues = QewtonColormap.from_stops("blues", QEWTON_SEQUENTIAL["blue"])
violets = QewtonColormap.from_stops("violets", QEWTON_SEQUENTIAL["violet"])
browns = QewtonColormap.from_stops("browns", QEWTON_SEQUENTIAL["brown"])

__all__ = [
    "Colormap", "ColormapRole", "QewtonColormap", "ROLES",
    "sequential", "diverging", "cyclic",
    "meadow", "dusk", "redblue", "orangeviolet_paper", "phase", "spectrum",
    "reds", "oranges", "yellows", "greens", "blues", "violets", "browns",
]
