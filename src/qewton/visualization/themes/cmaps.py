"""qewton's own colormaps, as typed objects - `qewton.viz.cmaps`.

Two kinds, both usable as `ColorSpec(cmap=...)` or as a Theme's cmap field:

- Roles - `sequential`, `diverging`, `cyclic`: "whichever colormap of this
  kind the current Theme uses". Resolved only at draw time against the
  Figure's theme (`Theme.resolve_cmap()`), so switching LIGHT_THEME to
  DARK_THEME also switches every plot that asked for a role.
- Concrete colormaps - `meadow`, `dusk`, `redblue`, ... and the
  single-hue `reds`, `blues`, ...: one fixed colorscale from palette.py,
  independent of the theme.

A plain string is never interpreted by qewton - it is passed through as
the renderer's own colormap name (e.g. Plotly's "Viridis"), as is an
explicit renderer-native colorscale list.
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
    """Base class for qewton's own colormap types."""


@dataclass(frozen=True)
class ColormapRole(Colormap):
    """A kind of colormap (sequential/diverging/cyclic), resolved to the
    current Theme's `<name>_cmap` field at draw time."""

    name: str

    def __repr__(self) -> str:
        return f"cmaps.{self.name}"


@dataclass(frozen=True)
class QewtonColormap(Colormap):
    """One concrete colorscale - (position, "#rrggbb") stops from 0 to 1.
    Stored as tuples so instances stay immutable and can be shared as a
    Theme field default."""

    name: str
    colorscale: tuple[tuple[float, str], ...]

    @classmethod
    def from_stops(cls, name: str, stops) -> "QewtonColormap":
        return cls(name, tuple((float(pos), color) for pos, color in stops))

    def __repr__(self) -> str:
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
