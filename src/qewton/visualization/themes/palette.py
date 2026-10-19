"""Qewton color palette as Plotly-native color values.

Seven color families with four shades each, from A (palest) to D
(strongest/darkest):

    red     A=#fce3de  B=#f7a38e  C=#f38f7f  D=#bc4b4f
    orange  A=#fde7d7  B=#fbc299  C=#f9b381  D=#e76727
    yellow  A=#fdf0d0  B=#fee1a5  C=#fddb91  D=#eaa52b   (data nodes)
    green   A=#e2efe5  B=#aed6be  C=#94c9ad  D=#3e835a   (loss nodes)
    blue    A=#e3ecf3  B=#b7d2e6  C=#99bcdf  D=#245691   (model nodes)
    violet  A=#e3dfed  B=#a09ec6  C=#8d89ba  D=#53548a
    brown   A=#ebe2d9  B=#b6a595  C=#a18e7f  D=#826b59

Every colorscale is interpolated in OKLab space between fixed anchors
taken from these families, sampled at 64 stops, in Plotly's format: a list
of ``[position, "#rrggbb"]`` pairs. Besides the families, the module
defines theme-dependent chrome colors (background, ink, border), the node
card colors and the categorical color order.

:mod:`~qewton.visualization.themes.cmaps` wraps the colorscales as typed
colormap objects; :class:`~qewton.visualization.themes.base.Theme` uses
the remaining values.
"""

from __future__ import annotations

import math
from typing import Sequence


# --------------------------------------------------------------------------
# OKLab conversion (Björn Ottosson), standard library only.
# --------------------------------------------------------------------------

def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin2s(c: float) -> float:
    c = max(0.0, min(1.0, c))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _hex2srgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore


def _srgb2hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _cbrt_signed(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


def _oklab_from_lin(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l = _cbrt_signed(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = _cbrt_signed(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = _cbrt_signed(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return (L, a, bb)


def _lin_from_oklab(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    L, a, b = lab
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return (r, g, bl)


def _oklab_from_hex(h: str) -> tuple[float, float, float]:
    return _oklab_from_lin(tuple(_s2lin(c) for c in _hex2srgb(h)))  # type: ignore


def _mix_oklab_hex(h1: str, h2: str, t: float) -> str:
    L1, a1, b1 = _oklab_from_hex(h1)
    L2, a2, b2 = _oklab_from_hex(h2)
    lab = (L1 + (L2 - L1) * t, a1 + (a2 - a1) * t, b1 + (b2 - b1) * t)
    return _srgb2hex(tuple(_lin2s(c) for c in _lin_from_oklab(lab)))


def _oklab_colorscale(anchors: Sequence[str], n: int = 64) -> list[list]:
    """Plotly colorscale (list of [position (0..1), '#rrggbb']) by
    piecewise OKLab interpolation between `anchors`, sampled at `n` stops.
    With 64 stops, Plotly's linear interpolation between them is not
    noticeable, and the figure JSON stays small."""
    k = len(anchors) - 1
    out = []
    for i in range(n):
        t = i / (n - 1)
        seg = min(int(t * k), k - 1)
        t0, t1 = seg / k, (seg + 1) / k
        local_t = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        out.append([t, _mix_oklab_hex(anchors[seg], anchors[seg + 1], local_t)])
    return out


# --------------------------------------------------------------------------
# Chrome tokens from the design system (project/tokens.json): neutral
# values for figure chrome (background, ink/lines, gridlines), not derived
# from the color families. One value per theme, unlike the families, which
# are theme-independent.
# --------------------------------------------------------------------------

QEWTON_SURFACE_CANVAS: dict[str, str] = {"light": "#ffffff", "dark": "#0a0e13"}
QEWTON_INK: dict[str, str] = {"light": "#14181c", "dark": "#eef1f4"}
# Translucent hairline token for subtle gridlines (CSS rgba(), which Plotly
# accepts as a color string).
QEWTON_BORDER: dict[str, str] = {"light": "rgba(20,24,28,0.12)", "dark": "rgba(255,255,255,0.12)"}


# --------------------------------------------------------------------------
# The 28 fixed anchor colors (from graph_farben.svg).
# --------------------------------------------------------------------------

QEWTON_FAMILIES: dict[str, dict[str, str]] = {
    "red":    {"A": "#fce3de", "B": "#f7a38e", "C": "#f38f7f", "D": "#bc4b4f"},
    "orange": {"A": "#fde7d7", "B": "#fbc299", "C": "#f9b381", "D": "#e76727"},
    "yellow": {"A": "#fdf0d0", "B": "#fee1a5", "C": "#fddb91", "D": "#eaa52b"},
    "green":  {"A": "#e2efe5", "B": "#aed6be", "C": "#94c9ad", "D": "#3e835a"},
    "blue":   {"A": "#e3ecf3", "B": "#b7d2e6", "C": "#99bcdf", "D": "#245691"},
    "violet": {"A": "#e3dfed", "B": "#a09ec6", "C": "#8d89ba", "D": "#53548a"},
    "brown":  {"A": "#ebe2d9", "B": "#b6a595", "C": "#a18e7f", "D": "#826b59"},
}

# Fill/border color pairs of the three node card roles.
QEWTON_NODE_COLORS: dict[str, dict[str, str]] = {
    "data":  {"fill": QEWTON_FAMILIES["yellow"]["B"], "border": QEWTON_FAMILIES["yellow"]["C"]},
    "model": {"fill": QEWTON_FAMILIES["blue"]["B"],   "border": QEWTON_FAMILIES["blue"]["C"]},
    "loss":  {"fill": QEWTON_FAMILIES["green"]["B"],  "border": QEWTON_FAMILIES["green"]["C"]},
}

# Categorical order optimized for color vision deficiency: worst-case ΔE
# between neighboring families is 15.4 for normal vision and 14.5 under
# simulated protanopia/deuteranopia.
QEWTON_CATEGORICAL_ORDER: list[str] = ["red", "yellow", "blue", "brown", "orange", "violet", "green"]
QEWTON_CATEGORICAL_COLORS: list[str] = [QEWTON_FAMILIES[f]["C"] for f in QEWTON_CATEGORICAL_ORDER]


# --------------------------------------------------------------------------
# Colorscales (Plotly format: list of [position, hex])
# --------------------------------------------------------------------------

def _sequential(name: str) -> list[list]:
    s = QEWTON_FAMILIES[name]
    return _oklab_colorscale(["#ffffff", s["A"], s["B"], s["C"], s["D"]])


QEWTON_SEQUENTIAL: dict[str, list[list]] = {name: _sequential(name) for name in QEWTON_FAMILIES}

QEWTON_MEADOW = _oklab_colorscale(
    [QEWTON_FAMILIES["violet"]["D"], QEWTON_FAMILIES["blue"]["D"],
     QEWTON_FAMILIES["green"]["D"], QEWTON_FAMILIES["yellow"]["D"], QEWTON_FAMILIES["yellow"]["A"]]
)  # like "viridis": dark violet -> blue -> green -> pale yellow

QEWTON_DUSK = _oklab_colorscale(
    [QEWTON_FAMILIES["blue"]["D"], QEWTON_FAMILIES["violet"]["D"], QEWTON_FAMILIES["red"]["D"],
     QEWTON_FAMILIES["orange"]["D"], QEWTON_FAMILIES["yellow"]["D"], QEWTON_FAMILIES["yellow"]["B"], "#ffffff"]
)  # like "plasma": blue -> violet -> red -> orange -> yellow -> white

QEWTON_REDBLUE = _oklab_colorscale(
    [QEWTON_FAMILIES["red"]["D"], QEWTON_FAMILIES["red"]["C"], QEWTON_FAMILIES["red"]["B"],
     QEWTON_FAMILIES["red"]["A"], "#ffffff",
     QEWTON_FAMILIES["blue"]["A"], QEWTON_FAMILIES["blue"]["B"], QEWTON_FAMILIES["blue"]["C"],
     QEWTON_FAMILIES["blue"]["D"]]
)  # diverging, pure white pivot

QEWTON_ORANGEVIOLET_PAPER = _oklab_colorscale(
    [QEWTON_FAMILIES["orange"]["D"], QEWTON_FAMILIES["orange"]["C"], QEWTON_FAMILIES["orange"]["B"],
     QEWTON_FAMILIES["orange"]["A"], QEWTON_FAMILIES["brown"]["A"],
     QEWTON_FAMILIES["violet"]["A"], QEWTON_FAMILIES["violet"]["B"], QEWTON_FAMILIES["violet"]["C"],
     QEWTON_FAMILIES["violet"]["D"]]
)  # diverging, warm paper-neutral pivot (brown A)

QEWTON_PHASE = _oklab_colorscale(
    [QEWTON_FAMILIES[n]["C"] for n in ("red", "orange", "yellow", "green", "blue", "violet", "red")]
)  # cyclic (closes exactly), for phase angles/periodic quantities

QEWTON_SPECTRUM = _oklab_colorscale(
    [QEWTON_FAMILIES[n]["C"] for n in ("red", "orange", "yellow", "green", "blue", "violet")]
)  # hue order without brown, NOT perceptually uniform: only for many
   # distinguishable bands, never for data with a magnitude

__all__ = [
    "QEWTON_SURFACE_CANVAS", "QEWTON_INK", "QEWTON_BORDER",
    "QEWTON_FAMILIES", "QEWTON_NODE_COLORS",
    "QEWTON_CATEGORICAL_ORDER", "QEWTON_CATEGORICAL_COLORS",
    "QEWTON_SEQUENTIAL", "QEWTON_MEADOW", "QEWTON_DUSK",
    "QEWTON_REDBLUE", "QEWTON_ORANGEVIOLET_PAPER", "QEWTON_PHASE", "QEWTON_SPECTRUM",
]
