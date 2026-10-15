"""
qewton_palette.py
==================

Plotly-native Farbdefinitionen aus der Qewton-Farbpalette, zur Nutzung durch
``qewton.visualization.themes``.

Sieben Farbfamilien, je vier Abstufungen A (blasseste Tönung) bis D (kräftigster/
dunkelster Ton) -- unverändert übernommen, nirgends neu gemischt:

    red     A=#fce3de  B=#f7a38e  C=#f38f7f  D=#bc4b4f
    orange  A=#fde7d7  B=#fbc299  C=#f9b381  D=#e76727
    yellow  A=#fdf0d0  B=#fee1a5  C=#fddb91  D=#eaa52b   (Daten-Knoten)
    green   A=#e2efe5  B=#aed6be  C=#94c9ad  D=#3e835a   (Loss-Knoten)
    blue    A=#e3ecf3  B=#b7d2e6  C=#99bcdf  D=#245691   (Modell-/Netzwerk-Knoten)
    violet  A=#e3dfed  B=#a09ec6  C=#8d89ba  D=#53548a
    brown   A=#ebe2d9  B=#b6a595  C=#a18e7f  D=#826b59

Jede Colorscale unten entsteht durch feine (64-stufige) Interpolation im
OKLab-Farbraum zwischen diesen festen Ankerpunkten -- wahrnehmungsgleich-
mäßiger als Plotlys eigene (lineare sRGB-)Interpolation zwischen wenigen
Stützstellen, die unterwegs oft bräunlich-graue Zwischentöne erzeugt. Plotly
akzeptiert eine Colorscale direkt als Liste von ``[position, "#rrggbb"]``-
Paaren (``colorscale=...``), ganz ohne globale Registrierung wie bei
matplotlib. Als typisierte Objekte (``qewton.viz.cmaps.meadow`` usw., plus
die Rollen ``cmaps.sequential/diverging/cyclic``) stehen sie in
``themes/cmaps.py`` -- so werden sie in ``ColorSpec(cmap=...)`` und den
``*_cmap``-Feldern des Themes verwendet.

Für nicht-matplotlib-freie Analyse/Notebook-Arbeit existiert außerdem
``qewton_colormaps.py`` mit denselben Farben als registrierte matplotlib-
Colormaps -- beide Module leiten aus denselben 28 Werten ab, keines
"gewinnt"; welches benutzt wird, hängt nur vom Werkzeug ab (Plotly hier
im Backend, matplotlib für Ad-hoc-Notebook-Plots).
"""

from __future__ import annotations

import math
from typing import Sequence


# --------------------------------------------------------------------------
# OKLab-Kern (Björn Ottosson) -- keine Abhängigkeiten außer der Standardlib.
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
    """Plotly-Colorscale (Liste von [position (0..1), '#rrggbb']) durch
    stückweise OKLab-Interpolation zwischen `anchors`. 64 Stützstellen sind
    fein genug, dass Plotlys eigene (lineare) Interpolation ZWISCHEN ihnen
    perzeptuell unauffällig bleibt, halten die Figure-JSON aber klein."""
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
# Chrome-Tokens aus dem Design System (project/tokens.json) -- NICHT aus den
# 7 SVG-Familien abgeleitet, sondern eigene neutrale Werte fuers Figuren-
# Chrome (Hintergrund, Tinte/Linien, Gitterlinien). Je ein Wert pro Theme,
# anders als die family-Swatches oben, die theme-unabhaengig sind.
# --------------------------------------------------------------------------

QEWTON_SURFACE_CANVAS: dict[str, str] = {"light": "#ffffff", "dark": "#0a0e13"}
QEWTON_INK: dict[str, str] = {"light": "#14181c", "dark": "#eef1f4"}
# Translucent hairline token (CSS rgba() -- Plotly accepts this directly as
# a shape/gridline color string) fuer dezente Gitterlinien/Haarlinien.
QEWTON_BORDER: dict[str, str] = {"light": "rgba(20,24,28,0.12)", "dark": "rgba(255,255,255,0.12)"}


# --------------------------------------------------------------------------
# Die 28 festen Ankerfarben aus graph_farben.svg -- unverändert.
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

# Die 3 im SVG schwarz umrandeten Node-/Karten-Farbpaare.
QEWTON_NODE_COLORS: dict[str, dict[str, str]] = {
    "data":  {"fill": QEWTON_FAMILIES["yellow"]["B"], "border": QEWTON_FAMILIES["yellow"]["C"]},
    "model": {"fill": QEWTON_FAMILIES["blue"]["B"],   "border": QEWTON_FAMILIES["blue"]["C"]},
    "loss":  {"fill": QEWTON_FAMILIES["green"]["B"],  "border": QEWTON_FAMILIES["green"]["C"]},
}

# CVD-optimierte Reihenfolge (worst-case ΔE 15.4 normalsichtig / 14.5 unter
# simulierter Protanopie/Deuteranopie zwischen benachbarten Familien).
QEWTON_CATEGORICAL_ORDER: list[str] = ["red", "yellow", "blue", "brown", "orange", "violet", "green"]
QEWTON_CATEGORICAL_COLORS: list[str] = [QEWTON_FAMILIES[f]["C"] for f in QEWTON_CATEGORICAL_ORDER]


# --------------------------------------------------------------------------
# Colorscales (Plotly-Format: Liste von [position, hex])
# --------------------------------------------------------------------------

def _sequential(name: str) -> list[list]:
    s = QEWTON_FAMILIES[name]
    return _oklab_colorscale(["#ffffff", s["A"], s["B"], s["C"], s["D"]])


QEWTON_SEQUENTIAL: dict[str, list[list]] = {name: _sequential(name) for name in QEWTON_FAMILIES}

QEWTON_MEADOW = _oklab_colorscale(
    [QEWTON_FAMILIES["violet"]["D"], QEWTON_FAMILIES["blue"]["D"],
     QEWTON_FAMILIES["green"]["D"], QEWTON_FAMILIES["yellow"]["D"], QEWTON_FAMILIES["yellow"]["A"]]
)  # wie "viridis": dunkles Violett -> Blau -> Grün -> blasses Gelb

QEWTON_DUSK = _oklab_colorscale(
    [QEWTON_FAMILIES["blue"]["D"], QEWTON_FAMILIES["violet"]["D"], QEWTON_FAMILIES["red"]["D"],
     QEWTON_FAMILIES["orange"]["D"], QEWTON_FAMILIES["yellow"]["D"], QEWTON_FAMILIES["yellow"]["B"], "#ffffff"]
)  # wie "plasma": Blau -> Violett -> Rot -> Orange -> Gelb -> Weiß

QEWTON_REDBLUE = _oklab_colorscale(
    [QEWTON_FAMILIES["red"]["D"], QEWTON_FAMILIES["red"]["C"], QEWTON_FAMILIES["red"]["B"],
     QEWTON_FAMILIES["red"]["A"], "#ffffff",
     QEWTON_FAMILIES["blue"]["A"], QEWTON_FAMILIES["blue"]["B"], QEWTON_FAMILIES["blue"]["C"],
     QEWTON_FAMILIES["blue"]["D"]]
)  # diverging, Pivot reines Weiß

QEWTON_ORANGEVIOLET_PAPER = _oklab_colorscale(
    [QEWTON_FAMILIES["orange"]["D"], QEWTON_FAMILIES["orange"]["C"], QEWTON_FAMILIES["orange"]["B"],
     QEWTON_FAMILIES["orange"]["A"], QEWTON_FAMILIES["brown"]["A"],
     QEWTON_FAMILIES["violet"]["A"], QEWTON_FAMILIES["violet"]["B"], QEWTON_FAMILIES["violet"]["C"],
     QEWTON_FAMILIES["violet"]["D"]]
)  # diverging, Pivot der warme Papier-Neutralton aus brown-A statt reinem Weiß

QEWTON_PHASE = _oklab_colorscale(
    [QEWTON_FAMILIES[n]["C"] for n in ("red", "orange", "yellow", "green", "blue", "violet", "red")]
)  # cyclic, schließt exakt -- für Phasenwinkel/periodische Größen

QEWTON_SPECTRUM = _oklab_colorscale(
    [QEWTON_FAMILIES[n]["C"] for n in ("red", "orange", "yellow", "green", "blue", "violet")]
)  # miscellaneous: echte Hue-Reihenfolge ohne brown, NICHT wahrnehmungsgleichmäßig
   # -- nur für viele unterscheidbare Bänder, nie für Daten mit Größenordnung

__all__ = [
    "QEWTON_SURFACE_CANVAS", "QEWTON_INK", "QEWTON_BORDER",
    "QEWTON_FAMILIES", "QEWTON_NODE_COLORS",
    "QEWTON_CATEGORICAL_ORDER", "QEWTON_CATEGORICAL_COLORS",
    "QEWTON_SEQUENTIAL", "QEWTON_MEADOW", "QEWTON_DUSK",
    "QEWTON_REDBLUE", "QEWTON_ORANGEVIOLET_PAPER", "QEWTON_PHASE", "QEWTON_SPECTRUM",
]
