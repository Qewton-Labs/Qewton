"""The theme as Matplotlib settings: rcParams, colormaps and color norms.

Theme sizes are in pixels, as Plotly takes them; Matplotlib takes points
(:func:`points`). The LaTeX look needs no LaTeX installation: text is set
in the theme's serif fonts, falling back to Matplotlib's bundled Computer
Modern (``cmr10``), and math in mathtext's Computer Modern.
"""

import functools
import re

import numpy as np
from cycler import cycler
from matplotlib import colormaps
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize

from qewton.visualization.scales import ColorScale, color_range

#: Generic font families, which name no font.
_GENERIC = ("serif", "sans-serif", "monospace", "cursive", "fantasy")

#: Fonts after the theme's: Computer Modern, then one for every glyph.
_FALLBACK_SERIF = ["cmr10", "DejaVu Serif"]


#: A CSS color function, e.g. ``rgba(20,24,28,0.12)``.
_CSS_RGB = re.compile(r"rgba?\(\s*([^)]*)\)")


def color(css: str):
    """A theme color as Matplotlib takes it: CSS ``rgb()``/``rgba()`` as
    a tuple, anything else (e.g. ``"#rrggbb"``) unchanged.

    Args:
        css: The color.

    Returns:
        str or tuple: A Matplotlib color.
    """
    match = _CSS_RGB.fullmatch(css.strip())
    if match is None:
        return css
    parts = [float(p) for p in match.group(1).split(",")]
    return tuple(p / 255 for p in parts[:3]) + tuple(parts[3:])


#: Pixels per inch of the theme's sizes, as in a browser.
PIXELS_PER_INCH = 96


def points(pixels: float) -> float:
    """A theme size in pixels as points (72 per inch), at
    :data:`PIXELS_PER_INCH`.

    Args:
        pixels: Size in pixels.

    Returns:
        float: Size in points.
    """
    return pixels * 72 / PIXELS_PER_INCH


def fonts(font_family: str) -> dict:
    """rcParams of a CSS-like font list, e.g. ``"Latin Modern Roman,
    Georgia, serif"``: its fonts in order, then the fallbacks.

    Args:
        font_family: Comma-separated font names, possibly ending with a
            generic family.

    Returns:
        dict: ``font.family`` and the font list of that family.
    """
    names = [n.strip().strip("'\"") for n in font_family.split(",") if n.strip()]
    generic = next((n for n in names if n in _GENERIC), "serif")
    named = [n for n in names if n not in _GENERIC]
    fallback = _FALLBACK_SERIF if generic == "serif" else ["DejaVu Sans"]
    return {"font.family": generic, f"font.{generic}": named + fallback}


def rc_params(theme, usetex: bool = False) -> dict:
    """The rcParams of a theme.

    Args:
        theme (Theme): The theme.
        usetex: Whether LaTeX renders all text.

    Returns:
        dict: rcParams, for ``matplotlib.rc_context``.
    """
    tick_length = 3.5 if theme.show_axis_line else 0.0
    background, text = color(theme.background_color), color(theme.text_color)
    line = color(theme.line_color)
    params = {
        "figure.facecolor": background,
        "axes.facecolor": background,
        "savefig.facecolor": background,
        "text.color": text,
        "axes.labelcolor": text,
        "axes.titlecolor": text,
        "axes.edgecolor": line,
        "xtick.color": line,
        "ytick.color": line,
        "xtick.labelcolor": text,
        "ytick.labelcolor": text,
        "xtick.major.size": tick_length,
        "ytick.major.size": tick_length,
        "axes.grid": theme.show_grid,
        "axes.axisbelow": True,
        "grid.color": color(theme.grid_color),
        "axes.spines.left": theme.show_axis_line,
        "axes.spines.bottom": theme.show_axis_line,
        "axes.spines.right": theme.show_axis_line and theme.axis_box,
        "axes.spines.top": theme.show_axis_line and theme.axis_box,
        "axes.prop_cycle": cycler(color=[color(c) for c in theme.primary_color_palette]),
        "lines.linewidth": points(theme.line_width),
        "lines.markersize": points(theme.marker_size),
        "font.size": points(theme.font_size_labels),
        "figure.titlesize": points(theme.font_size_title),
        "axes.titlesize": points(theme.font_size_axes),
        "axes.labelsize": points(theme.font_size_axes),
        "xtick.labelsize": points(theme.font_size_labels),
        "ytick.labelsize": points(theme.font_size_labels),
        "legend.fontsize": points(theme.font_size_labels),
        "legend.frameon": theme.legend_box,
        "legend.edgecolor": line,
        "legend.facecolor": background,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "text.usetex": usetex,
    }
    params.update(fonts(theme.font_family))
    return params


def colormap(resolved) -> Colormap:
    """The Matplotlib colormap of a colormap resolved by a theme.

    Args:
        resolved: The result of
            :meth:`~qewton.visualization.themes.base.Theme.resolve_cmap`:
            ``[position, color]`` stops, or a Matplotlib colormap name.

    Returns:
        matplotlib.colors.Colormap: The colormap.

    Raises:
        ValueError: If a name is no Matplotlib colormap.
    """
    if isinstance(resolved, str):
        if resolved not in colormaps:
            raise ValueError(f"{resolved!r} is no Matplotlib colormap.")
        return colormaps[resolved]
    return _from_stops(tuple((float(p), color(str(c))) for p, c in resolved))


@functools.cache
def _from_stops(stops: tuple) -> Colormap:
    return LinearSegmentedColormap.from_list("qewton", list(stops), N=256)


def norm(scale: ColorScale) -> Normalize:
    """The norm mapping transformed values of a color scale (log10 for a
    log scale) onto its colormap.

    Args:
        scale: The trained color scale.

    Returns:
        matplotlib.colors.Normalize: The norm of its transformed range.
    """
    return Normalize(*color_range(scale))


def mapped(values: np.ndarray, scale: ColorScale) -> np.ndarray:
    """Values in the space a :func:`norm` maps: log10 for a log scale,
    NaN where they cannot be drawn.

    Args:
        values: Values bound to `scale`.
        scale: The color scale.

    Returns:
        numpy.ndarray: The transformed values, masked where NaN.
    """
    return np.ma.masked_invalid(scale.transform(values))
