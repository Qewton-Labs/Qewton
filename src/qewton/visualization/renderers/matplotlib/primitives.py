"""Matplotlib artists for the primitive Results, one per Result class and
number of position components.

An artist draws a Result into the Axes of its cell as Matplotlib objects
(:meth:`MatplotlibArtist.draw_into`); updating removes them and draws
anew, at the same place in the drawing order. Colors are continuous:
Matplotlib maps values through a colormap per element, so no value bins
are needed. Colorbars are drawn by the renderer, never by an artist.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.legend import Legend
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from qewton.visualization.renderers.base import Artist
from qewton.visualization.renderers.matplotlib.style import (
    color,
    colormap,
    mapped,
    norm,
    points,
)

#: Matplotlib dash pattern of each named one.
_DASHES = {
    "solid": "-",
    "dash": "--",
    "dot": ":",
    "dashdot": "-.",
    "longdash": (0, (10, 4)),
    "longdashdot": (0, (10, 4, 2, 4)),
}

#: Matplotlib marker of each named symbol.
_SYMBOLS = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "cross": "P",
    "x": "X",
    "triangle-up": "^",
    "triangle-down": "v",
    "triangle-left": "<",
    "triangle-right": ">",
    "star": "*",
    "pentagon": "p",
    "hexagon": "h",
}

#: Width of the edges of filled triangles, in pixels.
_EDGE_WIDTH = 0.5


def cell_label(cell) -> str:
    """The label of a cell's Axes, by which artists find them.

    Args:
        cell (Cell): The cell.

    Returns:
        str: E.g. ``"cell-1-2"``.
    """
    return f"cell-{cell.row}-{cell.col}"


def axes_of(backend_figure, cell) -> Axes:
    """The Axes of a cell in a figure built by the renderer.

    Args:
        backend_figure (matplotlib.figure.Figure): The figure.
        cell (Cell): The cell.

    Returns:
        matplotlib.axes.Axes: Its Axes.
    """
    label = cell_label(cell)
    return next(ax for ax in backend_figure.axes if ax.get_label() == label)


class MatplotlibArtist(Artist):
    """Base class of the Matplotlib artists: owns the objects it drew into
    one Axes.

    Args:
        axes: The Axes drawn into.
        objects: The Matplotlib objects drawn.
        zorder: Place in the drawing order, kept across updates.

    Attributes:
        axes (matplotlib.axes.Axes): The Axes drawn into.
        objects (list): The Matplotlib objects drawn.

    Notes:
        Subclasses implement :meth:`draw_into`.
    """

    def __init__(self, axes: Axes, objects: list, zorder: float) -> None:
        self.axes = axes
        self.objects = objects
        self._zorder = zorder
        self._visible = True

    @classmethod
    def draw_into(cls, axes: Axes, result, theme) -> list:
        """Draw `result` into `axes`.

        Args:
            axes: The Axes.
            result (Result): The Result to draw.
            theme (Theme): The renderer's theme.

        Returns:
            list: The Matplotlib objects drawn; the first carries the
                legend entry.
        """
        raise NotImplementedError

    @classmethod
    def create(cls, backend_figure, result, cell, theme) -> "MatplotlibArtist":
        """See :meth:`~qewton.visualization.renderers.base.Artist.create`.

        Artists are drawn in the order they are created, as in Plotly,
        whatever their kind."""
        axes = axes_of(backend_figure, cell)
        artist = cls(axes, [], zorder=1.0 + 1e-3 * len(axes.get_children()))
        artist._draw(result, theme)
        return artist

    def update(self, backend_figure, result, cell, theme) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Artist.update`."""
        for drawn in self.objects:
            drawn.remove()
        self._draw(result, theme)

    def set_visible(self, backend_figure, visible: bool) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Artist.set_visible`."""
        self._visible = visible
        for drawn in self.objects:
            drawn.set_visible(visible)

    def _draw(self, result, theme) -> None:
        self.objects = self.draw_into(self.axes, result, theme)
        for index, drawn in enumerate(self.objects):
            drawn.set_zorder(self._zorder)
            drawn.set_visible(self._visible)
            entry = index == 0 and result.label and _has_legend_entry(drawn)
            drawn.set_label(result.label if entry else "_nolegend_")
        self.axes.autoscale_view()


def _has_legend_entry(drawn) -> bool:
    # Whether Matplotlib draws a legend entry for an object (not for
    # meshes, e.g. heatmaps).
    return Legend.get_legend_handler(Legend.get_default_handler_map(), drawn) is not None


def _colors(result, theme, values=None):
    # One color per element mapped through the result's scale (RGBA,
    # transparent where NaN), or its uniform color.
    if result.color_scale is None:
        return color(theme.get_color(result.color_index))
    scale = result.color_scale
    values = result.values if values is None else values
    return colormap(theme.resolve_cmap(scale.cmap))(norm(scale)(mapped(values, scale)))


def _mapping(result, theme) -> dict:
    # Keyword arguments of a collection colored by the result's scale.
    scale = result.color_scale
    return dict(cmap=colormap(theme.resolve_cmap(scale.cmap)), norm=norm(scale))


class HeatmapArtist(MatplotlibArtist):
    """Draws a HeatmapResult as a ``pcolormesh`` of cells centered on the
    grid points."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        mesh = axes.pcolormesh(
            result.x,
            result.y,
            mapped(result.values, result.color_scale),
            shading="nearest",
            alpha=result.opacity,
            **_mapping(result, theme),
        )
        return [mesh]


class FlatMeshArtist(MatplotlibArtist):
    """Draws a SurfaceResult in 2D: colored triangles interpolated between
    their points (``tripcolor`` with Gouraud shading), or filled in one
    color; plus the edges if shown."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        x, y = result.positions.T
        triangulation = Triangulation(x, y, result.cells)
        if result.values is None:
            uniform = color(theme.get_color(result.color_index))
            fill = PolyCollection(
                result.positions[result.cells],
                facecolors=uniform,
                edgecolors=uniform,
                linewidths=points(_EDGE_WIDTH),
                alpha=result.opacity,
            )
            axes.add_collection(fill)
        else:
            fill = axes.tripcolor(
                triangulation,
                mapped(result.values, result.color_scale),
                shading="gouraud",
                alpha=result.opacity,
                **_mapping(result, theme),
            )
        objects = [fill]
        if result.show_edges:
            objects += axes.triplot(
                triangulation,
                color=color(theme.line_color),
                linewidth=points(_EDGE_WIDTH),
            )
        return objects


class SurfaceArtist(MatplotlibArtist):
    """Draws a SurfaceResult in 3D as a ``Poly3DCollection``, each triangle
    in the color of the mean value of its points, or in one shaded color;
    with edges if shown."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        if result.values is None:
            # Shaded by a light source, so that a uniform surface shows its
            # shape.
            colors = color(theme.get_color(result.color_index))
        else:
            colors = _colors(result, theme, result.values[result.cells].mean(axis=1))
        surface = Poly3DCollection(
            result.positions[result.cells],
            facecolors=colors,
            shade=result.values is None,
            alpha=result.opacity,
            **(
                dict(edgecolors=color(theme.line_color), linewidths=points(_EDGE_WIDTH))
                if result.show_edges
                else dict(linewidths=0)
            ),
        )
        axes.add_collection3d(surface)
        return [surface]


class Line2DArtist(MatplotlibArtist):
    """Draws a LineResult in 2D as a ``LineCollection``; a colored segment
    takes the mean value of its end points."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        lines = LineCollection(
            result.positions[result.segments], **_line_style(result, theme)
        )
        axes.add_collection(lines)
        return [lines]


class Line3DArtist(MatplotlibArtist):
    """Draws a LineResult in 3D as a ``Line3DCollection``, colored as in
    2D."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        lines = Line3DCollection(
            result.positions[result.segments], **_line_style(result, theme)
        )
        axes.add_collection3d(lines)
        return [lines]


def _line_style(result, theme) -> dict:
    # Keyword arguments of a line collection.
    width = result.width if result.width is not None else theme.line_width
    style = dict(
        linewidths=points(width),
        linestyles=_DASHES.get(result.dash, result.dash or "-"),
        alpha=result.opacity,
    )
    if result.color_scale is None:
        style["colors"] = color(theme.get_color(result.color_index))
    else:
        values = result.values[result.segments].mean(axis=1)
        style.update(array=mapped(values, result.color_scale), **_mapping(result, theme))
    return style


class MarkerArtist(MatplotlibArtist):
    """Draws a MarkerResult in 2D or 3D as a ``scatter``."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        size = points(result.size or theme.marker_size)
        style = dict(
            s=size**2,
            marker=_SYMBOLS.get(result.symbol, result.symbol or "o"),
            alpha=result.opacity,
            linewidths=0,
        )
        if result.color_scale is None:
            style["color"] = color(theme.get_color(result.color_index))
        else:
            style.update(
                c=mapped(result.values, result.color_scale), **_mapping(result, theme)
            )
        if result.positions.shape[1] == 3:
            style["depthshade"] = False
        return [axes.scatter(*result.positions.T, **style)]


class BarArtist(MatplotlibArtist):
    """Draws a BarResult as ``bar``: bars 0.8 of the smallest spacing wide."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        x, heights = result.positions.T
        spacing = np.diff(np.unique(x))
        width = 0.8 * (spacing.min() if spacing.size else 1.0)
        bars = axes.bar(
            x, heights, width=width, color=_colors(result, theme), alpha=result.opacity
        )
        return list(bars.patches)


class ImageArtist(MatplotlibArtist):
    """Draws an ImageResult as a ``pcolormesh`` of RGB(A) cells centered on
    the grid points, so the spacing may vary."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        image = axes.pcolormesh(
            result.x, result.y, result.colors, shading="nearest", alpha=result.opacity
        )
        return [image]


class Arrow2DArtist(MatplotlibArtist):
    """Draws an ArrowResult in 2D as a ``quiver`` whose arrows are as long
    as their vectors in data units."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        style = dict(angles="xy", scale_units="xy", scale=1.0, alpha=result.opacity)
        arguments = [*result.positions.T, *result.vectors.T]
        if result.color_scale is None:
            style["color"] = color(theme.get_color(result.color_index))
        else:
            arguments.append(mapped(result.values, result.color_scale))
            style.update(_mapping(result, theme))
        return [axes.quiver(*arguments, **style)]


class Arrow3DArtist(MatplotlibArtist):
    """Draws an ArrowResult in 3D as a ``quiver`` whose arrows are as long
    as their vectors, each in one color."""

    @classmethod
    def draw_into(cls, axes, result, theme) -> list:
        """See :meth:`MatplotlibArtist.draw_into`."""
        colors = _colors(result, theme)
        if result.color_scale is not None:
            # A 3D quiver colors its shafts, then the two head lines of
            # each arrow.
            colors = np.concatenate([colors, np.repeat(colors, 2, axis=0)])
        arrows = axes.quiver(
            *result.positions.T,
            *result.vectors.T,
            colors=colors,
            arrow_length_ratio=0.2,
            linewidths=points(theme.line_width),
            alpha=result.opacity,
        )
        return [arrows]
