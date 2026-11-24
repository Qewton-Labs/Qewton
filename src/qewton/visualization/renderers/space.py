"""The room beside the plot areas of a figure, for every renderer.

A figure is a grid of plot areas of :meth:`Figure.plot_sizes
<qewton.visualization.figure.Figure.plot_sizes>`; beside every plot area
stand the tick labels and titles of its axes, its cell title and the
colorbars of the scales whose last cell it is. Their room is estimated, in
pixels, from the number of visible characters and the font sizes, so a
renderer sizes its figure without measuring texts; the labels of a range
are estimated from its ends.

The colorbars a renderer draws are noted in a ``{key: colorbar}`` dict
(:func:`note_colorbar`), with their cell and slot and what they showed in
every state drawn; colorbars of one slot share their room. A legend stands
right of the colorbars of its cell.
"""

import numpy as np

from qewton.visualization.renderers.text import as_plain
from qewton.visualization.scales import log_ticks

#: Width of a character and height of a line, as fractions of the font size.
_CHAR_WIDTH = 0.6
_LINE_HEIGHT = 1.4

#: Length of a tick, gap between two elements, margin of the figure,
#: thickness of a colorbar and width of a legend's symbol with its gap to
#: the label, in pixels.
TICK_LENGTH = 5
GAP = 8
MARGIN = 12
COLORBAR_THICKNESS = 14
LEGEND_SYMBOL = 40


def text_width(text: str | None, size: float) -> float:
    """The estimated width of a text, in pixels: its visible characters
    (math as plain text, without sub- and superscript markers).

    Args:
        text: A text, possibly with ``$...$`` parts.
        size: Font size in pixels.

    Returns:
        float: The width; 0 without text.
    """
    plain = (as_plain(text) or "").replace("_", "").replace("^", "")
    return len(plain) * _CHAR_WIDTH * size


def labels_width(labels, size: float) -> float:
    """The width of the widest of some texts, in pixels.

    Args:
        labels (iterable of str): The texts.
        size: Font size in pixels.

    Returns:
        float: The width; 0 without texts.
    """
    return max((text_width(str(label), size) for label in labels), default=0.0)


def line_height(size: float) -> float:
    """The height of one line of text, in pixels.

    Args:
        size: Font size in pixels.

    Returns:
        float: The height.
    """
    return _LINE_HEIGHT * size


def tick_labels(lo: float, hi: float, log: bool = False) -> list[str]:
    """Labels like those of the ticks of a range: its ends at the
    precision of a step of about a fifth of it (log: its powers of ten).

    Args:
        lo, hi: The range, in data values.
        log: Whether the axis is logarithmic.

    Returns:
        list of str: The labels.
    """
    if log:
        return log_ticks(np.log10(lo), np.log10(hi))[1]
    span = hi - lo
    if not span > 0:
        return [f"{lo:g}"]
    step = 10.0 ** np.floor(np.log10(span / 5))
    decimals = max(0, -int(np.floor(np.log10(step))))
    return [f"{round(v / step) * step:.{decimals}f}" for v in (lo, hi)]


def note_colorbar(colorbars: dict, key, cell, slot: int, title, labels) -> None:
    """Note what a colorbar shows in the state drawn.

    Args:
        colorbars: The renderer's ``{key: colorbar}``.
        key: The colorbar's key.
        cell (Cell): The cell it stands beside.
        slot: Its place beside the cell.
        title (str or None): Its title.
        labels (iterable of str): Its tick labels.
    """
    colorbar = colorbars.setdefault(key, dict(cell=cell, slot=slot, titled=False, labels=set()))
    colorbar["titled"] |= bool(title)
    colorbar["labels"] |= set(labels)


def colorbar_width(colorbar: dict, theme) -> float:
    """The room a colorbar takes right of its cell, in pixels: a gap, the
    bar, its tick labels and its title beside them (turned, one line).

    Args:
        colorbar: As noted by :func:`note_colorbar`.
        theme (Theme): The theme.

    Returns:
        float: The width.
    """
    width = GAP + COLORBAR_THICKNESS + TICK_LENGTH
    width += labels_width(colorbar["labels"], theme.font_size_labels)
    if colorbar["titled"]:
        width += GAP + line_height(theme.font_size_axes)
    return width


def legend_width(labels, theme) -> float:
    """The room a legend takes right of its cell, in pixels: a gap, then a
    symbol and a label per entry.

    Args:
        labels (list of str): Its entries.
        theme (Theme): The theme.

    Returns:
        float: The width.
    """
    return GAP + LEGEND_SYMBOL + labels_width(labels, theme.font_size_labels)


def _axis_labels(cell, channel: str) -> list[str]:
    # The tick labels of a position axis, given or estimated.
    ticks = cell.axis_ticks(channel)
    if ticks is not None and ticks[1] is not None:
        return list(ticks[1])
    drawn = cell.axis_range(channel)
    if drawn is None:
        return ["0.0"]
    return tick_labels(*drawn, log=cell.axes[channel].scale.log)


#: Room a cell laying out its own axes (e.g. parallel coordinates) keeps
#: beside them for the labels of its outer axes, in pixels.
OWN_AXES_SIDE = 40


def _gutters(cell, colorbars_width: float, theme) -> dict:
    # Pixels left, right, above and below the plot area of a cell.
    labels, axes = theme.font_size_labels, theme.font_size_axes
    top = GAP + (line_height(axes) if cell.title else 0.0)
    if cell.dim == 0:
        # Its axes' titles and range ends above, range ends below.
        return dict(
            left=OWN_AXES_SIDE,
            right=OWN_AXES_SIDE + colorbars_width,
            top=top + line_height(axes) + line_height(labels),
            bottom=GAP + line_height(labels),
        )
    if cell.dim != 2:
        return dict(left=0.0, right=colorbars_width, top=top, bottom=0.0)
    left = TICK_LENGTH + GAP + labels_width(_axis_labels(cell, "y"), labels)
    if cell.axis_title("y"):
        left += GAP + line_height(axes)
    bottom = TICK_LENGTH + GAP + line_height(labels)
    if cell.axis_title("x"):
        bottom += GAP + line_height(axes)
    return dict(left=left, right=colorbars_width, top=top, bottom=bottom)


def grid(figure, colorbars: dict, theme, legends=()) -> dict:
    """The room of the grid of a figure: its plot sizes, the gutters of
    every column and row, and the place of every colorbar and legend
    beside its cell.

    Args:
        figure (Figure): The trained figure.
        colorbars: As noted by :func:`note_colorbar`.
        theme (Theme): The theme.
        legends (list of tuple): The labels of every legend and the cell
            it stands beside.

    Returns:
        dict: ``widths``, ``heights`` (the plot sizes), ``left``,
            ``right`` (per column), ``top``, ``bottom`` (per row), all in
            pixels, ``colorbars`` ({key: offset}, the pixels from the right
            edge of its plot area to the left edge of its bar) and
            ``legends`` ([offset] in the order given, to its left edge).
    """
    widths, heights = figure.plot_sizes()
    n_rows, n_cols = figure.grid_shape
    slots: dict = {}
    for key, colorbar in colorbars.items():
        cell_slots = slots.setdefault(id(colorbar["cell"]), {})
        cell_slots.setdefault(colorbar["slot"], []).append(key)
    offsets, room = {}, {}
    for cell_id, cell_slots in slots.items():
        edge = 0.0
        for slot in sorted(cell_slots):
            for key in cell_slots[slot]:
                offsets[key] = edge + GAP
            # Room for whatever any colorbar of the slot shows.
            shared = dict(
                titled=any(colorbars[k]["titled"] for k in cell_slots[slot]),
                labels=set().union(*(colorbars[k]["labels"] for k in cell_slots[slot])),
            )
            edge += colorbar_width(shared, theme)
        room[cell_id] = edge
    legend_offsets = []
    for labels, cell in legends:
        edge = room.get(id(cell), 0.0)
        legend_offsets.append(edge + GAP)
        room[id(cell)] = edge + legend_width(labels, theme)
    left, right = [0.0] * n_cols, [0.0] * n_cols
    top, bottom = [0.0] * n_rows, [0.0] * n_rows
    for cell in figure.cells:
        gutters = _gutters(cell, room.get(id(cell), 0.0), theme)
        c, r = cell.col - 1, cell.row - 1
        left[c], right[c] = max(left[c], gutters["left"]), max(right[c], gutters["right"])
        top[r], bottom[r] = max(top[r], gutters["top"]), max(bottom[r], gutters["bottom"])
    return dict(
        widths=widths,
        heights=heights,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        colorbars=offsets,
        legends=legend_offsets,
    )


def title_height(figure, theme) -> float:
    """The room of a figure's title above its grid, in pixels.

    Args:
        figure (Figure): The figure.
        theme (Theme): The theme.

    Returns:
        float: The height; 0 without a title.
    """
    return line_height(theme.font_size_title) + GAP if figure.title else 0.0
