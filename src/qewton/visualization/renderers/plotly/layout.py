"""Pixel layout of a Plotly figure: the figure size and the places of its
cells, cell titles, colorbars, title and embedded controls in paper
coordinates.

Plotly keeps no room for anything outside the plot areas but its outer
margins, which this layout sets to 0: the room beside every plot area
comes from :mod:`~qewton.visualization.renderers.space`; embedded sliders
stand below the grid, dropdowns right of it.
"""

from qewton.visualization.renderers import space
from qewton.visualization.renderers.space import GAP, MARGIN

#: Height of an embedded slider (current value, rail, step labels), width
#: of the play and pause buttons left of an animation's slider, height of a
#: dropdown and room for its arrow, in pixels.
SLIDER_HEIGHT = 80
BUTTONS_WIDTH = 140
DROPDOWN_HEIGHT = 34
DROPDOWN_ARROW = 40


def place(figure, colorbars: dict, theme, legends=(), sliders=(), dropdowns=()) -> dict:
    """The places of a figure's parts, in paper coordinates of a figure
    without margins.

    Args:
        figure (Figure): The trained figure.
        colorbars, legends: As for
            :func:`~qewton.visualization.renderers.space.grid`.
        theme (Theme): The theme.
        sliders (list of bool): Per embedded slider, whether play buttons
            stand left of it.
        dropdowns (list of list of str): The labels of every embedded
            dropdown.

    Returns:
        dict: ``size`` (width, height in pixels), ``cells`` ({(row, col):
            (x0, x1, y0, y1)}), ``titles`` ({(row, col): (x, y)}, the
            bottom center of a cell title), ``colorbars`` ({key: (x, y,
            len)}, x of the left edge, y of the center), ``title`` ((x, y),
            the center of the figure title), ``legends`` ([(x, y)], their
            top left, in the order given), ``sliders`` ([(x, y, len)],
            y of the top edge), ``buttons`` ((x, y) of their top left, or
            None) and ``dropdowns`` ([(x, y)], their top left).
    """
    room = space.grid(figure, colorbars, theme, legends)
    n_rows, n_cols = figure.grid_shape
    xs, x = [], MARGIN
    for c in range(n_cols):
        x += room["left"][c]
        xs.append((x, x + room["widths"][c]))
        x += room["widths"][c] + room["right"][c]
    menus_x = x + GAP
    menu_width = max(
        (space.labels_width(labels, theme.font_size_labels) + DROPDOWN_ARROW for labels in dropdowns),
        default=-GAP,
    )
    width = menus_x + menu_width + MARGIN
    title = space.title_height(figure, theme)
    ys, y = [], MARGIN + title
    for r in range(n_rows):
        y += room["top"][r]
        ys.append((y, y + room["heights"][r]))
        y += room["heights"][r] + room["bottom"][r]
    sliders_y = y
    height = max(y + len(sliders) * SLIDER_HEIGHT, ys[0][0] + len(dropdowns) * DROPDOWN_HEIGHT)
    height += MARGIN

    # Paper coordinates: x from the left, y from the bottom.
    def px(value):
        return value / width

    def py(value):
        return 1.0 - value / height

    cells, titles, placed = {}, {}, {}
    for cell in figure.cells:
        (x0, x1), (y0, y1) = xs[cell.col - 1], ys[cell.row - 1]
        cells[(cell.row, cell.col)] = (px(x0), px(x1), py(y1), py(y0))
        titles[(cell.row, cell.col)] = (px((x0 + x1) / 2), py(y0 - GAP / 2))
    for key, offset in room["colorbars"].items():
        cell = colorbars[key]["cell"]
        (_, x1), (y0, y1) = xs[cell.col - 1], ys[cell.row - 1]
        placed[key] = (px(x1 + offset), py((y0 + y1) / 2), (y1 - y0) / height)
    placed_legends = []
    for (_, cell), offset in zip(legends, room["legends"]):
        (_, x1), (y0, _) = xs[cell.col - 1], ys[cell.row - 1]
        placed_legends.append((px(x1 + offset), py(y0)))
    grid_x0, grid_x1 = xs[0][0], xs[-1][1]
    placed_sliders, buttons = [], None
    for k, with_buttons in enumerate(sliders):
        top_y = sliders_y + k * SLIDER_HEIGHT
        start = grid_x0 + (BUTTONS_WIDTH if with_buttons else 0.0)
        placed_sliders.append((px(start), py(top_y), (grid_x1 - start) / width))
        if with_buttons:
            buttons = (px(grid_x0), py(top_y + GAP))
    return dict(
        size=(width, height),
        cells=cells,
        titles=titles,
        colorbars=placed,
        title=(px((grid_x0 + grid_x1) / 2), py(MARGIN + title / 2)),
        legends=placed_legends,
        sliders=placed_sliders,
        buttons=buttons,
        dropdowns=[(px(menus_x), py(ys[0][0] + k * DROPDOWN_HEIGHT)) for k in range(len(dropdowns))],
    )
