"""Building blocks of Plotly traces shared by the primitive artists and
the renderer: color mapping, value bins, segment and triangle coordinates,
colorbar placement."""

import numpy as np
from plotly.colors import sample_colorscale

from qewton.visualization.scales import ColorScale

#: Number of value bins of artists drawing one color per trace.
N_BINS = 32


def colorscale(scale: ColorScale, theme) -> str | list:
    """The Plotly colorscale of a ColorScale, resolved against a theme.

    Args:
        scale: The color scale.
        theme (Theme): The theme resolving colormap roles.

    Returns:
        str or list: A Plotly colorscale.
    """
    return theme.resolve_cmap(scale.cmap)


def color_range(scale: ColorScale) -> tuple[float, float]:
    """The trained range of a ColorScale in its transformed space (log10
    for a log scale), widened if degenerate.

    Args:
        scale: The trained color scale.

    Returns:
        tuple of float: ``(lo, hi)`` with ``lo < hi``; ``(0, 1)`` without a
            range.
    """
    if scale.range is None:
        return 0.0, 1.0
    lo, hi = (float(v) for v in scale.transform(np.asarray(scale.range)))
    return (lo, hi) if hi > lo else (lo, lo + 1e-9)


def bins(values: np.ndarray, scale: ColorScale, theme) -> tuple[np.ndarray, list[str]]:
    """Assign values to :data:`N_BINS` bins of a color scale.

    Args:
        values: Values bound to `scale`.
        scale: The trained color scale.
        theme (Theme): The theme resolving its colormap.

    Returns:
        tuple: The bin index of every value (-1 where it is NaN), and the
            color of every bin.
    """
    lo, hi = color_range(scale)
    transformed = scale.transform(values)
    edges = np.linspace(lo, hi, N_BINS + 1)
    index = np.clip(np.digitize(transformed, edges[1:-1]), 0, N_BINS - 1)
    index = np.where(np.isnan(transformed), -1, index)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = sample_colorscale(colorscale(scale, theme), list((centers - lo) / (hi - lo)))
    return index, colors


def segment_coordinates(positions: np.ndarray, segments: np.ndarray) -> list[list]:
    """Per position component, the end points of every segment separated by
    None, so one line trace draws the segments independently.

    Args:
        positions: Shape ``(n_points, dim)``.
        segments: Point index pairs, shape ``(n_segments, 2)``.

    Returns:
        list of list: One coordinate list per component.
    """
    columns = []
    for k in range(positions.shape[1]):
        ends = positions[segments, k]  # (n_segments, 2)
        separated = np.concatenate([ends, np.full((len(ends), 1), np.nan)], axis=1)
        columns.append([None if np.isnan(v) else float(v) for v in separated.ravel()])
    return columns


def triangle_coordinates(positions: np.ndarray, cells: np.ndarray) -> tuple[list, list]:
    """x and y of closed triangle outlines separated by None, for one filled
    2D trace per group of triangles.

    Args:
        positions: Shape ``(n_points, 2)``.
        cells: Triangles as point indices, shape ``(n_cells, 3)``.

    Returns:
        tuple of list: The x and the y coordinates.
    """
    outline = np.concatenate([cells, cells[:, :1]], axis=1)  # back to the first point
    columns = []
    for k in range(2):
        corners = positions[outline, k]
        separated = np.concatenate([corners, np.full((len(corners), 1), np.nan)], axis=1)
        columns.append([None if np.isnan(v) else float(v) for v in separated.ravel()])
    return columns[0], columns[1]


def colorbar_position(backend_figure, row: int, col: int) -> dict:
    """Position and size of a colorbar right of a grid cell: centered in the
    gap to the next cell (or just right of the last column), as tall as the
    cell.

    Args:
        backend_figure (plotly.graph_objects.Figure): A figure built with
            ``make_subplots``.
        row: Grid row of the cell, 1-indexed.
        col: Grid column of the cell, 1-indexed.

    Returns:
        dict: Plotly colorbar attributes.
    """
    x0, x1, y0, y1 = _cell_domain(backend_figure, row, col)
    next_domain = _cell_domain(backend_figure, row, col + 1)
    x = (x1 + next_domain[0]) / 2 if next_domain is not None else x1 + 0.02
    return dict(x=x, y=(y0 + y1) / 2, len=y1 - y0, thickness=14, thicknessmode="pixels")


def _cell_domain(
    backend_figure, row: int, col: int
) -> tuple[float, float, float, float] | None:
    # Paper coordinates (x0, x1, y0, y1) of a grid cell, None if empty.
    try:
        subplot = backend_figure.get_subplot(row, col)
    except (ValueError, KeyError):
        return None
    if subplot is None:
        return None
    domain = getattr(subplot, "domain", None)  # a 3D scene has a domain
    if domain is not None:
        return (*domain.x, *domain.y)
    return (*subplot.xaxis.domain, *subplot.yaxis.domain)
