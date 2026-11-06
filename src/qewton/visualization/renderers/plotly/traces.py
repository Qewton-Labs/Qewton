"""Building blocks of Plotly traces shared by the primitive artists and
the renderer: value bins, segment and triangle coordinates."""

import numpy as np
from plotly.colors import sample_colorscale

from qewton.visualization.scales import ColorScale, color_range

#: Number of value bins of artists drawing one color per trace.
N_BINS = 32


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
    colors = sample_colorscale(scale.colorscale(theme), list((centers - lo) / (hi - lo)))
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
