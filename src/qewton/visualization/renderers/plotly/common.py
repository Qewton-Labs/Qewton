"""Shared Plotly artist base class and trace/colorbar helpers."""

import itertools

from plotly import graph_objects as go
from plotly.colors import sample_colorscale
import numpy as np

from qewton.backends import resolve_backend
from qewton.visualization.renderers.base import Artist


def _mesh_edges(cells: np.ndarray) -> np.ndarray:
    """All unique edges of a cell list (triangles, tetrahedra, ...), as
    vertex index pairs."""
    n = cells.shape[1]
    edge_pairs = list(itertools.combinations(range(n), 2))
    edges = np.concatenate(
        [np.sort(cells[:, [a, b]], axis=1) for a, b in edge_pairs], axis=0
    )
    return np.unique(edges, axis=0)


def _edge_trace(
    vertices: np.ndarray, cells: np.ndarray, color: str = "black", opacity: float = 1.0
) -> go.Scatter3d:
    """A 3D line trace drawing every cell edge as a wireframe."""
    edges = _mesh_edges(cells)
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [vertices[a, 0], vertices[b, 0], None]
        ys += [vertices[a, 1], vertices[b, 1], None]
        zs += [vertices[a, 2], vertices[b, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=1),
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


#: Width of the outline covering seams between filled triangles.
_SEAM_WIDTH = 0.5


def _triangle_fill_trace(
    vertices: np.ndarray, cells: np.ndarray, color: str, opacity: float = 1.0
) -> go.Scatter:
    """Fills a 2D triangulation as one trace, one None-separated segment
    per triangle (filled independently, so holes need no special handling;
    empty `cells` gives an empty trace). Each triangle is outlined thinly
    in its fill color to hide anti-aliasing seams."""
    xs, ys = [], []
    for tri in cells:
        pts = vertices[tri]
        xs.extend([pts[0, 0], pts[1, 0], pts[2, 0], pts[0, 0], None])
        ys.extend([pts[0, 1], pts[1, 1], pts[2, 1], pts[0, 1], None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        fill="toself",
        fillcolor=color,
        opacity=opacity,
        line=dict(color=color, width=_SEAM_WIDTH),
        hoverinfo="skip",
        showlegend=False,
    )


def _edge_trace_2d(
    vertices: np.ndarray,
    edges: np.ndarray,
    color: str = "black",
    width: float = 1.5,
    opacity: float = 1.0,
) -> go.Scatter:
    """Draws unordered 2D edges (e.g. boundary faces) as one line trace."""
    xs, ys = [], []
    for a, b in edges:
        xs.extend([vertices[a, 0], vertices[b, 0], None])
        ys.extend([vertices[a, 1], vertices[b, 1], None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


class PlotlyArtist(Artist):
    """Base class for Plotly artists.

    Args:
        idx (int): Index of the artist's (first) trace in
            ``backend_figure.data``.

    Attributes:
        figure_idx (int): Index of the artist's (first) trace in
            ``backend_figure.data``.
        trace_indices (list of int): Every trace in ``backend_figure.data`` the
            artist owns and refreshes in :meth:`update`, i.e. what
            :meth:`~qewton.visualization.renderers.plotly.PlotlyRenderer.animate`,
            :meth:`~qewton.visualization.renderers.plotly.PlotlyRenderer.apply_slider`
            and
            :meth:`~qewton.visualization.renderers.plotly.PlotlyRenderer.apply_selector`
            capture. ``[figure_idx]`` by default.

    Notes:
        Subclasses must implement ``create()`` and ``update()`` (see
        :class:`~qewton.visualization.renderers.base.Artist`). Subclasses that
        own more than one trace must override ``trace_indices``.
    """

    def __init__(self, idx) -> None:
        super().__init__()
        self.figure_idx = idx

    @property
    def trace_indices(self) -> list[int]:
        return [self.figure_idx]


def _mask_nan_color_as_gaps(x, y, z, color):
    """Propagates NaN in `color` into the position arrays.

    go.Surface only leaves a hole where x/y/z is NaN; NaN in surfacecolor
    alone is still drawn.
    """
    if color is None:
        return x, y, z
    x, y, z, color = (_to_numpy(v) for v in (x, y, z, color))
    mask = np.isnan(color)
    x[mask] = np.nan
    y[mask] = np.nan
    z[mask] = np.nan
    return x, y, z


def _spatial_variable(geometry):
    """The Variable naming a geometry's x/y/z axes, for axis titles.

    For a resampling grid with a `mesh_geometry` (PlaneSliceGeometry,
    VolumeGridGeometry), the source mesh's variable, since the grid's own
    variable names its parametrization; otherwise the geometry's own.
    """
    source = getattr(geometry, "mesh_geometry", None)
    return source.variable if source is not None else geometry.variable


def _backend_to_numpy(tensor) -> np.ndarray:
    """Converts via the tensor's ComputingBackend; anything without one
    (numpy arrays, lists, scalars) goes through np.asarray."""
    try:
        backend = resolve_backend(tensor)
    except ValueError:
        return np.asarray(tensor)
    return np.asarray(backend.to_numpy(tensor))


def _detach_to_numpy(tensor) -> np.ndarray:
    """Like _to_numpy, but keeps the dtype - for integer cell/index
    arrays."""
    return _backend_to_numpy(tensor)


def _to_numpy(tensor) -> np.ndarray:
    """A float copy of `tensor` as a plain numpy array.

    Converts and copies in two steps: np.array(tensor, copy=True) warns for
    torch tensors under numpy>=2.0.
    """
    return _backend_to_numpy(tensor).astype(float, copy=True)


def _subplot_x_domain(backend_figure, row, col) -> tuple[float, float] | None:
    """(x0, x1) of one grid cell in paper coordinates, as resolved by
    make_subplots(); None if the cell doesn't exist."""
    try:
        subplot = backend_figure.get_subplot(row, col)
    except (ValueError, KeyError):
        return None
    if subplot is None:
        return None
    domain = getattr(subplot, "domain", None)  # a 3D "scene" subplot
    return domain.x if domain is not None else subplot.xaxis.domain  # a 2D "xy" subplot


def _colorbar_position(backend_figure, row, col) -> dict | None:
    """Colorbar position and size in the gap right of a grid cell.

    Centered between this cell and the next one in the row (or just right
    of the last column), as tall as the cell, with a fixed 14 px thickness
    so it fits the gap. None without a row/col or a matching cell.
    """
    if row is None or col is None:
        return None
    domain = _subplot_x_domain(backend_figure, row, col)
    if domain is None:
        return None
    x0, x1 = domain
    subplot = backend_figure.get_subplot(row, col)
    y_domain = getattr(subplot, "domain", None)
    y0, y1 = y_domain.y if y_domain is not None else subplot.yaxis.domain

    next_domain = _subplot_x_domain(backend_figure, row, col + 1)
    x = (x1 + next_domain[0]) / 2 if next_domain is not None else x1 + 0.02
    return dict(x=x, y=(y0 + y1) / 2, len=y1 - y0, thickness=14, thicknessmode="pixels")


def _apply_scale(
    scale,
    min_key: str = "cmin",
    max_key: str = "cmax",
    backend_figure=None,
    row=None,
    col=None,
    title: str | None = None,
) -> dict:
    """cmin/cmax/showscale/colorbar kwargs for a trace, given a plot's
    ColorSpec.scale.

    Without a shared Scale, Plotly infers the range and the colorbar is
    always shown. With one, the range is the Scale's trained range and only
    the first trace to claim it shows a colorbar (Scale.claim_colorbar()),
    placed next to Scale.colorbar_cell. min_key/max_key are the trace
    type's range keys (zmin/zmax for go.Heatmap). backend_figure/row/col
    place the colorbar next to its grid cell; `title` labels it.
    """
    if scale is None:
        kwargs = {"showscale": True}
        show = True
    else:
        show = scale.claim_colorbar()
        kwargs = {"showscale": show}
        rng = scale.range
        if rng is not None:
            kwargs[min_key], kwargs[max_key] = rng
    if show:
        # A shared Scale's single colorbar goes to the cell resolved by
        # Figure._assign_colorbar_cells().
        target_row, target_col = row, col
        if scale is not None and scale.colorbar_cell is not None:
            target_row, target_col = scale.colorbar_cell
        position = (
            _colorbar_position(backend_figure, target_row, target_col)
            if backend_figure is not None
            else None
        )
        if position is not None:
            kwargs["colorbar"] = position
        elif scale is not None:
            kwargs["colorbar"] = dict(x=1.02)
        if title is not None:
            kwargs.setdefault("colorbar", {})
            kwargs["colorbar"] = {**kwargs["colorbar"], "title": dict(text=title)}
    return kwargs


def _bin_values(values, n_bins, scale):
    """(bin_idx, bin_centers, vmin, vmax) for drawing `values` as `n_bins`
    flat-colored traces. vmin/vmax are the shared Scale's range when set,
    else the values' own range."""
    if scale is not None and scale.range is not None:
        vmin, vmax = scale.range
    elif len(values) > 0:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9  # degenerate (uniform field) - avoid a zero-width range
    edges = np.linspace(vmin, vmax, n_bins + 1)
    bin_idx = np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return bin_idx, centers, vmin, vmax


def _bin_colors(cmap, centers, vmin, vmax) -> list[str]:
    """One colorscale sample per bin center from _bin_values()."""
    return sample_colorscale(cmap, list(np.clip((centers - vmin) / (vmax - vmin), 0, 1)))


def _colorbar_carrier_trace(
    cmap, vmin, vmax, scale, backend_figure=None, row=None, col=None, three_d=False
):
    """Invisible single-point trace that shows the continuous colorbar for
    binned traces, with cmin/cmax set to the bin range. `three_d` makes it
    a go.Scatter3d for a `scene`."""
    marker = {
        "colorscale": cmap, "color": [vmin],
        **_apply_scale(scale, backend_figure=backend_figure, row=row, col=col),
        "cmin": vmin, "cmax": vmax,
    }
    if three_d:
        return go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", marker=marker,
            showlegend=False, hoverinfo="skip",
        )
    return go.Scatter(
        x=[None], y=[None], mode="markers", marker=marker,
        showlegend=False, hoverinfo="skip",
    )


def _cycled_color(plot) -> str:
    """The theme palette color for `plot`'s color_index (set by
    Figure.add_plot()), or the first palette color without one. Used by
    artists without data-driven coloring."""
    return plot.theme.get_color(plot.color_index if plot.color_index is not None else 0)
