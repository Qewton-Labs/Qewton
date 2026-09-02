import itertools

from plotly import graph_objects as go
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


def _triangle_fill_trace(
    vertices: np.ndarray, cells: np.ndarray, color: str, opacity: float = 1.0
) -> go.Scatter:
    """Fills a 2D triangulation as one trace. Each triangle is a
    None-separated segment, which Plotly fills independently - so holes and
    disconnected components need no special handling, and an empty `cells`
    yields a valid, empty trace rather than an error."""
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
        line=dict(width=0),  # no interior edges - boundary drawn separately
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
    """Draws unordered 2D edges. Works for boundary_faces directly - line
    segments need no traversal order, unlike filled polygons."""
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
    """Base class for a Plotly Artist, tracking which trace in
    `backend_figure.data` it owns."""

    def __init__(self, idx) -> None:
        super().__init__()
        self.figure_idx = idx


def _mask_nan_color_as_gaps(x, y, z, color):
    """Propagates NaN in `color` into the position arrays too.

    go.Surface only cuts an actual hole in the rendered mesh where the
    POSITION (x/y/z) is NaN - NaN in surfacecolor alone, with the geometry
    still fully defined, just leaves that patch oddly/flatly colored instead
    of removing it. Plots that decouple position from color (EmbeddedGridPlot:
    x/y/z come from the geometry, color from a separately-interpolated field
    that can be NaN outside the mesh) need this; plots where color already
    doubles as z (SurfaceArtist) don't - z is NaN there for free.
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
    """The Variable naming a geometry's x/y/z axes, for artists that draw a
    geometry's own positions and so have no AxisSpec of their own to title
    from (mesh/geometry/vector/embedded-grid plots) - see
    axis_names_from_variable() in plots/base.py.

    The SOURCE mesh's variable when this geometry is a resampling grid built
    from one (PlaneSliceGeometry/VolumeGridGeometry set `mesh_geometry`
    precisely so this is recoverable) - their own `variable` names the grid's
    parametrization (e.g. "u"/"v"/"height_level"), not the space it was
    resampled into, which would mislabel the axes. Otherwise the geometry's
    own variable directly - correct for a MeshGeometry (no indirection
    needed, its variable already is the ambient space), and for a hand-built
    parametric grid, whatever the caller named its own axes.
    """
    source = getattr(geometry, "mesh_geometry", None)
    return source.variable if source is not None else geometry.variable


def _backend_to_numpy(tensor) -> np.ndarray:
    """Converts via the tensor's own ComputingBackend where one is
    recognized (torch/tensorflow/...), so this stays correct for whichever
    backend produced the data instead of duck-typing detach()/cpu(). Plain
    numpy arrays and anything else unrecognized (python lists, scalars) fall
    back to np.asarray directly - resolve_backend only knows about tensor
    types wrapped by an actual ComputingBackend."""
    try:
        backend = resolve_backend(tensor)
    except ValueError:
        return np.asarray(tensor)
    return np.asarray(backend.to_numpy(tensor))


def _detach_to_numpy(tensor) -> np.ndarray:
    """Like _to_numpy, but preserves dtype - for cell/index arrays, which
    must stay integer (used for indexing elsewhere, e.g. _mesh_edges), so
    _to_numpy's float cast isn't an option. Handing Plotly a raw torch
    tensor directly (skipping this) triggers the same numpy>=2.0 /
    __array__ DeprecationWarning _to_numpy works around, just for `i`/`j`/`k`
    cell indices instead of vertex coordinates."""
    return _backend_to_numpy(tensor)


def _to_numpy(tensor) -> np.ndarray:
    """Plotly ultimately needs plain arrays anyway, so this is a safe place
    to detach from whichever backend (numpy/torch/...) produced the data.

    np.array(tensor, copy=True) triggers a numpy>=2.0 DeprecationWarning for
    a torch tensor specifically - torch's __array__ doesn't yet implement
    numpy's copy-keyword protocol, so numpy warns and falls back. Splitting
    into np.asarray() (the actual conversion) + .astype(copy=True) (the
    actual copy) sidesteps that entirely, with the same result.
    """
    return _backend_to_numpy(tensor).astype(float, copy=True)


def _subplot_x_domain(backend_figure, row, col) -> tuple[float, float] | None:
    """(x0, x1) of one grid cell, in paper-fraction coordinates - Plotly's
    own resolved domain, read back rather than recomputed (make_subplots()
    already accounted for spacing, subplot titles, ...). None if that cell
    doesn't exist."""
    try:
        subplot = backend_figure.get_subplot(row, col)
    except (ValueError, KeyError):
        return None
    if subplot is None:
        return None
    domain = getattr(subplot, "domain", None)  # a 3D "scene" subplot
    return domain.x if domain is not None else subplot.xaxis.domain  # a 2D "xy" subplot


def _colorbar_position(backend_figure, row, col) -> dict | None:
    """A colorbar position/size confined to the gap right of this trace's
    own subplot cell, sized to reliably fit there - None outside a
    faceted/multi-panel grid, where the caller's own default position is
    already fine as-is.

    `x` is the midpoint of the actual gap between this cell and the next
    one in the same row, not a fixed offset guessing where that gap is -
    a fixed offset plus Plotly's default colorbar width (30px) can exceed
    a tight gap and spill onto the next cell entirely, which is what "the
    colorbar sits on top of the next plot" was: `thickness` is also
    pinned to a narrow, known pixel width for the same reason - relying on
    Plotly's default leaves the actual rendered width (and thus whether it
    fits) up to the figure's overall pixel size, not this cell's own
    fraction of it. The last column has no next cell to center within, so
    it falls back to a small fixed margin past its own right edge - the
    same case a single, ungridded panel is in.
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
    scale, min_key: str = "cmin", max_key: str = "cmax", backend_figure=None, row=None, col=None
) -> dict:
    """cmin/cmax/showscale kwargs for a trace, given a plot's ColorSpec.scale.

    Without a shared Scale (scale is None), the backend infers its own range
    per trace and always shows its colorbar. With one, cmin/cmax come from
    the range trained across every plot referencing it, and only the first
    trace to claim it shows a colorbar - see Scale.claim_colorbar().
    min_key/max_key differ per Plotly trace type (go.Heatmap uses zmin/zmax,
    go.Surface and go.Mesh3d use cmin/cmax).

    backend_figure/row/col (all optional - a caller drawing into a facet/
    panel grid has them on hand already) place the colorbar within that
    cell instead of Plotly's own figure-wide default position, which
    several visible colorbars would otherwise collide at.
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
        # A shared Scale's colorbar goes after every panel referencing it
        # (Figure._assign_colorbar_cells() already resolved that to one
        # cell), not just this trace's own - it's one colorbar for the
        # whole group, so it belongs at the group's own right edge.
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
    return kwargs


def _cycled_color(plot) -> str:
    """The theme's palette color for `plot`'s own position among every plot
    in its Figure - used by artists whose family has no other data-driven
    coloring (a fixed line, a set of bars, ...), so multiple such plots
    overlaid in one Figure read as visually distinct traces instead of a
    fixed, unthemed color repeated for each. Falls back to the first
    palette color if `plot` was never added to a Figure (color_index is
    only ever set by Figure.add_plot())."""
    return plot.theme.get_color(plot.color_index if plot.color_index is not None else 0)
