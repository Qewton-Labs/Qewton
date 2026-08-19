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


def _edge_trace(vertices: np.ndarray, cells: np.ndarray, color: str = "black") -> go.Scatter3d:
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


def _apply_scale(scale, min_key: str = "cmin", max_key: str = "cmax") -> dict:
    """cmin/cmax/showscale kwargs for a trace, given a plot's ColorSpec.scale.

    Without a shared Scale (scale is None), the backend infers its own range
    per trace and always shows its colorbar. With one, cmin/cmax come from
    the range trained across every plot referencing it, and only the first
    trace to claim it shows a colorbar - see Scale.claim_colorbar().
    min_key/max_key differ per Plotly trace type (go.Heatmap uses zmin/zmax,
    go.Surface and go.Mesh3d use cmin/cmax).
    """
    if scale is None:
        return {"showscale": True}
    kwargs = {"showscale": scale.claim_colorbar()}
    rng = scale.range
    if rng is not None:
        kwargs[min_key], kwargs[max_key] = rng
    return kwargs
