"""Plotly artists for vector fields."""

import numpy as np
from plotly import graph_objects as go

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import (
    PlotlyArtist,
    _bin_colors,
    _bin_values,
    _colorbar_carrier_trace,
    _spatial_variable,
    _to_numpy,
)


class _BinnedArrowArtist(PlotlyArtist):
    """Base class for arrow artists coloring arrows by value bins.

    Arrows are grouped into ``plot.n_bins`` value bins, one flat-colored
    trace per bin (some possibly empty), plus an invisible colorbar carrier
    trace. Without a color field, all arrows go into one trace in
    ``theme.vector_color`` and there is no colorbar.

    Subclasses must implement the static methods ``_arrow_trace`` (one
    bin's trace), ``_copy_arrow_trace`` (in-place update) and
    ``_set_axis_titles``, and set ``three_d`` for 3D.
    """

    three_d = False

    def __init__(self, arrow_indices, colorbar_idx=None, row=None, col=None):
        super().__init__(arrow_indices[0])
        self.arrow_indices = arrow_indices
        self.colorbar_idx = colorbar_idx
        self.row, self.col = row, col

    @property
    def trace_indices(self) -> list[int]:
        colorbar = [self.colorbar_idx] if self.colorbar_idx is not None else []
        return self.arrow_indices + colorbar

    @staticmethod
    def _bins(plot, result):
        """(masks, colors, colorbar): one arrow mask and one color per
        trace, and (cmap, vmin, vmax) or None for uncolored arrows."""
        n = len(result.positions)
        if result.color is None:
            return [np.ones(n, dtype=bool)], [plot.theme.vector_color], None
        cmap = plot.theme.resolve_cmap(plot.color.cmap)
        color = _to_numpy(result.color)
        bin_idx, centers, vmin, vmax = _bin_values(color, plot.n_bins, plot.color.scale)
        masks = [bin_idx == i for i in range(plot.n_bins)]
        return masks, _bin_colors(cmap, centers, vmin, vmax), (cmap, vmin, vmax)

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        result = plot.evaluate()
        positions, vectors = _to_numpy(result.positions), _to_numpy(result.vectors)
        masks, colors, colorbar = cls._bins(plot, result)

        arrow_indices = []
        for mask, color in zip(masks, colors):
            arrow_indices.append(len(backend_figure.data))
            backend_figure.add_trace(
                cls._arrow_trace(plot, positions[mask], vectors[mask], color),
                row=row,
                col=col,
            )

        colorbar_idx = None
        if colorbar is not None:
            cmap, vmin, vmax = colorbar
            colorbar_idx = len(backend_figure.data)
            backend_figure.add_trace(
                _colorbar_carrier_trace(
                    cmap, vmin, vmax, plot.color.scale, backend_figure, row, col,
                    three_d=cls.three_d,
                ),
                row=row,
                col=col,
            )

        geometry = plot.drawn_geometry_axes.geometry
        cls._set_axis_titles(backend_figure, _spatial_variable(geometry), row, col)
        return cls(arrow_indices, colorbar_idx, row, col)

    def update(self, backend_figure, plot, row=None, col=None):
        """Refresh this artist's traces from `plot`.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
        result = plot.evaluate()
        positions, vectors = _to_numpy(result.positions), _to_numpy(result.vectors)
        masks, colors, colorbar = self._bins(plot, result)

        for trace_idx, mask, color in zip(self.arrow_indices, masks, colors):
            new = self._arrow_trace(plot, positions[mask], vectors[mask], color)
            self._copy_arrow_trace(backend_figure.data[trace_idx], new)

        if self.colorbar_idx is not None and colorbar is not None:
            _, vmin, vmax = colorbar
            carrier = backend_figure.data[self.colorbar_idx]
            carrier.marker.cmin, carrier.marker.cmax = vmin, vmax


class ArrowField2DArtist(_BinnedArrowArtist):
    """Draws a 2D vector field as arrows.

    Used for 2D :class:`~qewton.visualization.plots.data.grid.QuiverPlot` and
    :class:`~qewton.visualization.plots.data.mesh.MeshVectorPlot`. Each arrow
    is a line segment from its position to position + vector with an arrow
    marker at the tip, rotated along the segment via
    ``angleref="previous"``. Coloring by value bins as described in
    ``_BinnedArrowArtist``.

    Args:
        arrow_indices (list of int): Indices of the arrow traces, one per bin.
        colorbar_idx (int, optional): Index of the colorbar carrier trace, if
            colored.
        row (int, optional): Facet grid row.
        col (int, optional): Facet grid column.

    Attributes:
        arrow_indices (list of int): Indices of the arrow traces.
        colorbar_idx (int or None): Index of the colorbar carrier trace.
        row (int or None): Facet grid row.
        col (int or None): Facet grid column.
        trace_indices (list of int): The arrow traces and, if present, the
            colorbar carrier.
    """

    @staticmethod
    def _arrow_trace(plot, positions, vectors, color):
        tips = positions + vectors
        n = len(positions)
        xs = np.full(3 * n, None, dtype=object)
        ys = np.full(3 * n, None, dtype=object)
        xs[0::3], xs[1::3] = positions[:, 0], tips[:, 0]
        ys[0::3], ys[1::3] = positions[:, 1], tips[:, 1]
        sizes = np.tile([0, 8, 0], n)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color=color),
            marker=dict(symbol="arrow", angleref="previous", size=sizes, color=color),
            opacity=plot.theme.opacity_default,
            showlegend=False,
            hoverinfo="skip",
        )

    @staticmethod
    def _copy_arrow_trace(trace, new):
        trace.x, trace.y = new.x, new.y
        trace.line, trace.marker = new.line, new.marker

    @staticmethod
    def _set_axis_titles(backend_figure, variable, row, col):
        x_name, y_name = axis_names_from_variable(variable, 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)


class ArrowField3DArtist(_BinnedArrowArtist):
    """Draws a 3D vector field as cones.

    Used for 3D :class:`~qewton.visualization.plots.data.grid.QuiverPlot` and
    :class:`~qewton.visualization.plots.data.mesh.MeshVectorPlot`. Each bin is
    one ``go.Cone`` trace with a single-color colorscale and
    ``sizemode="raw"``, so every cone is exactly as long as its vector.
    Coloring by value bins as described in ``_BinnedArrowArtist``.

    Args:
        arrow_indices (list of int): Indices of the cone traces, one per bin.
        colorbar_idx (int, optional): Index of the colorbar carrier trace, if
            colored.
        row (int, optional): Facet grid row.
        col (int, optional): Facet grid column.

    Attributes:
        arrow_indices (list of int): Indices of the cone traces.
        colorbar_idx (int or None): Index of the colorbar carrier trace.
        row (int or None): Facet grid row.
        col (int or None): Facet grid column.
        trace_indices (list of int): The cone traces and, if present, the
            colorbar carrier.
    """

    three_d = True

    @staticmethod
    def _arrow_trace(plot, positions, vectors, color):
        return go.Cone(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            u=vectors[:, 0],
            v=vectors[:, 1],
            w=vectors[:, 2],
            anchor="tail",
            sizemode="raw",
            sizeref=1.0,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=plot.theme.opacity_default,
            hoverinfo="skip",
        )

    @staticmethod
    def _copy_arrow_trace(trace, new):
        trace.x, trace.y, trace.z = new.x, new.y, new.z
        trace.u, trace.v, trace.w = new.u, new.v, new.w
        trace.colorscale = new.colorscale

    @staticmethod
    def _set_axis_titles(backend_figure, variable, row, col):
        x_name, y_name, z_name = axis_names_from_variable(variable, 3)
        backend_figure.update_scenes(
            row=row, col=col,
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name),
        )
