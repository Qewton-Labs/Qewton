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
    """Arrows colored per arrow, drawn as one flat-colored trace per value
    bin - neither a single go.Scatter (2D) nor a single go.Cone (3D) can
    color its arrows by an arbitrary per-arrow value (Scatter has one line
    color per trace, Cone only colors by its own u/v/w norm, which breaks
    as soon as the arrows are normalized or colored by another field). Same
    approach as FilledMeshArtist: `plot.n_bins` traces, some possibly empty,
    so a redraw never changes the trace count, plus one invisible trace
    carrying the continuous colorbar.

    A plot with `color=False` draws all arrows into a single trace in the
    theme's flat `vector_color`, with no colorbar.

    Subclasses provide `_arrow_trace` (one bin's trace), `_copy_arrow_trace`
    (in-place update) and `_set_axis_titles`.
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
        """(masks, colors, colorbar) - one boolean mask over the arrows and
        one color per trace, and (cmap, vmin, vmax) for the colorbar, or
        None for flat, uncolored arrows."""
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

        geometry = plot.data_config.geometry_axes.geometry
        cls._set_axis_titles(backend_figure, _spatial_variable(geometry), row, col)
        return cls(arrow_indices, colorbar_idx, row, col)

    def update(self, backend_figure, plot, row=None, col=None):
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
    """Arrows as line segments with a rotated-marker arrowhead at the tip -
    Plotly has no native 2D quiver primitive in graph_objects, so this builds
    one from go.Scatter the same way common._edge_trace builds mesh edges:
    one (start, tip, None) triple per arrow in a single trace per color bin,
    with marker.angleref="previous" rotating the tip marker to match the
    segment direction."""

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
    """3D arrows via go.Cone - Plotly's native vector-field primitive, so
    (unlike the 2D case) no manual segment-building is needed. Cones use
    sizemode="raw": each cone is exactly as long as its (scaled) vector,
    the same convention the 2D artist draws tips at - "scaled" would size
    every bin's trace relative to its own max norm, so cones of different
    colors would no longer be comparable."""

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
