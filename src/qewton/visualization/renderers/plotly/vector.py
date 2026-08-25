from plotly import graph_objects as go

from qewton.visualization.plots.base import axis_names_from_variable
from qewton.visualization.renderers.plotly.common import PlotlyArtist, _spatial_variable, _to_numpy


class ArrowField2DArtist(PlotlyArtist):
    """Arrows as line segments with a rotated-marker arrowhead at the tip -
    Plotly has no native 2D quiver primitive in graph_objects, so this builds
    one from go.Scatter the same way common._edge_trace builds mesh edges:
    one (start, tip, None) triple per arrow in a single trace, with
    marker.angleref="previous" rotating the tip marker to match the segment
    direction.

    Per-segment line color isn't supported by a single go.Scatter trace, so
    color_by_magnitude colors only the tip markers, continuously; the shafts
    stay a flat theme color. VectorSpec.n_color_bins (discrete per-arrow
    coloring) is not wired up here - it would need one trace per bin.
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        trace = cls._trace(plot, plot.evaluate())
        backend_figure.add_trace(trace, row=row, col=col)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)
        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name = axis_names_from_variable(_spatial_variable(geometry), 2)
        backend_figure.update_xaxes(title=x_name, row=row, col=col)
        backend_figure.update_yaxes(title=y_name, row=row, col=col)
        return cls(len(backend_figure.data) - 1)

    @staticmethod
    def _trace(plot, result):
        positions = _to_numpy(result.positions)
        tips = positions + _to_numpy(result.vectors)
        magnitude = _to_numpy(result.magnitude)

        xs, ys, sizes, colors = [], [], [], []
        for (sx, sy), (tx, ty), mag in zip(positions, tips, magnitude):
            xs += [sx, tx, None]
            ys += [sy, ty, None]
            sizes += [0, 8, 0]
            colors += [mag, mag, mag]

        line_color = plot.theme.vector_color
        marker = dict(symbol="arrow", angleref="previous", size=sizes)
        if plot.vector.color_by_magnitude:
            cmap = plot.vector.cmap or plot.theme.default_cmap
            marker.update(color=colors, colorscale=cmap, showscale=True)
        else:
            marker.update(color=line_color)

        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color=line_color),
            marker=marker,
            opacity=plot.theme.opacity_default,
            hoverinfo="skip",
        )

    def update(self, backend_figure, plot):
        new = self._trace(plot, plot.evaluate())
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y = new.x, new.y
        trace.marker = new.marker


class ArrowField3DArtist(PlotlyArtist):
    """3D arrows via go.Cone - Plotly's native vector-field primitive, so
    (unlike the 2D case) no manual segment-building is needed: Cone colors by
    the norm of (u, v, w) automatically, which is color_by_magnitude's
    continuous case for free."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        trace = cls._trace(plot, plot.evaluate())
        backend_figure.add_trace(trace, row=row, col=col)
        if plot.title is not None:
            backend_figure.update_layout(title=plot.title)
        geometry = plot.data_config.geometry_axes.geometry
        x_name, y_name, z_name = axis_names_from_variable(_spatial_variable(geometry), 3)
        backend_figure.update_scenes(
            row=row, col=col,
            xaxis=dict(title=x_name), yaxis=dict(title=y_name), zaxis=dict(title=z_name),
        )
        return cls(len(backend_figure.data) - 1)

    @staticmethod
    def _trace(plot, result):
        positions = _to_numpy(result.positions)
        vectors = _to_numpy(result.vectors)

        if plot.vector.color_by_magnitude:
            colorscale = plot.vector.cmap or plot.theme.default_cmap
        else:
            color = plot.theme.vector_color
            colorscale = [[0, color], [1, color]]

        return go.Cone(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            u=vectors[:, 0],
            v=vectors[:, 1],
            w=vectors[:, 2],
            anchor="tail",
            sizemode="scaled",
            sizeref=1.0,
            colorscale=colorscale,
            showscale=plot.vector.color_by_magnitude,
            opacity=plot.theme.opacity_default,
        )

    def update(self, backend_figure, plot):
        new = self._trace(plot, plot.evaluate())
        trace = backend_figure.data[self.figure_idx]
        trace.x, trace.y, trace.z = new.x, new.y, new.z
        trace.u, trace.v, trace.w = new.u, new.v, new.w
