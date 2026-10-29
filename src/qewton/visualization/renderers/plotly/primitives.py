"""Plotly artists for the primitive Results, one per Result class and
number of position components.

An artist builds a fixed number of traces from a Result, its cell and the
renderer's theme (:meth:`PlotlyArtist.traces`); creating adds them to the
cell, updating overwrites them in place. Where Plotly draws one color per 2D trace
(filled triangles, lines), colored elements are grouped into
:data:`~qewton.visualization.renderers.plotly.traces.N_BINS` value bins
of one trace each. Colorbars are drawn by the renderer, never by an
artist.
"""

import numpy as np
from plotly import graph_objects as go

from qewton.visualization.renderers.base import Artist
from qewton.visualization.scales import log_ticks
from qewton.visualization.renderers.plotly.traces import (
    N_BINS,
    bins,
    color_range,
    colorscale,
    segment_coordinates,
    triangle_coordinates,
)


class PlotlyArtist(Artist):
    """Base class of the Plotly artists: owns a fixed list of traces.

    Args:
        trace_indices: Indices of its traces in ``backend_figure.data``.

    Attributes:
        trace_indices (list of int): Indices of its traces.

    Notes:
        Subclasses implement :meth:`traces`, returning the same number of
        traces for every state.
    """

    def __init__(self, trace_indices: list[int]) -> None:
        self.trace_indices = trace_indices

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """The traces drawing `result` in `cell`.

        Args:
            result (Result): The Result to draw.
            cell (Cell): The cell drawn into.
            theme (Theme): The renderer's theme.

        Returns:
            list: Plotly traces.
        """
        raise NotImplementedError

    @classmethod
    def create(cls, backend_figure, result, cell, theme) -> "PlotlyArtist":
        """See :meth:`~qewton.visualization.renderers.base.Artist.create`."""
        indices = []
        for trace in cls.traces(result, cell, theme):
            indices.append(len(backend_figure.data))
            backend_figure.add_trace(trace, row=cell.row, col=cell.col)
        return cls(indices)

    def update(self, backend_figure, result, cell, theme) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Artist.update`."""
        for index, trace in zip(self.trace_indices, self.traces(result, cell, theme)):
            backend_figure.data[index].update(_without_axes(trace), overwrite=True)

    def set_visible(self, backend_figure, visible: bool) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Artist.set_visible`."""
        for index in self.trace_indices:
            backend_figure.data[index].visible = visible


def _without_axes(trace) -> dict:
    # A trace's properties except its subplot binding, which stays as set
    # when it was added.
    properties = trace.to_plotly_json()
    for key in ("type", "xaxis", "yaxis", "scene"):
        properties.pop(key, None)
    return properties


def _legend(result) -> dict:
    return dict(name=result.label, showlegend=result.label is not None)


def _hover_values(result, channels: str) -> dict:
    # Hover text with the values themselves where a log color scale maps
    # their log10 to colors.
    if result.color_scale is None or not result.color_scale.log:
        return {}
    position = ", ".join(f"%{{{c}}}" for c in channels)
    return dict(
        customdata=result.values,
        hovertemplate=f"({position}): %{{customdata:.4g}}<extra></extra>",
    )


def _mapped_colors(result, theme) -> dict:
    # Marker or line color settings: a uniform color, or values mapped
    # through the result's color scale (no colorbar of its own).
    if result.color_scale is None:
        return dict(color=theme.get_color(result.color_index))
    lo, hi = color_range(result.color_scale)
    return dict(
        color=result.color_scale.transform(result.values),
        colorscale=colorscale(result.color_scale, theme),
        cmin=lo,
        cmax=hi,
        showscale=False,
    )


class HeatmapArtist(PlotlyArtist):
    """Draws a HeatmapResult as a ``go.Heatmap``."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        lo, hi = color_range(result.color_scale)
        return [
            go.Heatmap(
                x=result.x,
                y=result.y,
                z=result.color_scale.transform(result.values),
                colorscale=colorscale(result.color_scale, theme),
                zmin=lo,
                zmax=hi,
                showscale=False,
                opacity=result.opacity,
                **_hover_values(result, "xy"),
                **_legend(result),
            )
        ]


class FlatMeshArtist(PlotlyArtist):
    """Draws a SurfaceResult in 2D as filled triangles: one trace per value
    bin (one trace if uncolored), plus one for the edges if shown."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        if result.values is None:
            groups = [(result.cells, theme.get_color(result.color_index))]
        else:
            cell_values = result.values[result.cells].mean(axis=1)
            index, colors = bins(cell_values, result.color_scale, theme)
            groups = [(result.cells[index == i], colors[i]) for i in range(N_BINS)]
        traces = [
            cls._fill(result, cells, color, first=i == 0)
            for i, (cells, color) in enumerate(groups)
        ]
        if result.show_edges:
            x, y = segment_coordinates(result.positions, _edges(result.cells))
            line = dict(color=theme.line_color, width=0.5)
            traces.append(
                go.Scatter(
                    x=x, y=y, mode="lines", line=line, hoverinfo="skip", showlegend=False
                )
            )
        return traces

    @staticmethod
    def _fill(result, cells, color, first: bool) -> go.Scatter:
        x, y = triangle_coordinates(result.positions, cells)
        legend = _legend(result) if first else dict(showlegend=False)
        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="toself",
            fillcolor=color,
            line=dict(color=color, width=0.5),
            opacity=result.opacity,
            hoverinfo="skip",
            **legend,
        )


def _edges(cells: np.ndarray) -> np.ndarray:
    # The edges of triangles as point index pairs.
    return np.concatenate([cells[:, [a, b]] for a, b in ((0, 1), (1, 2), (2, 0))])


class SurfaceArtist(PlotlyArtist):
    """Draws a SurfaceResult in 3D as a ``go.Mesh3d``, plus a line trace
    for the edges if shown."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        x, y, z = result.positions.T
        i, j, k = result.cells.T
        if result.values is None:
            colors = dict(color=theme.get_color(result.color_index))
        else:
            lo, hi = color_range(result.color_scale)
            colors = dict(
                intensity=result.color_scale.transform(result.values),
                colorscale=colorscale(result.color_scale, theme),
                cmin=lo,
                cmax=hi,
                showscale=False,
            )
        traces = [
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                opacity=result.opacity,
                **colors,
                **_hover_values(result, "xyz"),
                **_legend(result),
            )
        ]
        if result.show_edges:
            ex, ey, ez = segment_coordinates(result.positions, _edges(result.cells))
            line = dict(color=theme.line_color, width=1)
            traces.append(
                go.Scatter3d(
                    x=ex, y=ey, z=ez, mode="lines", line=line, hoverinfo="skip",
                    showlegend=False,
                )
            )
        return traces


class Line2DArtist(PlotlyArtist):
    """Draws a LineResult in 2D: one line trace, or one per value bin if
    colored (a segment takes the mean value of its end points)."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        width = result.width if result.width is not None else theme.line_width
        if result.values is None:
            groups = [(result.segments, theme.get_color(result.color_index))]
        else:
            segment_values = result.values[result.segments].mean(axis=1)
            index, colors = bins(segment_values, result.color_scale, theme)
            groups = [(result.segments[index == i], colors[i]) for i in range(N_BINS)]
        traces = []
        for n, (segments, color) in enumerate(groups):
            x, y = segment_coordinates(result.positions, segments)
            legend = _legend(result) if n == 0 else dict(showlegend=False)
            line = dict(color=color, width=width, dash=result.dash)
            traces.append(
                go.Scatter(
                    x=x, y=y, mode="lines", line=line, opacity=result.opacity, **legend
                )
            )
        return traces


class Line3DArtist(PlotlyArtist):
    """Draws a LineResult in 3D as a ``go.Scatter3d`` line, colored along
    its points if colored."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        x, y, z = segment_coordinates(result.positions, result.segments)
        line = dict(
            width=result.width if result.width is not None else theme.line_width * 2
        )
        if result.color_scale is None:
            line.update(color=theme.get_color(result.color_index))
        else:
            per_point = np.concatenate(
                [
                    result.values[result.segments],
                    np.full((len(result.segments), 1), np.nan),
                ],
                axis=1,
            ).ravel()
            lo, hi = color_range(result.color_scale)
            line.update(
                color=result.color_scale.transform(per_point),
                colorscale=colorscale(result.color_scale, theme),
                cmin=lo,
                cmax=hi,
            )
        return [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=line,
                opacity=result.opacity,
                **_legend(result),
            )
        ]


class Marker2DArtist(PlotlyArtist):
    """Draws a MarkerResult in 2D as a ``go.Scatter`` of markers."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        x, y = result.positions.T
        marker = dict(size=result.size or theme.marker_size, symbol=result.symbol)
        marker.update(_mapped_colors(result, theme))
        return [
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=marker,
                opacity=result.opacity,
                **_legend(result),
            )
        ]


class Marker3DArtist(PlotlyArtist):
    """Draws a MarkerResult in 3D as a ``go.Scatter3d`` of markers."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        x, y, z = result.positions.T
        marker = dict(size=result.size or theme.marker_size, symbol=result.symbol)
        marker.update(_mapped_colors(result, theme))
        return [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=marker,
                opacity=result.opacity,
                **_legend(result),
            )
        ]


class BarArtist(PlotlyArtist):
    """Draws a BarResult as a ``go.Bar``."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        x, y = result.positions.T
        return [
            go.Bar(
                x=x,
                y=y,
                marker=_mapped_colors(result, theme),
                opacity=result.opacity,
                **_legend(result),
            )
        ]


class ImageArtist(PlotlyArtist):
    """Draws an ImageResult as a ``go.Image``."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`.

        Raises:
            ValueError: If the grid spacing is not uniform, which a Plotly
                image needs.
        """
        x0, dx = _uniform(result.x, "x")
        y0, dy = _uniform(result.y, "y")
        z = result.colors.copy()
        z[..., :3] *= 255  # Plotly takes RGB in [0, 255], alpha in [0, 1]
        return [
            go.Image(
                z=z,
                x0=x0,
                dx=dx,
                y0=y0,
                dy=dy,
                colormodel="rgba" if z.shape[-1] == 4 else "rgb",
                opacity=result.opacity,
                hoverinfo="x+y",
                name=result.label,  # an image has no legend entry
            )
        ]


def _uniform(coordinates: np.ndarray, name: str) -> tuple[float, float]:
    # Start and step of uniformly spaced coordinates.
    if len(coordinates) == 1:
        return float(coordinates[0]), 1.0
    steps = np.diff(coordinates)
    if not np.allclose(steps, steps[0]):
        raise ValueError(f"Plotly draws images on grids with uniform {name} spacing only.")
    return float(coordinates[0]), float(steps[0])


def _arrow_groups(result, theme) -> list[tuple[np.ndarray, str]]:
    # (arrow mask, color) per trace: one uniform group, or one per value bin.
    if result.values is None:
        return [(np.ones(len(result.positions), dtype=bool), theme.get_color(result.color_index))]
    index, colors = bins(result.values, result.color_scale, theme)
    return [(index == i, colors[i]) for i in range(N_BINS)]


class Arrow2DArtist(PlotlyArtist):
    """Draws an ArrowResult in 2D: per value bin (one if uncolored) a line
    trace of the shafts with an arrow head marker at each tip."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        traces = []
        for n, (mask, color) in enumerate(_arrow_groups(result, theme)):
            tails = result.positions[mask]
            tips = tails + result.vectors[mask]
            ends = np.stack([tails, tips], axis=1).reshape(-1, 2)
            segments = np.arange(len(ends)).reshape(-1, 2)
            x, y = segment_coordinates(ends, segments)
            legend = _legend(result) if n == 0 else dict(showlegend=False)
            marker = dict(
                symbol="arrow",
                angleref="previous",
                size=np.tile([0, theme.marker_size, 0], len(tails)),
                color=color,
            )
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line=dict(color=color, width=theme.line_width),
                    marker=marker,
                    opacity=result.opacity,
                    hoverinfo="skip",
                    **legend,
                )
            )
        return traces


class Arrow3DArtist(PlotlyArtist):
    """Draws an ArrowResult in 3D: per value bin (one if uncolored) a
    ``go.Cone`` trace whose cones are as long as their arrows."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        traces = []
        for n, (mask, color) in enumerate(_arrow_groups(result, theme)):
            x, y, z = result.positions[mask].T
            u, v, w = result.vectors[mask].T
            legend = _legend(result) if n == 0 else dict(showlegend=False)
            traces.append(
                go.Cone(
                    x=x,
                    y=y,
                    z=z,
                    u=u,
                    v=v,
                    w=w,
                    anchor="tail",
                    sizemode="raw",
                    sizeref=1.0,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=result.opacity,
                    hoverinfo="skip",
                    **legend,
                )
            )
        return traces


class ParallelCoordinatesArtist(PlotlyArtist):
    """Draws a ParallelCoordinatesResult as a ``go.Parcoords``, with its
    own colorbar: Plotly lays out the axes of parallel coordinates itself,
    so a colorbar beside a cell does not apply."""

    @classmethod
    def traces(cls, result, cell, theme) -> list:
        """See :meth:`PlotlyArtist.traces`."""
        dimensions = []
        for dimension in result.dimensions:
            settings = dict(
                label=dimension.title, values=dimension.values, range=list(dimension.range)
            )
            if dimension.ticks is not None:
                positions, labels = dimension.ticks
                settings.update(tickvals=list(positions), ticktext=list(labels))
            dimensions.append(settings)
        return [go.Parcoords(dimensions=dimensions, line=cls._line(result, theme))]

    @staticmethod
    def _line(result, theme) -> dict:
        scale = result.color_scale
        if scale is None:
            return dict(color=theme.get_color(result.color_index))
        lo, hi = color_range(scale)
        colorbar = dict(title=dict(text=result.color_title))
        if scale.log:
            tickvals, ticktext = log_ticks(lo, hi)
            colorbar.update(tickvals=tickvals, ticktext=ticktext)
        return dict(
            color=scale.transform(result.values),
            colorscale=colorscale(scale, theme),
            cmin=lo,
            cmax=hi,
            showscale=True,
            colorbar=colorbar,
        )
