"""A renderer that records what it is asked to draw, for tests: it proves
that a renderer needs nothing but Results, cells and its theme."""

import json

from qewton.visualization.plots.result import (
    ArrowResult,
    BarResult,
    GraphLayoutResult,
    HeatmapResult,
    ImageResult,
    LineResult,
    MarkerResult,
    ParallelCoordinatesResult,
    SurfaceResult,
)
from qewton.visualization.renderers.base import Artist, Renderer


class RecordingArtist(Artist):
    """Keeps the Result it was last given and whether it is shown."""

    def __init__(self, result, cell):
        self.result = result
        self.cell = cell
        self.visible = True
        self.updates = 0

    @classmethod
    def create(cls, backend_figure, result, cell, theme):
        artist = cls(result, cell)
        backend_figure["artists"].append(artist)
        return artist

    def update(self, backend_figure, result, cell, theme):
        self.result = result
        self.updates += 1

    def set_visible(self, backend_figure, visible):
        self.visible = visible


class RecordingRenderer(Renderer):
    """Records artists, axes, colorbars and embedded snapshots in a dict."""

    artists = {
        (r, d): RecordingArtist
        for r in (
            HeatmapResult,
            SurfaceResult,
            LineResult,
            MarkerResult,
            BarResult,
            ImageResult,
            ArrowResult,
        )
        for d in (2, 3)
    } | {(ParallelCoordinatesResult, 0): RecordingArtist, (GraphLayoutResult, 2): RecordingArtist}

    def create_backend_figure(self, figure):
        return {"artists": [], "axes": {}, "colorbars": {}, "embedded": {}}

    def draw_axes(self, cell):
        self.backend_figure["axes"][(cell.row, cell.col)] = {
            channel: (cell.axis_title(channel), axis.scale.range)
            for channel, axis in cell.axes.items()
        }

    def draw_colorbar(self, scale, title, cells, visible=True):
        self.backend_figure["colorbars"][id(scale)] = (
            title,
            scale.range,
            [(c.row, c.col) for c in cells],
            visible,
        )

    def clear_embeddings(self):
        self.backend_figure["embedded"] = {}

    def snapshot(self, keys, cells, scales):
        artists = [self.artist_of(key) for key in keys]
        return [(a.result, a.visible) for a in artists]

    def _embed(self, control, snapshots):
        self.backend_figure["embedded"][id(control)] = snapshots

    animate = apply_slider = apply_selector = _embed

    def show(self):
        pass

    def save_html(self, path):
        summary = {
            "artists": len(self.backend_figure["artists"]),
            "embedded": len(self.backend_figure["embedded"]),
        }
        with open(path, "w") as file:
            json.dump(summary, file)
