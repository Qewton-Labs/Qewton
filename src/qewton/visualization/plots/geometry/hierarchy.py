"""Hierarchy: the levels of a mesh hierarchy, one at a time."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from qewton.visualization.axes import POSITION_CHANNELS, ControlAxis, SliderAxis
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.result import LineResult, MarkerResult, Result
from qewton.visualization.scales import ColorScale, Scale

if TYPE_CHECKING:
    from qewton.geometries.discrete.meshes.hierarchy import Level


class HierarchyPlot(Plot):
    """The levels of a mesh hierarchy (see
    :meth:`~qewton.geometries.discrete.meshes.mesh.Mesh.hierarchy`), one at a
    time: the edges and vertices of a level, and behind them the vertices of
    the level before, colored by the vertex of this level they belong to.
    Neighbouring aggregates get different colors.

    Args:
        levels: The levels, e.g. ``mesh.hierarchy(4)``, with 2 or 3
            coordinates.
        control: Chooses the drawn level: a SliderAxis named ``"level"`` if
            None, a TimeAxis to play the levels, a FacetAxis to draw them side
            by side or a FixedAxis for one of them.
        title: Heading of the plot's cell.

    Raises:
        ValueError: Without levels, if the vertices do not have 2 or 3
            coordinates, or if `control` draws every level at once (an
            OverlayAxis).

    Attributes:
        levels (list of Level): The drawn levels.
        control (ControlAxis): The control over the levels.
    """

    def __init__(
        self,
        levels: list[Level],
        control: ControlAxis | None = None,
        *,
        title: str | None = None,
    ) -> None:
        control = control if control is not None else SliderAxis("level")
        if control.copies:
            raise ValueError(
                f"A HierarchyPlot draws one level at a time, not every level at once "
                f"({type(control).__name__})."
            )
        if not levels:
            raise ValueError("A HierarchyPlot needs at least one level.")
        super().__init__(title=title, controls=[control])
        self.levels = list(levels)
        self.control = control
        self._positions = [
            np.asarray(level.backend.to_numpy(level.vertices), dtype=float)
            for level in self.levels
        ]
        self._edges = [np.asarray(level.backend.to_numpy(level.edges)) for level in self.levels]
        self._dim = self._positions[0].shape[1]
        if self._dim not in (2, 3):
            raise ValueError(
                f"HierarchyPlot draws vertices with 2 or 3 coordinates, not {self._dim}."
            )
        # The color of each vertex of the level before, by its aggregate.
        self._aggregate_colors = [np.empty(0, dtype=int)] + [
            greedy_coloring(self._edges[k], len(self._positions[k]))[
                np.asarray(self.levels[k - 1].backend.to_numpy(self.levels[k - 1].cluster))
            ]
            for k in range(1, len(self.levels))
        ]
        self._aggregate_scale = ColorScale(categorical=True)
        control.resolve(
            len(self.levels),
            labels=[f"{k} ({len(p)} vertices)" for k, p in enumerate(self._positions)],
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def scales(self) -> list[Scale]:
        # The colors of the aggregates need no colorbar.
        return [self._aggregate_scale]

    @property
    def n_copies(self) -> int:
        # The vertices of the level before, the edges and the vertices.
        return 3

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        return POSITION_CHANNELS[: self.dim]

    def train_scales(self, axes: dict[str, Scale]) -> None:
        """See :meth:`~qewton.visualization.plots.base.Plot.train_scales`."""
        for k in self.control.trained_states:
            for positions in self._positions[max(k - 1, 0) : k + 1]:
                for i, channel in enumerate(POSITION_CHANNELS[: self.dim]):
                    axes[channel].train(positions[:, i])
            if k > 0:
                self._aggregate_scale.train(self._aggregate_colors[k])

    def evaluate(self) -> list[Result]:
        """See :meth:`~qewton.visualization.plots.base.Plot.evaluate`: the
        vertices of the level before (none on the first), then the edges and
        the vertices of the current level."""
        k = self.control.state
        before = self._positions[k - 1] if k > 0 else np.empty((0, self.dim))
        return [
            MarkerResult(
                positions=before,
                values=self._aggregate_colors[k],
                color_scale=self._aggregate_scale,
                size=marker_size(len(before), largest=4),
                opacity=0.8,
            ),
            LineResult(
                positions=self._positions[k],
                segments=self._edges[k],
                color_index=None,
                width=1,
                opacity=0.5,
            ),
            MarkerResult(
                positions=self._positions[k],
                color_index=None,
                size=marker_size(len(self._positions[k]), largest=6),
            ),
        ]


def marker_size(count: int, largest: float) -> float:
    """A marker size for `count` vertices, so that dense levels do not hide
    their edges: about a third of their spacing, between 2 and `largest`."""
    return float(np.clip(120 / np.sqrt(max(count, 1)), 2, largest))


def greedy_coloring(edges: np.ndarray, vertex_count: int) -> np.ndarray:
    """A color per vertex of a graph, so that neighbours differ: by
    increasing index, every vertex takes the smallest color none of its
    neighbours has.

    Args:
        edges: The undirected edges, shape [E, 2].
        vertex_count: The number of vertices.

    Returns:
        numpy.ndarray: The color of each vertex, integers from 0.
    """
    neighbours = [[] for _ in range(vertex_count)]
    for a, b in edges.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    colors = [-1] * vertex_count
    for v in range(vertex_count):
        taken = {colors[u] for u in neighbours[v]}
        colors[v] = next(c for c in itertools.count() if c not in taken)
    return np.asarray(colors, dtype=int)
