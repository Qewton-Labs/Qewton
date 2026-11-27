"""HierarchyPlot: the levels of a mesh hierarchy, one at a time."""

import numpy as np
import pytest

from qewton.backends.numpy.base import NumPyBackend
from qewton.config.devices import cuda, cuda_available
from qewton.geometries.discrete.meshes import Mesh, build_hierarchy
from qewton.visualization import (
    FacetAxis,
    Figure,
    FixedAxis,
    HierarchyPlot,
    OverlayAxis,
    SliderAxis,
    TimeAxis,
)
from qewton.visualization.plots.geometry.hierarchy import greedy_coloring
from qewton.visualization.plots.result import LineResult, MarkerResult
from qewton.visualization.renderers import MatplotlibRenderer, PlotlyRenderer
from qewton.visualization.themes import LIGHT_THEME

from recording_renderer import RecordingRenderer


def _unit_square(n: int = 8, **kwargs) -> Mesh:
    xs = np.linspace(0.0, 1.0, n + 1)
    vertices = [[x, y] for y in xs for x in xs]
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            cells += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return Mesh(vertices, cells, **kwargs)


def _levels(n_levels: int = 3):
    return _unit_square(backend=NumPyBackend).hierarchy(n_levels)


def _numpy(level, name):
    return np.asarray(level.backend.to_numpy(getattr(level, name)))


class TestLevels:
    def test_a_slider_over_the_levels_by_default(self):
        levels = _levels()
        plot = HierarchyPlot(levels)
        (control,) = plot.controls
        assert isinstance(control, SliderAxis) and control is plot.control
        assert control.name == "level"
        assert control.states == [0, 1, 2]
        assert control.label(1) == f"1 ({levels[1].vertex_count} vertices)"

    def test_a_level_is_drawn_over_the_vertices_of_the_level_before(self):
        levels = _levels()
        plot = HierarchyPlot(levels)
        plot.control.state = 1
        before, edges, vertices = plot.evaluate()
        assert isinstance(before, MarkerResult) and isinstance(vertices, MarkerResult)
        assert isinstance(edges, LineResult)
        np.testing.assert_allclose(before.positions, _numpy(levels[0], "vertices"))
        np.testing.assert_array_equal(edges.segments, _numpy(levels[1], "edges"))
        np.testing.assert_allclose(vertices.positions, _numpy(levels[1], "vertices"))
        # Edges and vertices are no data: the theme's line color.
        assert edges.color_index is None and vertices.color_index is None
        assert before.color_scale is not None

    def test_the_first_level_has_nothing_behind_it(self):
        plot = HierarchyPlot(_levels())
        before, _, vertices = plot.evaluate()
        assert before.positions.shape == (0, 2) and len(before.values) == 0
        assert len(vertices.positions) == _levels()[0].vertex_count
        for state in plot.control.states:
            plot.control.state = state
            assert len(plot.evaluate()) == plot.n_copies

    def test_neighbouring_aggregates_get_different_colors(self):
        levels = _levels()
        plot = HierarchyPlot(levels)
        plot.control.state = 1
        colors = plot.evaluate()[0].values
        cluster = _numpy(levels[0], "cluster")
        for aggregate in np.unique(cluster):
            assert len(np.unique(colors[cluster == aggregate])) == 1
        for a, b in _numpy(levels[0], "edges"):
            if cluster[a] != cluster[b]:
                assert colors[a] != colors[b]

    def test_greedy_coloring(self):
        triangle = np.array([[0, 1], [1, 2], [0, 2]])
        assert sorted(greedy_coloring(triangle, 3).tolist()) == [0, 1, 2]
        path = np.array([[0, 1], [1, 2], [2, 3]])
        assert greedy_coloring(path, 4).tolist() == [0, 1, 0, 1]
        assert greedy_coloring(np.empty((0, 2), dtype=int), 2).tolist() == [0, 0]

    @pytest.mark.skipif(not cuda_available(), reason="needs a GPU")
    def test_levels_on_a_gpu(self):
        from qewton.backends.torch.base import TorchBackend

        levels = _unit_square(backend=TorchBackend, device=cuda(0)).hierarchy(3)
        plot = HierarchyPlot(levels)
        plot.control.state = 2
        assert len(plot.evaluate()[2].positions) == levels[2].vertex_count


class TestControls:
    def test_a_facet_axis_draws_every_level_side_by_side(self):
        figure = Figure(HierarchyPlot(_levels(), FacetAxis("level")), renderer=RecordingRenderer())
        figure.draw()
        assert len(figure.cells) == 3
        assert figure.cells[2].title == f"level = 2 ({_levels()[2].vertex_count} vertices)"

    def test_a_time_axis_plays_the_levels(self):
        time = TimeAxis("level")
        HierarchyPlot(_levels(), time)
        assert time.states == [0, 1, 2]

    def test_a_fixed_axis_draws_one_level(self):
        levels = _levels()
        plot = HierarchyPlot(levels, FixedAxis("level", init_state=2))
        assert len(plot.evaluate()[2].positions) == levels[2].vertex_count

    def test_every_level_at_once_raises(self):
        with pytest.raises(ValueError, match="one level at a time"):
            HierarchyPlot(_levels(), OverlayAxis("level"))

    def test_no_levels_raise(self):
        with pytest.raises(ValueError, match="at least one level"):
            HierarchyPlot([])

    def test_vertices_with_one_coordinate_raise(self):
        vertices = NumPyBackend.build_tensor([[0.0], [1.0], [2.0]])
        edges = NumPyBackend.build_tensor([[0, 1], [1, 2]])
        with pytest.raises(ValueError, match="2 or 3 coordinates"):
            HierarchyPlot(build_hierarchy(vertices, edges, 2, NumPyBackend))


class TestDrawing:
    def test_the_axes_are_trained_on_every_level(self):
        figure = Figure(HierarchyPlot(_levels()), renderer=RecordingRenderer())
        backend = figure.draw()
        (axes,) = backend["axes"].values()
        assert axes["x"][1] == (0.0, 1.0) and axes["y"][1] == (0.0, 1.0)

    def test_drawn_with_plotly_with_a_slider_and_without_a_colorbar(self):
        figure = Figure(HierarchyPlot(_levels()), renderer=PlotlyRenderer())
        backend = figure.draw(embedded=figure.controls)
        assert len(backend.layout.sliders) == 1
        assert len(backend.layout.sliders[0].steps) == 3
        assert not any(getattr(t, "showscale", False) for t in backend.data)

    def test_drawn_with_matplotlib(self):
        figure = Figure(HierarchyPlot(_levels(), FacetAxis("level")), renderer=MatplotlibRenderer())
        figure.draw()

    def test_marks_without_a_palette_color_take_the_line_color(self):
        assert LIGHT_THEME.get_color(None) == LIGHT_THEME.line_color
