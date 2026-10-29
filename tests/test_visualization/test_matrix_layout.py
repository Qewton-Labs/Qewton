"""Test matrix of layouts: arrangement × renderer.

Every arrangement of plots (single plots, overlays, rows, columns, nested
and irregular grids, facets) states the grid it gives, the cells that hold
plots, their titles and the number of drawn artists. Colorbars sit right
of the rightmost cell of their scale. auto_layout places plots in one
Overlay or side by side, by a table of pairs.
"""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.visualization import (
    ColorScale,
    Column,
    FacetAxis,
    Figure,
    FixedAxis,
    HeatmapPlot,
    LinePlot,
    Overlay,
    PositionAxis,
    Row,
    ScatterPlot,
    SurfacePlot,
    VectorPlot,
    auto_layout,
)
from qewton.visualization.renderers import PlotlyRenderer

from recording_renderer import RecordingRenderer

T, X = Variable("t", 1), Variable("x", 2)
U, P = Variable("u", 1), Variable("p", 1)


def _line(title=None, controls=()):
    batch = BatchAxes(3)
    config = DataConfiguration(
        batch, GeometryAxes(GridGeometry.build_unit_grid(T, 5)), FeatureAxes(U * P)
    )
    data = np.random.default_rng(0).random(config.shape)
    control = [c(batch) for c in controls]
    return LinePlot(data, config, x=T, y=U, title=title, controls=control or [_fixed(batch)])


def _fixed(batch):
    return FixedAxis(batch, init_state=0, title="run")


def _heatmap(title=None, scale=None):
    config = DataConfiguration(GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U))
    data = np.random.default_rng(0).random(config.shape)
    return HeatmapPlot(data, config, position=X, color=U, color_scale=scale, title=title)


#: Arrangement: (layout factory, grid shape, cells with plots, artists drawn,
#: cell titles in reading order).
LAYOUTS = {
    "a plot": (lambda: _heatmap("a"), (1, 1), 1, 1, ["a"]),
    "an overlay": (lambda: Overlay(_line("a"), _line("b")), (1, 1), 1, 2, ["a / b, run = 0"]),
    "a row of three": (
        lambda: Row(_heatmap("a"), _heatmap("b"), _heatmap("c")), (1, 3), 3, 3, ["a", "b", "c"]
    ),
    "a column of two": (lambda: Column(_heatmap("a"), _heatmap("b")), (2, 1), 2, 2, ["a", "b"]),
    "a column of rows": (
        lambda: Column(Row(_heatmap("a"), _heatmap("b")), Row(_heatmap("c"), _heatmap("d"))),
        (2, 2),
        4,
        4,
        ["a", "b", "c", "d"],
    ),
    "a narrower row padded": (
        lambda: Column(Row(_heatmap("a"), _heatmap("b")), _heatmap("c")),
        (2, 2),
        3,
        3,
        ["a", "b", "c"],
    ),
    "nested rows and columns": (
        lambda: Row(Column(_heatmap("a"), _heatmap("b")), _heatmap("c")),
        (2, 2),
        3,
        3,
        ["a", "c", "b"],
    ),
    "a facet": (
        lambda: _line("a", controls=[FacetAxis]),
        (1, 3),
        3,
        3,
        ["a, batch = 0", "a, batch = 1", "a, batch = 2"],
    ),
}

RENDERERS = {"plotly": PlotlyRenderer, "recording": RecordingRenderer}


@pytest.mark.parametrize("renderer_name", list(RENDERERS))
@pytest.mark.parametrize("name", list(LAYOUTS))
def test_layout(name, renderer_name):
    make, shape, n_cells, n_artists, titles = LAYOUTS[name]
    figure = Figure(make(), renderer=RENDERERS[renderer_name](), title="figure")
    assert figure.grid_shape == shape
    assert len(figure.cells) == n_cells
    assert [cell.title for cell in figure.cells] == titles
    backend = figure.draw()
    if renderer_name == "recording":
        assert len(backend["artists"]) == n_artists
    else:
        assert backend.layout.title.text == "figure"
        drawn_titles = [a.text for a in backend.layout.annotations]
        assert drawn_titles == titles


def test_plots_get_distinct_colors_in_reading_order():
    plots = [_line(), _line(), _line()]
    figure = Figure(Overlay(plots[2], *plots[:2]))
    assert [p.color_index for p in figure.plots] == [0, 1, 2]


def test_a_figure_without_a_title_shows_none():
    backend = Figure(_heatmap("a")).draw()
    assert backend.layout.title.text is None


#: Arrangement of three heatmaps with scales: which cells' colorbars.
COLORBARS = {
    "own scales": ([None, None, None], 3),
    "first two share": (["shared", "shared", None], 2),
    "all share": (["shared"] * 3, 1),
}


@pytest.mark.parametrize("name", list(COLORBARS))
def test_colorbars_sit_right_of_the_last_cell_of_their_scale(name):
    names, n_colorbars = COLORBARS[name]
    shared = ColorScale()
    plots = [_heatmap(scale=shared if n == "shared" else None) for n in names]
    backend = Figure(Row(*plots)).draw()
    carriers = [t for t in backend.data if t.uid and t.uid.startswith("colorbar")]
    assert len(carriers) == n_colorbars
    positions = sorted(t.marker.colorbar.x for t in carriers)
    assert len(set(positions)) == n_colorbars
    cell_right_edges = [backend.get_subplot(1, c).xaxis.domain[1] for c in (1, 2, 3)]
    rightmost = {"own scales": [0, 1, 2], "first two share": [1, 2], "all share": [2]}[name]
    for position, column in zip(positions, rightmost):
        assert position > cell_right_edges[column]
        if column < 2:
            assert position < backend.get_subplot(1, column + 2).xaxis.domain[0]


# --- auto_layout ---------------------------------------------------------------


def _field(kind):
    # A plot of u, p or the vector r on one 2D grid.
    R = Variable("r", 2)
    config = DataConfiguration(GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U * P * R))
    data = np.random.default_rng(0).random(config.shape)
    if kind == "heatmap":
        return HeatmapPlot(data, config, position=X, color=U)
    if kind == "arrows":
        return VectorPlot(data, config, position=X, vector=R)
    if kind == "colored_points":
        return ScatterPlot(data, config, position=X, color=P)
    return ScatterPlot(data, config, position=X)


def _curve(y, over=T):
    config = DataConfiguration(GeometryAxes(GridGeometry.build_unit_grid(over, 5)), FeatureAxes(U * P))
    data = np.random.default_rng(0).random(config.shape)
    return LinePlot(data, config, x=over, y=y)


def _surface_3d():
    Y = Variable("y", 3)
    config = DataConfiguration(GeometryAxes(Box(Y, [0.0] * 3, 1, 1, 1).create_mesh(0.6)), FeatureAxes(U))
    return SurfacePlot(np.zeros(config.shape), config, position=Y, color=U)


#: Plots placed together: (factory, Overlay or Row, drawing order in an
#: Overlay as indices into the given plots).
AUTO_LAYOUTS = {
    "arrows over a heatmap": (lambda: [_field("arrows"), _field("heatmap")], Overlay, [1, 0]),
    "points over a heatmap": (lambda: [_field("points"), _field("heatmap")], Overlay, [1, 0]),
    "two heatmaps": (lambda: [_field("heatmap"), _field("heatmap")], Row, None),
    "two sets of colored points": (
        lambda: [_field("colored_points"), _field("colored_points")], Row, None
    ),
    "one quantity from two sources": (lambda: [_curve(U), _curve(U)], Overlay, [0, 1]),
    "two quantities": (lambda: [_curve(U), _curve(P)], Row, None),
    "two different x axes": (lambda: [_curve(U), _curve(U, Variable("m", 1))], Row, None),
    "2d and 3d": (lambda: [_field("heatmap"), _surface_3d()], Row, None),
    "two vector fields": (lambda: [_field("arrows"), _field("arrows")], Row, None),
    "arrows and colored points": (
        lambda: [_field("arrows"), _field("colored_points")], Overlay, [1, 0]
    ),
}


@pytest.mark.parametrize("name", list(AUTO_LAYOUTS))
def test_auto_layout(name):
    make, kind, order = AUTO_LAYOUTS[name]
    plots = make()
    layout = auto_layout(plots)
    assert type(layout) is kind
    if order is not None:
        assert layout.plots == [plots[i] for i in order]


# --- linked cells -------------------------------------------------------------


def test_side_by_side_plots_of_one_domain_share_their_axes():
    layout = auto_layout([_field("heatmap"), _field("heatmap")])
    first, second = (Overlay is type(o) for o in layout.items)
    assert first and second
    assert layout.items[0].axes["x"] is layout.items[1].axes["x"]


def test_cells_sharing_an_axis_zoom_together():
    shared = dict(x=PositionAxis(), y=PositionAxis())
    layout = Row(Overlay(_field("heatmap"), **shared), Overlay(_field("heatmap"), **shared))
    backend = Figure(layout, renderer=PlotlyRenderer()).draw()
    assert backend.layout.xaxis2.matches == "x" and backend.layout.yaxis2.matches == "y"
    # The first cell is drawn to scale; Plotly forbids matches with scaleanchor.
    assert backend.layout.yaxis.scaleanchor == "x"
    assert backend.layout.yaxis2.scaleanchor is None


def test_facet_cells_zoom_together():
    backend = Figure(_line(controls=[FacetAxis]), renderer=PlotlyRenderer()).draw()
    assert backend.layout.xaxis2.matches == "x" and backend.layout.xaxis3.matches == "x"


def test_cells_with_axes_of_their_own_zoom_alone():
    backend = Figure(Row(_field("heatmap"), _field("heatmap")), renderer=PlotlyRenderer()).draw()
    assert backend.layout.xaxis2.matches is None


def test_3d_cells_sharing_their_axes_move_their_cameras_together():
    shared = dict(x=PositionAxis(), y=PositionAxis(), z=PositionAxis())
    layout = Row(*(Overlay(_surface_3d(), **shared) for _ in range(2)), _surface_3d())
    figure = Figure(layout, renderer=PlotlyRenderer())
    figure.draw()
    assert figure.renderer.linked_scenes() == [["scene", "scene2"]]
    page = figure.renderer.html(full_html=False)
    assert "plotly_relayout" in page and '"scene2"' in page
