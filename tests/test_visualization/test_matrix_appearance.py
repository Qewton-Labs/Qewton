"""Test matrix of the appearance of figures: case × renderer.

A case is a typical figure: heatmaps on square and wide domains, alone and
in grids, with shared and separate color scales, facets, overlays, lines,
bars, scatters, vectors, 3D surfaces, table plots and Graph.visualize.
Every case draws with every renderer. RANGES gives per case how the drawn
position ranges end: exactly at the data (rasters and surfaces end at
their domain) or with a margin (marks); bars hold their baseline 0.
LEGENDS gives the legends of every case: entries only for marks of a
uniform color, a legend only for two entries or more. SIZES gives the size of the plot areas: a fixed aspect fits into the
theme's plot box, anything else fills it. SLOTS gives the cell and place of every colorbar: beside the last cell of
its scale, one place per plot. THEME_RULES gives what every switch of the theme does in each renderer,
and the font sizes every text is drawn in.
"""

import dataclasses

import numpy as np
import pytest
import torch

from qewton.algorithms.dl_models.fcn import FCN
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.data.dataloaders.sampler.grid_sampler import GridSampler
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.graphs.graphs import Graph
from qewton.visualization import (
    BarPlot,
    DashApplication,
    ParallelCoordinatesPlot,
    WidgetApplication,
    ColorScale,
    Column,
    FacetAxis,
    Figure,
    HeatmapPlot,
    ImagePlot,
    LinePlot,
    Overlay,
    OverlayAxis,
    Row,
    ScatterPlot,
    Selector,
    SurfacePlot,
    TableHeatMap,
    TableScatter,
    VectorPlot,
)
from qewton.visualization.renderers import MatplotlibRenderer, PlotlyRenderer
from qewton.visualization.renderers.matplotlib.primitives import axes_of, colorbar_axes
from qewton.visualization.renderers.matplotlib.style import PIXELS_PER_INCH, points
from qewton.visualization.themes import DARK_THEME, LIGHT_THEME

T, X, Y = Variable("t", 1), Variable("x", 2), Variable("y", 3)
U, P, V, C = Variable("u", 1), Variable("p", 1), Variable("v", 2), Variable("c", 3)

RENDERERS = {"plotly": PlotlyRenderer, "matplotlib": MatplotlibRenderer}


def _field(grid=None, extra=None):
    # u, p, v and RGB colors c on a 2D grid (default: the unit square).
    grid = grid or GridGeometry.build_unit_grid(X, 8)
    config = DataConfiguration(
        *([extra] if extra else []), GeometryAxes(grid), FeatureAxes(U * P * V * C)
    )
    points = np.asarray(grid.discretization_points)
    x, y = points[..., 0], points[..., 1]
    values = np.stack(
        [np.sin(2 * np.pi * x) * np.cos(np.pi * y), 1 + 99 * x * y, -y, x, x, y, 0 * x + 0.5],
        axis=-1,
    )
    shape = config.shape[1:] if extra else config.shape
    data = np.broadcast_to(values.reshape(shape), config.shape).astype(np.float32)
    return data, config


def _wide():
    # A rectilinear grid on [0, 2] x [0, 0.5].
    xs, ys = np.meshgrid(np.linspace(0, 2, 16), np.linspace(0, 0.5, 5))
    return GridGeometry(X, np.stack([xs, ys], axis=-1).astype(np.float32))


def _heatmap(color=U, **options):
    return HeatmapPlot(*_field(), position=X, color=color, **options)


def _curves(extra=None, n=20):
    grid = GridGeometry.build_unit_grid(T, n)
    config = DataConfiguration(
        *([extra] if extra else []), GeometryAxes(grid), FeatureAxes(U * P)
    )
    t = np.linspace(0, 1, n)
    copies = config.shape[0] if extra else 1
    data = np.stack(
        [np.stack([np.sin(2 * np.pi * (t + 0.1 * k)), np.exp(-t) * (k + 1)], -1) for k in range(copies)]
    )
    return data.reshape(config.shape).astype(np.float32), config


def _surface_3d():
    box = Box(Y, [0.0] * 3, 1.0, 1.0, 1.0).create_mesh(0.5)
    config = DataConfiguration(GeometryAxes(box), FeatureAxes(U))
    points = np.asarray(box.discretization_points)
    return SurfacePlot(
        points.sum(-1, keepdims=True).astype(np.float32), config, position=Y, color=U
    )


def _runs():
    rng = np.random.default_rng(0)
    return {
        "lr": 10 ** rng.uniform(-4, -1, 24),
        "optimizer": np.tile(["adam", "sgd", "lbfgs"], 8),
        "loss": rng.uniform(0.01, 1.0, 24),
        "error": 10 ** rng.uniform(-3, 0, 24),
    }


def _comparison():
    # Graph.visualize of a model with a port reference: reference,
    # prediction and error side by side; seeded, so it draws alike.
    torch.manual_seed(0)
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
    sampler = GridSampler(square, 20)
    model, reference = (
        FCN(in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=1) for _ in range(2)
    )
    graph = Graph()
    graph.connect(sampler, model)
    graph.connect(sampler, reference)
    graph.setup()
    return graph.visualize(
        model.output_ports[0], reference=reference.output_ports[0], error="absolute"
    )


def _facets():
    batch = BatchAxes(3)
    return HeatmapPlot(*_field(extra=batch), position=X, color=U, controls=[FacetAxis(batch)])


def _overlay():
    batch = BatchAxes(3)
    return LinePlot(*_curves(batch), x=T, y=U, controls=[OverlayAxis(batch)])


def _line_facets():
    batch = BatchAxes(3)
    return LinePlot(*_curves(batch), x=T, y=U, controls=[FacetAxis(batch)])


#: Case name: layout factory.
CASES = {
    "heatmap": _heatmap,
    "heatmap_wide": lambda: HeatmapPlot(*_field(_wide()), position=X, color=U),
    "heatmap_log": lambda: _heatmap(color=P, color_scale=ColorScale(log=True)),
    "row_separate_scales": lambda: Row(_heatmap(U), _heatmap(P), _heatmap(U)),
    "row_shared_scale": lambda: Row(*(_heatmap(color_scale=scale) for scale in [ColorScale()] * 2)),
    "column": lambda: Column(_heatmap(U), _heatmap(P)),
    "grid_2x2": lambda: Column(Row(_heatmap(U), _heatmap(P)), Row(_heatmap(U), _heatmap(P))),
    "facets": _facets,
    "image": lambda: ImagePlot(*_field(), position=X, color=C),
    "vector": lambda: VectorPlot(*_field(), position=X, vector=V),
    "scatter_domain": lambda: ScatterPlot(*_field(), position=X, color=U),
    "surface_height": lambda: SurfacePlot(*_field(), position=X, z=U),
    "surface_3d": _surface_3d,
    "line_overlay": _overlay,
    "line_tuple": lambda: LinePlot(*_curves(), x=T, y=(U, P)),
    "line_facets": _line_facets,
    "bar": lambda: BarPlot(*_curves(n=8), x=T, y=U),
    "line_and_heatmap": lambda: Row(LinePlot(*_curves(), x=T, y=U), _heatmap()),
    "table_categorical": lambda: TableScatter(
        _runs(), x="lr", y="loss", color="optimizer", log_columns=["lr"]
    ),
    "table_log": lambda: TableScatter(
        _runs(), x="lr", y="error", color="loss", log_columns=["lr", "error"]
    ),
    "table_heatmap": lambda: TableHeatMap(
        _runs(), x="lr", y="loss", color="error", bins_x=4, bins_y=4, log_columns=["lr"]
    ),
    "comparison": _comparison,
    "parallel": lambda: ParallelCoordinatesPlot(
        _runs(), columns=["lr", "loss", "error"], color="loss", log_columns=["lr", "error"]
    ),
}

#: (case, renderer) -> why the renderer does not draw the case.
NOT_DRAWN = {("parallel", "matplotlib"): "Matplotlib has no native parallel coordinates"}


#: Case: how the drawn position ranges end, for all cells or per cell.
RANGES = {
    **{
        name: "exact"
        for name in (
            "heatmap", "heatmap_wide", "heatmap_log", "row_separate_scales",
            "row_shared_scale", "column", "grid_2x2", "facets", "image",
            "surface_height", "surface_3d", "table_heatmap", "parallel",
        )
    },
    **{
        name: "margin"
        for name in (
            "vector", "scatter_domain", "line_overlay", "line_tuple", "line_facets",
            "bar", "table_categorical", "table_log", "comparison",
        )
    },
    "line_and_heatmap": ("margin", "exact"),
}

#: Case: (column widths, row heights) of the plot areas, in pixels, in the
#: default plot box of 400 x 320; a square domain gives 320 x 320.
SIZES = {
    **{
        name: ([320], [320])
        for name in ("heatmap", "heatmap_log", "image", "vector", "scatter_domain")
    },
    **{
        name: ([400], [320])
        for name in (
            "surface_height", "surface_3d", "line_overlay", "line_tuple", "bar",
            "table_categorical", "table_log", "table_heatmap", "parallel",
        )
    },
    "heatmap_wide": ([400], [100]),  # [0, 2] x [0, 0.5]
    "row_separate_scales": ([320] * 3, [320]),
    "row_shared_scale": ([320] * 2, [320]),
    "column": ([320], [320] * 2),
    "grid_2x2": ([320] * 2, [320] * 2),
    "facets": ([320] * 3, [320]),
    "line_facets": ([400] * 3, [320]),
    "line_and_heatmap": ([400, 320], [320]),
    "comparison": ([400] * 3, [300]),  # 5 x 4 sampled points
}

#: Case: the entries of every legend; no legend for the others.
LEGENDS = {
    "line_overlay": [["$u$, batch = 0", "$u$, batch = 1", "$u$, batch = 2"]],
    "line_tuple": [["$u$", "$p$"]],
}

#: The renderers whose layout is checked: plot areas of their sizes,
#: colorbars beside the plot area of their last cell, as tall as it and over
#: no other plot area.
LAID_OUT = {"plotly", "matplotlib"}

#: How far a drawn plot area may miss its size, in pixels: Matplotlib's
#: layout places the plot areas in room that is estimated, not measured.
#: Its 3D boxes take their slot by their projection, unchecked.
SIZE_TOLERANCE = {"plotly": 1, "matplotlib": 20}

#: The fraction of the span a margin adds on both sides.
MARGIN = 0.04


@pytest.mark.parametrize("renderer_name", list(RENDERERS))
@pytest.mark.parametrize("case_name", list(CASES))
def test_appearance(case_name, renderer_name):
    if (case_name, renderer_name) in NOT_DRAWN:
        pytest.skip(NOT_DRAWN[(case_name, renderer_name)])
    figure = Figure(CASES[case_name](), renderer=RENDERERS[renderer_name](), title=case_name)
    backend = figure.draw()
    _check_ranges(figure, backend, renderer_name, RANGES[case_name])
    widths, heights = SIZES[case_name]
    assert figure.plot_sizes() == (pytest.approx(widths), pytest.approx(heights))
    if case_name == "bar":
        assert figure.cells[0].axes["y"].scale.range[0] <= 0.0
    expected = LEGENDS.get(case_name, [])
    assert [list(entries) for entries, _ in figure.legends()] == expected
    assert _drawn_legends(figure, backend, renderer_name) == [
        [figure.renderer.text(label) for label in labels] for labels in expected
    ]
    if renderer_name in LAID_OUT:
        _check_layout(figure, backend, renderer_name)
    if renderer_name == "plotly":
        _check_scenes(figure, backend)


def test_every_case_has_its_expectations():
    assert set(RANGES) == set(SIZES) == set(CASES)


def _check_ranges(figure, backend, renderer_name, kinds):
    # The drawn range of every position axis is the scale's range,
    # widened by the margin of its span (in the scale's transformed
    # values).
    kinds = [kinds] * len(figure.cells) if isinstance(kinds, str) else kinds
    for cell, kind in zip(figure.cells, kinds):
        for channel, axis in cell.axes.items():
            scale = axis.scale
            lo, hi = scale.transform(np.asarray(scale.range))
            pad = (MARGIN if kind == "margin" else 0.0) * (hi - lo)
            drawn = _drawn_range(backend, renderer_name, cell, channel)
            assert scale.transform(drawn) == pytest.approx([lo - pad, hi + pad]), (
                f"{channel} of {cell.title}"
            )


def _drawn_range(backend, renderer_name, cell, channel):
    # The range a renderer drew a position axis with, in data values.
    if renderer_name == "matplotlib":
        return getattr(axes_of(backend, cell), f"get_{channel}lim")()
    subplot = backend.get_subplot(cell.row, cell.col)
    drawn = getattr(subplot, f"{channel}axis").range
    scale = cell.axes[channel].scale
    return scale.inverse(drawn)


# --- theme rules -------------------------------------------------------------


def _colorbar_axes(backend):
    (bar,) = colorbar_axes(backend)
    return bar


#: Theme field: (case, {renderer: the parts of the drawn figure the switch
#: turns on or off}).
THEME_RULES = {
    "axis_box": (
        "heatmap",
        {
            "plotly": lambda b: (b.layout.xaxis.mirror, b.data[-1].marker.colorbar.outlinewidth > 0),
            "matplotlib": lambda b: (
                b.axes[0].spines["top"].get_visible(),
                _colorbar_axes(b).spines["outline"].get_visible(),
            ),
        },
    ),
    "show_zero_line": (
        "line_tuple",
        {
            "plotly": lambda b: (b.layout.xaxis.zeroline,),
            "matplotlib": lambda b: (
                any(list(line.get_ydata()) == [0, 0] for line in b.axes[0].lines),
            ),
        },
    ),
    "legend_box": (
        "line_overlay",
        {
            "plotly": lambda b: (b.layout.legend.borderwidth > 0,),
            "matplotlib": lambda b: (b.axes[0].get_legend().get_frame_on(),),
        },
    ),
    "show_grid": (
        "line_tuple",
        {
            "plotly": lambda b: (b.layout.xaxis.showgrid,),
            "matplotlib": lambda b: (b.axes[0].xaxis.get_gridlines()[0].get_visible(),),
        },
    ),
}


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("renderer_name", list(RENDERERS))
@pytest.mark.parametrize("field", list(THEME_RULES))
def test_theme_rule(field, renderer_name, value):
    case, reads = THEME_RULES[field]
    theme = dataclasses.replace(LIGHT_THEME, **{field: value})
    backend = Figure(CASES[case](), renderer=RENDERERS[renderer_name](theme=theme)).draw()
    parts = reads[renderer_name](backend)
    assert [bool(part) for part in parts] == [value] * len(parts)


def test_the_presets_follow_the_data_ink_rules():
    for theme in (LIGHT_THEME, DARK_THEME):
        assert not (theme.axis_box or theme.show_grid or theme.show_zero_line or theme.legend_box)
        assert theme.show_axis_line


@pytest.mark.parametrize("renderer_name", list(RENDERERS))
def test_texts_are_drawn_in_the_theme_sizes(renderer_name):
    # Tick labels in font_size_labels; axis, cell and colorbar titles in
    # font_size_axes; the figure title in font_size_title.
    theme = LIGHT_THEME
    figure = Figure(_heatmap(title="cell"), renderer=RENDERERS[renderer_name](), title="figure")
    backend = figure.draw()
    labels, axes, title = theme.font_size_labels, theme.font_size_axes, theme.font_size_title
    if renderer_name == "plotly":
        layout = backend.layout
        assert layout.xaxis.tickfont.size == labels
        assert layout.xaxis.title.font.size == axes
        assert [a.font.size for a in layout.annotations] == [axes]
        assert backend.data[-1].marker.colorbar.title.font.size == axes
        assert layout.title.font.size == title
        return
    cell = backend.axes[0]
    assert cell.xaxis.get_ticklabels()[0].get_fontsize() == points(labels)
    assert cell.xaxis.label.get_fontsize() == points(axes)
    assert cell.title.get_fontsize() == points(axes)
    assert _colorbar_axes(backend).yaxis.label.get_fontsize() == points(axes)
    (suptitle,) = [t for t in backend.texts if t.get_text() == "figure"]
    assert suptitle.get_fontsize() == points(title)


# --- layout ------------------------------------------------------------------


def _check_layout(figure, backend, renderer_name):
    if renderer_name == "matplotlib":
        with figure.renderer.style():
            backend.draw_without_rendering()  # constrained layout places when rendering
    widths, heights = figure.plot_sizes()
    areas = _plot_areas(figure, backend, renderer_name)
    tolerance = SIZE_TOLERANCE[renderer_name]
    for cell in figure.cells:
        if renderer_name == "matplotlib" and cell.dim == 3:
            continue
        x0, x1, y0, y1 = areas[cell]
        assert (x1 - x0, y1 - y0) == (
            pytest.approx(widths[cell.col - 1], abs=tolerance),
            pytest.approx(heights[cell.row - 1], abs=tolerance),
        ), cell.title
    for cell, (x0, x1, y0, y1) in _colorbars(figure, backend, renderer_name):
        cx0, cx1, cy0, cy1 = areas[cell]
        assert x0 >= cx1 - 1
        assert (y0, y1) == (pytest.approx(cy0, abs=1), pytest.approx(cy1, abs=1))
        for other, (ox0, ox1, oy0, oy1) in areas.items():
            overlaps = x0 < ox1 and ox0 < x1 and y0 < oy1 and oy0 < y1
            assert not overlaps, f"a colorbar of {cell.title} covers {other.title}"


def _matplotlib_box(backend, axes) -> tuple:
    # The box of Axes in pixels from the top left.
    width, height = backend.get_size_inches() * PIXELS_PER_INCH
    box = axes.get_position()
    return (box.x0 * width, box.x1 * width, (1 - box.y1) * height, (1 - box.y0) * height)


def _plot_areas(figure, backend, renderer_name) -> dict:
    # The plot area of every cell, in pixels from the top left.
    if renderer_name == "matplotlib":
        return {cell: _matplotlib_box(backend, axes_of(backend, cell)) for cell in figure.cells}
    width, height = backend.layout.width, backend.layout.height
    areas = {}
    for cell in figure.cells:
        subplot = backend.get_subplot(cell.row, cell.col)
        if cell.dim == 0:  # its trace lays out its own axes
            # The plot's trace; the carrier of its colorbar follows it.
            trace = next(backend.select_traces(selector=dict(type="parcoords")))
            (x0, x1), (y0, y1) = trace.domain.x, trace.domain.y
        elif cell.dim == 3:
            (x0, x1), (y0, y1) = subplot.domain.x, subplot.domain.y
        else:
            (x0, x1), (y0, y1) = subplot.xaxis.domain, subplot.yaxis.domain
        areas[cell] = (x0 * width, x1 * width, (1 - y1) * height, (1 - y0) * height)
    return areas


def _colorbars(figure, backend, renderer_name) -> list:
    # Every colorbar with the cell it stands beside, as its bar in pixels.
    if renderer_name == "matplotlib":
        bars = [
            (cell, _matplotlib_box(backend, bar))
            for cell in figure.cells
            for bar in colorbar_axes(backend, cell)
        ]
        assert len(bars) == len(figure.color_scales)
        return bars
    width, height = backend.layout.width, backend.layout.height
    bars = []
    for scale in figure.color_scales:
        bar = figure.renderer.colorbar(scale)
        x0 = bar.x * width
        pad = bar.ypad if bar.ypad is not None else 10  # Plotly's default
        y0 = (1 - bar.y - bar.len / 2) * height + pad
        bars.append((figure.colorbar_cell(scale), (x0, x0 + bar.thickness, y0, y0 + bar.len * height - 2 * pad)))
    return bars


def _drawn_legends(figure, backend, renderer_name) -> list:
    # The entries of every drawn legend, as the renderer wrote them.
    if renderer_name == "matplotlib":
        return [
            [t.get_text() for t in legend.get_texts()]
            for legend in (axes_of(backend, c).get_legend() for c in figure.cells)
            if legend is not None
        ]
    legends: dict = {}
    for trace in backend.data:
        if getattr(trace, "showlegend", False):
            legends.setdefault(trace.legend or "legend", []).append(trace.name)
    return list(legends.values())


# --- colorbar slots ----------------------------------------------------------


def _two_colored_plots():
    # A heatmap of u with the points of the grid colored by p, in one cell.
    data, config = _field()
    return Overlay(
        HeatmapPlot(data, config, position=X, color=U),
        ScatterPlot(data, config, position=X, color=P),
    )


#: Layout: the (row, col) and slot of the colorbar of every color scale, in
#: the figure's order of scales.
SLOTS = {
    "candidates of a selector share a place": (
        lambda: HeatmapPlot(*_field(), position=X, color=Selector([U, P])),
        [((1, 1), 0), ((1, 1), 0)],
    ),
    "two colored plots of a cell stand side by side": (_two_colored_plots, [((1, 1), 0), ((1, 1), 1)]),
    "cells of their own": (CASES["row_separate_scales"], [((1, 1), 0), ((1, 2), 0), ((1, 3), 0)]),
    "a shared scale beside its last cell": (CASES["row_shared_scale"], [((1, 2), 0)]),
}


@pytest.mark.parametrize("name", list(SLOTS))
def test_colorbar_slots(name):
    make, expected = SLOTS[name]
    figure = Figure(make())
    places = [
        ((figure.colorbar_cell(s).row, figure.colorbar_cell(s).col), figure.colorbar_slot(s))
        for s in figure.color_scales
    ]
    assert places == expected


# --- notebook output ---------------------------------------------------------


#: Renderer variant: (renderer factory, module whose _in_notebook decides,
#: what shows the figure's size in its HTML).
NOTEBOOK = {
    "plotly": (PlotlyRenderer, "qewton.visualization.renderers.plotly", "width"),
    "plotly_markup": (
        lambda: PlotlyRenderer(mathjax=False),
        "qewton.visualization.renderers.plotly",
        "width",
    ),
    "matplotlib": (MatplotlibRenderer, "qewton.visualization.renderers.matplotlib", "<img"),
}


@pytest.mark.parametrize("variant", list(NOTEBOOK))
def test_a_notebook_shows_a_figure_at_its_size_scrolling(variant, monkeypatch):
    pytest.importorskip("IPython")  # present wherever a notebook runs
    make, module, sized = NOTEBOOK[variant]
    shown = []
    monkeypatch.setattr(f"{module}._in_notebook", lambda: True)
    monkeypatch.setattr("IPython.display.display", lambda obj: shown.append(obj.data))
    Figure(CASES["row_separate_scales"](), renderer=make()).show()
    (page,) = shown
    assert page.startswith('<div class="qewton-figure" style="overflow-x: auto;')
    assert "max-width: none" in page and sized in page


#: The application showing each renderer's figures, and what an update
#: keeps of its layout.
APPLICATIONS = {
    "plotly": (DashApplication, "dash", lambda b: (b.layout.width, b.layout.height)),
    "matplotlib": (WidgetApplication, "ipympl", lambda b: tuple(b.get_size_inches())),
}


@pytest.mark.parametrize("renderer_name", list(APPLICATIONS))
def test_an_update_keeps_the_room_of_every_candidate(renderer_name):
    # A connected Selector between a numeric and a categorical color: the
    # room beside the cell holds the titles and labels of both colorbars
    # from the start.
    application, package, size_of = APPLICATIONS[renderer_name]
    pytest.importorskip(package)
    selector = Selector(["loss", "optimizer"])
    plot = TableScatter(_runs(), x="lr", y="loss", color=selector, log_columns=["lr"])
    figure = Figure(plot, renderer=RENDERERS[renderer_name]())
    application.create(figure)
    size = size_of(figure.backend_figure)
    for state in selector.states:
        application.update(figure, [selector], [state])
        assert size_of(figure.backend_figure) == size


def _check_scenes(figure, backend):
    # Scenes render no MathJax: their titles are markup. Their camera is
    # far enough for the box and its labels to fit.
    for cell in figure.cells:
        if cell.dim != 3:
            continue
        scene = backend.get_subplot(cell.row, cell.col)
        titles = [getattr(scene, f"{c}axis").title.text for c in "xyz"]
        assert all(t and "$" not in t for t in titles), titles
        assert scene.camera.eye.x > 1.25
