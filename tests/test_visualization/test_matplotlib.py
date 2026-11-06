"""Rules of the Matplotlib renderer that the matrices do not check: theme
settings, texts, shared axes, drawing order, outputs, omitted plots and
embedded frames."""

import io
import warnings

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import same_color

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    Figure,
    GraphPlot,
    HeatmapPlot,
    Overlay,
    PositionAxis,
    Row,
    ScatterPlot,
    SliderAxis,
    TimeAxis,
)
from qewton.visualization.renderers import MatplotlibRenderer
from qewton.visualization.renderers.matplotlib import style
from qewton.visualization.renderers.matplotlib.text import as_mathtext
from qewton.visualization.themes import DARK_THEME, LIGHT_THEME, cmaps

T, X, U = Variable("t", 1), Variable("x", 2), Variable("u", 1)


def _heatmap():
    config = DataConfiguration(
        GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U)
    )
    data = np.random.default_rng(0).random(config.shape)
    return HeatmapPlot(data, config, position=X, color=U)


def _points(batch=None):
    axes = (batch,) if batch is not None else ()
    config = DataConfiguration(
        *axes, GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U)
    )
    data = np.random.default_rng(1).random(config.shape)
    controls = [SliderAxis(batch)] if batch is not None else []
    return ScatterPlot(data, config, position=X, controls=controls)


def _cells(backend):
    return [a for a in backend.axes if a.get_label().startswith("cell-")]


# --- theme ----------------------------------------------------------------------


@pytest.mark.parametrize("theme", [LIGHT_THEME, DARK_THEME], ids=["light", "dark"])
def test_the_theme_applies_to_the_figure_only(theme):
    before = dict(matplotlib.rcParams)
    figure = Figure(_heatmap(), renderer=MatplotlibRenderer(theme=theme))
    backend = figure.draw()
    figure.renderer.save_image(io.BytesIO(), "png")
    assert same_color(backend.get_facecolor(), style.color(theme.background_color))
    (axes,) = _cells(backend)
    assert same_color(axes.get_facecolor(), style.color(theme.background_color))
    assert dict(matplotlib.rcParams) == before


#: Theme color: (css, Matplotlib color).
COLORS = {
    "hex": ("#a1b2c3", "#a1b2c3"),
    "rgb": ("rgb(255, 0, 0)", (1.0, 0.0, 0.0)),
    "rgba": ("rgba(0,0,255,0.5)", (0.0, 0.0, 1.0, 0.5)),
}


@pytest.mark.parametrize("name", list(COLORS))
def test_theme_colors_become_matplotlib_colors(name):
    css, expected = COLORS[name]
    assert same_color(style.color(css), expected)


#: Font list: (font.family, the fonts tried in order).
FONTS = {
    "serif": (
        "Latin Modern Roman, CMU Serif, serif",
        ("serif", ["Latin Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"]),
    ),
    "sans-serif": (
        "Arial, Helvetica, sans-serif",
        ("sans-serif", ["Arial", "Helvetica", "DejaVu Sans"]),
    ),
    "without generic": ("Georgia", ("serif", ["Georgia", "cmr10", "DejaVu Serif"])),
}


@pytest.mark.parametrize("name", list(FONTS))
def test_fonts_fall_back_to_computer_modern(name):
    font_family, (family, fonts) = FONTS[name]
    params = style.fonts(font_family)
    assert params["font.family"] == family
    assert params[f"font.{family}"] == fonts


#: Colormap: (as resolved by a theme, its color at 0).
COLORMAPS = {
    "qewton": (
        LIGHT_THEME.resolve_cmap(cmaps.meadow),
        style.color(cmaps.meadow.colorscale[0][1]),
    ),
    "native": ("viridis", matplotlib.colormaps["viridis"](0.0)),
}


@pytest.mark.parametrize("name", list(COLORMAPS))
def test_colormaps(name):
    resolved, first = COLORMAPS[name]
    assert same_color(style.colormap(resolved)(0.0), first)


def test_an_unknown_colormap_name_raises():
    with pytest.raises(ValueError, match="no Matplotlib colormap"):
        style.colormap("Viridis")


# --- texts ----------------------------------------------------------------------

#: Text: (as written, whether mathtext renders it).
TEXTS = {
    "plain": ("loss", True),
    "math": ("$x_1$", True),
    "mixed": ("$t$ = 0.5", True),
    "odd dollars": ("costs $ 5", True),
    "unknown command": (r"$\unknowncommand$ = 1", False),
}


@pytest.mark.parametrize("name", list(TEXTS))
def test_texts_are_mathtext_or_plain(name):
    text, renders = TEXTS[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shown = as_mathtext(text)
    assert (shown == text) == renders
    assert bool(caught) == (not renders)
    if not renders:
        assert "$" not in shown


def test_usetex_needs_latex(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="LaTeX"):
        MatplotlibRenderer(usetex=True)


# --- axes and drawing order -----------------------------------------------------


def _sharing_row():
    shared = dict(x=PositionAxis(), y=PositionAxis())
    return Row(Overlay(_heatmap(), **shared), Overlay(_heatmap(), **shared))


#: Layout: (factory, whether the second cell shares the first one's axes).
SHARING = {
    "shared axes": (_sharing_row, True),
    "own axes": (lambda: Row(_heatmap(), _heatmap()), False),
}


@pytest.mark.parametrize("name", list(SHARING))
def test_cells_sharing_a_position_axis_share_the_matplotlib_axis(name):
    make, shared = SHARING[name]
    first, second = _cells(Figure(make(), renderer=MatplotlibRenderer()).draw())
    assert first.get_shared_x_axes().joined(first, second) == shared
    assert first.get_shared_y_axes().joined(first, second) == shared


def test_an_update_keeps_the_drawing_order():
    points = _points(BatchAxes(2))
    figure = Figure(Overlay(_heatmap(), points), renderer=MatplotlibRenderer())
    (axes,) = _cells(figure.draw())
    points.controls[0].state = 1
    figure.draw()
    mesh, scatter = sorted(axes.collections, key=lambda c: c.get_zorder())
    assert (
        type(mesh).__name__ == "QuadMesh" and type(scatter).__name__ == "PathCollection"
    )


# --- outputs and omitted plots --------------------------------------------------

#: Format: the first bytes of the file.
FORMATS = {"png": b"\x89PNG", "svg": b"<?xml", "pdf": b"%PDF"}


@pytest.mark.parametrize("format", list(FORMATS))
def test_images_need_no_external_tools(format, tmp_path):
    figure = Figure(_heatmap(), renderer=MatplotlibRenderer())
    figure.draw()
    path = tmp_path / f"figure.{format}"
    figure.renderer.save_image(path, format)
    assert path.read_bytes().startswith(FORMATS[format])


def test_html_is_a_page_without_scripts(tmp_path):
    figure = Figure(_heatmap(), renderer=MatplotlibRenderer())
    figure.save_html(tmp_path / "figure.html")
    page = (tmp_path / "figure.html").read_text()
    assert "data:image/svg+xml;base64," in page and "<script" not in page


def test_graphs_are_not_drawn(simple_graph):
    with pytest.raises(
        ValueError, match="MatplotlibRenderer cannot draw a GraphLayoutResult"
    ):
        Figure(GraphPlot(simple_graph), renderer=MatplotlibRenderer()).draw()


# --- embedded controls ----------------------------------------------------------


def _animated(control):
    batch = BatchAxes(4)
    config = DataConfiguration(
        batch, GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U)
    )
    data = np.random.default_rng(2).random(config.shape)
    axis = control(batch)
    plot = HeatmapPlot(data, config, position=X, color=U, controls=[axis])
    return Figure(plot, renderer=MatplotlibRenderer(), title="title"), axis


def test_a_gif_has_a_frame_per_time_step(tmp_path):
    from PIL import Image

    figure, _ = _animated(TimeAxis)
    figure.save_gif(tmp_path / "figure.gif")
    assert Image.open(tmp_path / "figure.gif").n_frames == 4


@pytest.mark.parametrize("control", [TimeAxis, SliderAxis], ids=["time", "slider"])
def test_writing_frames_keeps_the_current_state(control, tmp_path):
    figure, axis = _animated(control)
    axis.state = 2
    figure.draw()
    (cell,) = _cells(figure.backend_figure)
    before = cell.collections[0].get_array().copy()
    figure.save_html(tmp_path / "figure.html")
    np.testing.assert_array_equal(cell.collections[0].get_array(), before)
    assert figure.backend_figure.get_suptitle() == "title"
