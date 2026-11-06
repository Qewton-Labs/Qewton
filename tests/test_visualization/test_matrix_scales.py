"""Test matrix of color scales: plot with color × scale variant.

Every plot is stepped by a slider over three states whose values differ,
so a scale trained over all states spans more than the current one. Each
variant states the range it expects from the values of every state, and
the number of colorbars the figure draws; log scales get ticks labeled by
values. Position axis scales follow as a smaller table.
"""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.continuous.domains_3d.box import Box
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    ColorScale,
    Figure,
    HeatmapPlot,
    LinePlot,
    Overlay,
    ParallelCoordinatesPlot,
    PositionAxis,
    Row,
    Scale,
    ScatterPlot,
    Selector,
    SliderAxis,
    SurfacePlot,
    TableScatter,
    VectorPlot,
)
from qewton.visualization.renderers import MatplotlibRenderer, PlotlyRenderer
from qewton.visualization.renderers.matplotlib.primitives import colorbar_axes
from qewton.visualization.scales import log_ticks

from recording_renderer import RecordingRenderer

T, X, Y = Variable("t", 1), Variable("x", 2), Variable("y", 3)
U, P, R = Variable("u", 1), Variable("p", 1), Variable("r", 2)
STEPS = 3


def _data(domain, offset=0.0):
    """u and p over the domain in 3 states; state k is shifted by k, so the
    values of all states span more than one state's."""
    batch = BatchAxes(STEPS)
    config = DataConfiguration(batch, GeometryAxes(domain), FeatureAxes(U * P * R))
    values = np.random.default_rng(0).random(config.shape) * 0.9 + 0.1
    values += np.arange(STEPS).reshape(-1, *[1] * (len(config.shape) - 1)) + offset
    return values, config, batch


def _table(offset=0.0):
    rng = np.random.default_rng(1)
    loss = rng.random(12) * 0.9 + 0.1 + offset
    return {"width": np.repeat([8.0, 16.0, 32.0], 4), "lr": rng.random(12), "loss": loss}


def _square_mesh():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)


#: Plot name: factory(color reference, color_scale, offset) -> (plot, values
#: of the color over every state).
def _heatmap(color, scale, offset):
    data, config, batch = _data(GridGeometry.build_unit_grid(X, 4), offset)
    plot = HeatmapPlot(data, config, position=X, color=color, color_scale=scale,
                       controls=[SliderAxis(batch)])
    return plot, data


def _mesh_heatmap(color, scale, offset):
    data, config, batch = _data(_square_mesh(), offset)
    plot = HeatmapPlot(data, config, position=X, color=color, color_scale=scale,
                       controls=[SliderAxis(batch)])
    return plot, data


def _scatter(color, scale, offset):
    data, config, batch = _data(_square_mesh(), offset)
    plot = ScatterPlot(data, config, position=X, color=color, color_scale=scale,
                       controls=[SliderAxis(batch)])
    return plot, data


def _path(color, scale, offset):
    data, config, batch = _data(GridGeometry.build_unit_grid(T, 6), offset)
    plot = LinePlot(data, config, position=R, color=color, color_scale=scale,
                    controls=[SliderAxis(batch)])
    return plot, data


def _surface(color, scale, offset):
    data, config, batch = _data(Box(Y, [0.0] * 3, 1, 1, 1).create_mesh(0.6), offset)
    plot = SurfacePlot(data, config, position=Y, color=color, color_scale=scale,
                       controls=[SliderAxis(batch)])
    return plot, data


def _arrows(color, scale, offset):
    data, config, batch = _data(_square_mesh(), offset)
    plot = VectorPlot(data, config, position=X, vector=R, color=color, color_scale=scale,
                      controls=[SliderAxis(batch)])
    return plot, data


def _table_scatter(color, scale, offset):
    table = _table(offset)
    plot = TableScatter(table, x="lr", y="width", color=_column(color), color_scale=scale,
                        controls=[SliderAxis("width")])
    return plot, table


def _parallel(color, scale, offset):
    table = _table(offset)
    plot = ParallelCoordinatesPlot(table, columns=["lr", "loss"], color=_column(color),
                                   color_scale=scale, controls=[SliderAxis("width")])
    return plot, table


def _column(color):
    # The table column of a color reference: u is the loss.
    if isinstance(color, Selector):
        return Selector(["loss", "lr"])
    return "loss"


PLOTS = {
    "heatmap": _heatmap,
    "mesh_heatmap": _mesh_heatmap,
    "scatter": _scatter,
    "path": _path,
    "surface": _surface,
    "arrows": _arrows,
    "table_scatter": _table_scatter,
    "parallel": _parallel,
}

#: Plots Matplotlib does not draw.
NOT_IN_MATPLOTLIB = {"parallel"}


def _values(plot_name, data, candidate=0):
    # Every value of the color candidate over all states.
    if plot_name in ("table_scatter", "parallel"):
        return np.asarray(data[["loss", "lr"][candidate]])
    return np.asarray(data)[..., candidate]


#: Variant: (scale factory, expected range from all values).
SCALES = {
    "default": (lambda: None, lambda v: (v.min(), v.max())),
    "fixed": (lambda: ColorScale(vmin=0.0, vmax=10.0), lambda v: (0.0, 10.0)),
    "symmetric": (
        lambda: ColorScale(symmetric=True),
        lambda v: (-np.abs(v).max(), np.abs(v).max()),
    ),
    "log": (lambda: ColorScale(log=True), lambda v: (v.min(), v.max())),
}


@pytest.mark.parametrize("scale_name", list(SCALES))
@pytest.mark.parametrize("plot_name", list(PLOTS))
def test_color_scale(plot_name, scale_name):
    make_scale, expected_range = SCALES[scale_name]
    plot, data = PLOTS[plot_name](U, make_scale(), 0.0)
    backend = Figure(plot, renderer=RecordingRenderer()).draw()
    assert plot.color_scale.range == pytest.approx(expected_range(_values(plot_name, data)))
    assert len(backend["colorbars"]) == 1
    if scale_name == "log":
        _check_log_ticks(plot)
        if plot_name not in NOT_IN_MATPLOTLIB:
            _check_log_ticks_matplotlib(plot)


def _check_log_ticks_matplotlib(plot):
    # The colorbar is drawn by log10 values, its ticks labeled by values
    # within the range.
    renderer = MatplotlibRenderer()
    backend = Figure(plot, renderer=renderer).draw()
    (bar,) = colorbar_axes(backend)
    ticks = bar.get_yticks()
    labels = [t.get_text() for t in bar.get_yticklabels()]
    assert 10 ** np.asarray(ticks) == pytest.approx([float(t) for t in labels])
    lo, hi = plot.color_scale.range
    assert np.all((10**ticks >= lo * 0.999) & (10**ticks <= hi * 1.001))


def _check_log_ticks(plot):
    # Ticks of a log scale are labeled by values within the range.
    renderer = PlotlyRenderer()
    Figure(plot, renderer=renderer).draw()
    colorbar = renderer.colorbar(plot.color_scale)
    values = 10 ** np.asarray(colorbar.tickvals)
    assert values == pytest.approx([float(t) for t in colorbar.ticktext])
    lo, hi = plot.color_scale.range
    assert np.all((values >= lo * 0.999) & (values <= hi * 1.001))


@pytest.mark.parametrize("plot_name", list(PLOTS))
def test_a_shared_scale_spans_both_plots_with_one_colorbar(plot_name):
    scale = ColorScale()
    first, data = PLOTS[plot_name](U, scale, 0.0)
    second, more = PLOTS[plot_name](U, scale, 10.0)
    backend = Figure(Row(first, second), renderer=RecordingRenderer()).draw()
    values = np.concatenate(
        [_values(plot_name, data).ravel(), _values(plot_name, more).ravel()]
    )
    assert scale.range == pytest.approx((values.min(), values.max()))
    assert len(backend["colorbars"]) == 1


#: Selector variant: (color_scale argument, distinct scales, shared range).
SELECTOR_SCALES = {
    "one_per_candidate": (lambda: None, 2, False),
    "one_shared": (lambda: ColorScale(), 1, True),
    "a_list": (lambda: [ColorScale(), ColorScale(symmetric=True)], 2, False),
}


@pytest.mark.parametrize("variant", list(SELECTOR_SCALES))
@pytest.mark.parametrize("plot_name", list(PLOTS))
def test_scales_of_a_selector(plot_name, variant):
    make, n_scales, shared = SELECTOR_SCALES[variant]
    selector = Selector([U, P])
    plot, data = PLOTS[plot_name](selector, make(), 0.0)
    selector = plot.selectors[-1]
    backend = Figure(plot, renderer=RecordingRenderer()).draw()
    assert len(plot.color_scales) == n_scales
    if shared:
        both = np.concatenate([_values(plot_name, data, 0).ravel(), _values(plot_name, data, 1).ravel()])
        assert plot.color_scale.range == pytest.approx((both.min(), both.max()))
    shown = [visible for *_, visible in backend["colorbars"].values()]
    assert shown.count(True) == 1 and len(shown) == n_scales


def test_a_list_must_give_one_scale_per_candidate():
    with pytest.raises(ValueError, match="one per Selector candidate"):
        _heatmap(Selector([U, P]), [ColorScale()], 0.0)


def test_a_log_scale_needs_positive_bounds():
    with pytest.raises(ValueError, match="positive bounds"):
        ColorScale(log=True, vmin=0.0)


# --- position axes -----------------------------------------------------------


#: Variant: (axis scale, check of the trained range against the values).
AXIS_SCALES = {
    "linear": (Scale(), lambda r, v: r == pytest.approx((v.min(), v.max()))),
    "log": (Scale(log=True), lambda r, v: r == pytest.approx((v.min(), v.max()))),
    "fixed": (Scale(vmin=-1.0, vmax=100.0), lambda r, v: r == (-1.0, 100.0)),
}


@pytest.mark.parametrize("variant", list(AXIS_SCALES))
def test_position_axis_scale(variant):
    scale, check = AXIS_SCALES[variant]
    data, config, batch = _data(GridGeometry.build_unit_grid(T, 6))
    line = LinePlot(data, config, x=T, y=U, controls=[SliderAxis(batch)])
    overlay = Overlay(line, y=PositionAxis(scale=scale))
    backend = Figure(overlay, renderer=PlotlyRenderer()).draw()
    assert check(scale.range, data[..., 0])
    assert backend.layout.yaxis.type == ("log" if scale.log else "linear")
    figure = Figure(overlay, renderer=MatplotlibRenderer())
    (axes,) = figure.draw().axes
    assert axes.get_yscale() == ("log" if scale.log else "linear")
    assert axes.get_ylim() == pytest.approx(figure.cells[0].axis_range("y"))


def test_a_shared_position_scale_spans_both_overlays():
    scale = Scale()
    overlays = []
    for offset in (0.0, 10.0):
        data, config, batch = _data(GridGeometry.build_unit_grid(T, 6), offset)
        line = LinePlot(data, config, x=T, y=U, controls=[SliderAxis(batch)])
        overlays.append(Overlay(line, y=PositionAxis(scale=scale)))
    Figure(Row(*overlays), renderer=RecordingRenderer()).draw()
    lo, hi = scale.range
    assert lo < 1.0 and hi > 10.0


# --- channels drawn to one scale ------------------------------------------------


def _surface_over_plane():
    data, config, batch = _data(_square_mesh())
    return SurfacePlot(data, config, position=X, z=U, controls=[SliderAxis(batch)])


def _volume_points():
    data, config, batch = _data(Box(Y, [0.0] * 3, 1, 1, 1).create_mesh(0.6))
    return ScatterPlot(data, config, position=Y, color=U, controls=[SliderAxis(batch)])


def _values_in_3d():
    Q = Variable("q", 1)
    config = DataConfiguration(BatchAxes(10), FeatureAxes(U * P * Q))
    data = np.random.default_rng(0).random(config.shape)
    return ScatterPlot(data, config, x=U, y=P, z=Q)


def _graph_over_time():
    data, config, batch = _data(GridGeometry.build_unit_grid(T, 6))
    return LinePlot(data, config, x=T, y=U, controls=[SliderAxis(batch)])


def _path():
    data, config, batch = _data(GridGeometry.build_unit_grid(T, 6))
    return LinePlot(data, config, position=R, controls=[SliderAxis(batch)])


def _field():
    data, config, batch = _data(_square_mesh())
    return HeatmapPlot(data, config, position=X, color=U, controls=[SliderAxis(batch)])


#: Plot: (factory, channels drawn to one scale, fixed 2D aspect, 3D box).
#: A height is as tall as the plane is wide; values against values get a
#: cube.
TO_SCALE = {
    "a field over a plane": (_field, ("x", "y"), True, None),
    "a path through a vector": (_path, ("x", "y"), True, None),
    "a value over time": (_graph_over_time, (), False, None),
    "a height over a plane": (_surface_over_plane, ("x", "y"), False, (1, 1, 1)),
    "points of a volume": (_volume_points, ("x", "y", "z"), False, (1, 1, 1)),
    "values against values": (_values_in_3d, (), False, (1, 1, 1)),
}


@pytest.mark.parametrize("name", list(TO_SCALE))
def test_channels_drawn_to_one_scale(name):
    make, channels, fixed, box = TO_SCALE[name]
    plot = make()
    assert plot.scaled_channels == channels
    _check_aspect_matplotlib(plot, aspect)
    figure = Figure(plot, renderer=PlotlyRenderer())
    backend = figure.draw()
    (cell,) = figure.cells
    assert (cell.aspect is not None) == fixed
    assert cell.box_aspect == (pytest.approx(box) if box is not None else None)
    if plot.dim == 2:
        assert (backend.layout.yaxis.scaleanchor is not None) == fixed
    else:
        assert tuple(backend.layout.scene.aspectratio[c] for c in "xyz") == pytest.approx(box)
    _check_aspect_matplotlib(plot, fixed, box)


def _check_aspect_matplotlib(plot, fixed, box):
    backend = Figure(plot, renderer=MatplotlibRenderer()).draw()
    (axes,) = [a for a in backend.axes if a.get_label().startswith("cell-")]
    if plot.dim == 2:
        assert (axes.get_aspect() == 1.0) == fixed
        return
    drawn = np.asarray(axes.get_box_aspect())
    assert drawn / drawn.max() == pytest.approx(np.asarray(box) / max(box))


# --- log ticks -----------------------------------------------------------------


#: Range: the labels of its log ticks, at most six.
LOG_TICKS = {
    "a decade": ((1, 10), ["1", "2", "5", "10"]),
    "less than a decade": ((2, 5), ["2", "3", "4", "5"]),
    "two decades": ((0.01, 1), ["0.01", "0.03", "0.1", "0.3", "1"]),
    "three decades": ((1e-4, 0.1), ["0.0001", "0.001", "0.01", "0.1"]),
    "within three decades": ((1.2e-4, 0.07), ["0.0003", "0.001", "0.003", "0.01", "0.03"]),
    "many decades": ((1, 1e10), ["1", "100", "10000", "1e+06", "1e+08", "1e+10"]),
}


@pytest.mark.parametrize("name", list(LOG_TICKS))
def test_log_ticks(name):
    (lo, hi), labels = LOG_TICKS[name]
    positions, drawn = log_ticks(np.log10(lo), np.log10(hi))
    assert drawn == labels
    assert positions == pytest.approx([np.log10(float(label)) for label in labels])
