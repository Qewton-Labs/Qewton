"""WidgetApplication: a widget per control of a Matplotlib figure; a
change sets the state and redraws in place."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    FacetAxis,
    Figure,
    FixedAxis,
    HeatmapPlot,
    Selector,
    SliderAxis,
    TimeAxis,
    WidgetApplication,
)
from qewton.visualization.renderers import MatplotlibRenderer, PlotlyRenderer

widgets = pytest.importorskip("ipywidgets")
pytest.importorskip("ipympl")

X, U, P = Variable("x", 2), Variable("u", 1), Variable("p", 1)


def _figure(control, renderer=MatplotlibRenderer):
    # A heatmap with `control` on its batch; a Selector of the color with
    # the batch fixed if `control` is None.
    batch = BatchAxes(3)
    config = DataConfiguration(
        batch, GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U * P)
    )
    data = np.random.default_rng(0).random(config.shape)
    if control is None:
        selector = Selector([U, P])
        plot = HeatmapPlot(
            data, config, position=X, color=selector, controls=[FixedAxis(batch)]
        )
        return Figure(plot, renderer=renderer()), selector
    axis = control(batch)
    plot = HeatmapPlot(data, config, position=X, color=U, controls=[axis])
    return Figure(plot, renderer=renderer()), axis


#: Control: (factory, the widget types of its row, e.g. a play button).
WIDGETS = {
    "slider": (SliderAxis, [widgets.HTMLMath, widgets.SelectionSlider]),
    "time": (TimeAxis, [widgets.HTMLMath, widgets.Play, widgets.SelectionSlider]),
    "selector": (None, [widgets.HTMLMath, widgets.Dropdown]),
}


@pytest.mark.parametrize("name", list(WIDGETS))
def test_a_widget_sets_its_control_and_redraws(name):
    control, kinds = WIDGETS[name]
    figure, control_object = _figure(control)
    app = WidgetApplication.create(figure)
    canvas, row = app.children
    assert [type(w) for w in row.children] == kinds
    (cell,) = [a for a in figure.backend_figure.axes if a.get_label().startswith("cell-")]
    before = cell.collections[0].get_array().copy()
    row.children[-1].value = control_object.states[1]
    assert control_object.state == control_object.states[1]
    assert not np.array_equal(cell.collections[0].get_array(), before)


def test_a_play_button_moves_the_slider():
    figure, time = _figure(TimeAxis)
    _, row = WidgetApplication.create(figure).children
    _, play, slider = row.children
    play.value = 2
    assert slider.value == 2 and time.state == 2


def test_controls_without_a_widget_need_none():
    figure, _ = _figure(FacetAxis)
    assert len(WidgetApplication.create(figure).children) == 1


def test_a_figure_of_another_renderer_raises():
    figure, _ = _figure(SliderAxis, renderer=PlotlyRenderer)
    with pytest.raises(TypeError, match="MatplotlibRenderer"):
        WidgetApplication.create(figure)
