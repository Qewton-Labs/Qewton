"""Scales, axes, titles, colors and figure rules of drawn plots, beyond the
test matrix."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    ColorScale,
    Figure,
    FixedAxis,
    HeatmapPlot,
    LinePlot,
    Overlay,
    PositionAxis,
    Row,
    Scale,
    ScatterPlot,
    Selector,
    SliderAxis,
    TimeAxis,
)

from qewton.visualization.renderers import PlotlyRenderer

from recording_renderer import RecordingRenderer

X, T = Variable("x", 2), Variable("t", 1)
U, P = Variable("u", 1), Variable("p", 1)


def _grid_data(offset=0.0, steps=3):
    batch = BatchAxes(steps)
    config = DataConfiguration(
        batch, GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U * P)
    )
    data = np.random.default_rng(0).random(config.shape) + offset
    return data, config, batch


def _curve_data():
    runs = BatchAxes(2)
    config = DataConfiguration(
        runs, GeometryAxes(GridGeometry.build_unit_grid(T, 5)), FeatureAxes(U * P)
    )
    data = np.random.default_rng(1).random(config.shape) + 0.5
    return data, config, runs


class TestScales:
    def test_a_log_color_scale_hovers_the_values_themselves(self):
        config = DataConfiguration(
            GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U)
        )
        data = np.geomspace(1e-3, 1e3, 16).reshape(config.shape)
        plot = HeatmapPlot(data, config, position=X, color=U, color_scale=ColorScale(log=True))
        (heatmap, _) = Figure(plot, renderer=PlotlyRenderer()).draw().data
        assert np.asarray(heatmap.customdata).min() == pytest.approx(1e-3)
        assert "customdata" in heatmap.hovertemplate

class TestTitles:
    def test_axis_titles_come_from_the_bound_components(self):
        data, config, batch = _grid_data()
        cell = Figure(
            HeatmapPlot(data, config, position=X, color=U, controls=[SliderAxis(batch)])
        ).cells[0]
        assert (cell.axis_title("x"), cell.axis_title("y")) == ("$x_1$", "$x_2$")

    def test_different_variables_on_one_axis_give_no_title_but_legend_entries(self):
        data, config, runs = _curve_data()
        plot = LinePlot(data, config, x=T, y=(U, P), controls=[SliderAxis(runs)])
        cell = Figure(plot).cells[0]
        assert cell.axis_title("x") == "$t$"
        assert cell.axis_title("y") is None
        assert [r.label for r in plot.evaluate()] == ["$u$", "$p$"]

    def test_titles_follow_a_selector(self):
        data, config, batch = _grid_data()
        selector = Selector([U, P])
        plot = HeatmapPlot(
            data, config, position=X, color=selector, controls=[SliderAxis(batch)]
        )
        figure = Figure(plot, renderer=RecordingRenderer())
        backend = figure.draw()
        assert backend["colorbars"][id(plot.color_scale)][0] == "$u$"
        selector.state = 1
        figure.redraw()
        assert backend["colorbars"][id(plot.color_scale)][0] == "$p$"

    def test_a_fixed_axis_is_named_in_the_cell_title(self):
        data, config, batch = _grid_data()
        plot = HeatmapPlot(
            data,
            config,
            position=X,
            color=U,
            controls=[FixedAxis(batch, init_state=2)],
            title="field",
        )
        assert Figure(plot).cells[0].title == "field, batch = 2"


class TestColors:
    def test_plots_and_copies_get_distinct_categorical_colors(self):
        data, config, runs = _curve_data()
        first = LinePlot(data, config, x=T, y=(U, P), controls=[SliderAxis(runs)])
        second = LinePlot(data, config, x=T, y=U, controls=[SliderAxis(runs)])
        Figure(Overlay(first, second))
        colors = [r.color_index for plot in (first, second) for r in plot.evaluate()]
        assert len(set(colors)) == 3


class TestOverlays:
    def test_plots_of_an_overlay_share_its_axes(self):
        data, config, batch = _grid_data()
        heatmap = HeatmapPlot(
            data, config, position=X, color=U, controls=[SliderAxis(batch)]
        )
        points = ScatterPlot(data, config, position=X, controls=[SliderAxis(batch)])
        figure = Figure(Overlay(heatmap, points), renderer=RecordingRenderer())
        backend = figure.draw()
        assert len(figure.cells) == 1
        assert {a.cell.row for a in backend["artists"]} == {1}
        assert figure.cells[0].scaled_channels == ("x", "y")

    def test_plots_of_two_and_three_dimensions_cannot_share_an_overlay(self):
        data, config, batch = _grid_data()
        flat = ScatterPlot(data, config, position=X, controls=[SliderAxis(batch)])
        lifted = ScatterPlot(data, config, x=U, y=P, z=U, controls=[SliderAxis(batch)])
        with pytest.raises(ValueError, match="one dimension"):
            Overlay(flat, lifted)


class TestFigureRules:
    def test_a_figure_takes_one_time_axis(self):
        (d1, c1, b1), (d2, c2, b2) = _grid_data(), _grid_data()
        plots = [
            HeatmapPlot(d, c, position=X, color=U, controls=[TimeAxis(b)])
            for d, c, b in ((d1, c1, b1), (d2, c2, b2))
        ]
        with pytest.raises(ValueError, match="one TimeAxis"):
            Figure(Row(*plots))

    def test_a_dimension_neither_drawn_nor_controlled_is_named(self):
        data, config, batch = _grid_data()
        with pytest.raises(ValueError, match="neither drawn nor reduced"):
            HeatmapPlot(data, config, position=X, color=U)

    def test_a_shared_control_must_fit_every_plot(self):
        (d1, c1, b1), (d2, c2, b2) = _grid_data(steps=3), _grid_data(steps=4)
        slider = SliderAxis(b1)
        HeatmapPlot(d1, c1, position=X, color=U, controls=[slider])
        with pytest.raises(ValueError):
            HeatmapPlot(d2, c2, position=X, color=U, controls=[slider])


def test_data_of_another_backend_is_drawn():
    torch = pytest.importorskip("torch")
    data, config, batch = _grid_data()
    plot = HeatmapPlot(
        torch.tensor(data), config, position=X, color=U, controls=[SliderAxis(batch)]
    )
    # build_unit_grid varies y along the first grid axis: values[iy, ix].
    np.testing.assert_allclose(plot.evaluate()[0].values, data[0, :, :, 0], rtol=1e-6)
