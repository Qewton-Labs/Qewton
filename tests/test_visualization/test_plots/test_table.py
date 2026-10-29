"""Table plots: columns, controls over columns, log and categorical
columns, and the three plot types."""

import numpy as np
import pytest

from qewton.visualization import (
    ColorScale,
    FacetAxis,
    Figure,
    ParallelCoordinatesPlot,
    Selector,
    SliderAxis,
    TableHeatMap,
    TableScatter,
)
from qewton.visualization.plots.result import (
    HeatmapResult,
    MarkerResult,
    ParallelCoordinatesResult,
)
from qewton.visualization.plots.table import table_columns
from qewton.visualization.renderers import PlotlyRenderer

from recording_renderer import RecordingRenderer


def _runs(n=20):
    rng = np.random.default_rng(0)
    return {
        "learning_rate": rng.uniform(1e-4, 1e-1, n),
        "width": rng.choice([8.0, 16.0, 32.0], n),
        "optimizer": rng.choice(["adam", "sgd", "lbfgs"], n),
        "loss": rng.uniform(0.01, 1.0, n),
        "error": rng.uniform(0.1, 5.0, n),
    }


class TestColumns:
    def test_numeric_columns_stay_numeric(self):
        (column,) = table_columns({"a": [1.0, 2.0]}).values()
        assert column.categories is None
        np.testing.assert_allclose(column.values, [1.0, 2.0])

    def test_text_columns_are_coded_in_sorted_order(self):
        (column,) = table_columns({"kind": np.array(["b", "a", "b", "c"])}).values()
        assert column.categories == ["a", "b", "c"]
        assert [column.categories[i] for i in column.values] == ["b", "a", "b", "c"]

    def test_bool_columns_are_coded_false_true(self):
        (column,) = table_columns({"flag": np.array([True, False, True])}).values()
        assert column.categories == ["False", "True"]
        assert column.values.tolist() == [1, 0, 1]

    def test_columns_of_different_length_raise(self):
        with pytest.raises(ValueError, match="differ in length"):
            table_columns({"a": [1.0], "b": [1.0, 2.0]})


class TestChecks:
    def test_a_reference_names_a_column(self):
        with pytest.raises(ValueError, match="no columns"):
            TableScatter(_runs(), x="learning_rate", y="nope")

    def test_a_log_column_is_numeric_and_positive(self):
        with pytest.raises(ValueError, match="numeric and positive"):
            TableScatter(_runs(), x="learning_rate", y="loss", log_columns=["optimizer"])

    def test_a_control_references_a_column(self):
        with pytest.raises(ValueError, match="reference a column"):
            TableScatter(_runs(), x="learning_rate", y="loss", controls=[SliderAxis("nope")])


class TestScatter:
    def test_a_marker_per_row_colored_by_a_column(self):
        runs = _runs()
        plot = TableScatter(runs, x="learning_rate", y="loss", color="error")
        (result,) = plot.evaluate()
        assert isinstance(result, MarkerResult)
        np.testing.assert_allclose(result.positions[:, 0], runs["learning_rate"])
        np.testing.assert_allclose(result.values, runs["error"])
        assert plot.axis_title("x") == "learning_rate"  # column names are plain

    def test_a_log_column_is_drawn_by_log10_with_value_ticks(self):
        runs = _runs()
        plot = TableScatter(runs, x="learning_rate", y="loss", log_columns=["learning_rate"])
        (result,) = plot.evaluate()
        np.testing.assert_allclose(result.positions[:, 0], np.log10(runs["learning_rate"]))
        positions, labels = plot.axis_ticks("x")
        assert 10 ** np.asarray(positions) == pytest.approx([float(t) for t in labels])

    def test_a_categorical_column_is_drawn_by_codes_with_category_ticks(self):
        plot = TableScatter(_runs(), x="optimizer", y="loss")
        assert plot.axis_ticks("x") == ([0, 1, 2], ["adam", "lbfgs", "sgd"])

    def test_jitter_is_fixed_per_row(self):
        plot = TableScatter(_runs(), x="width", y="loss", jitter=0.05)
        (first,) = plot.evaluate()
        (second,) = plot.evaluate()
        np.testing.assert_array_equal(first.positions, second.positions)
        assert len(np.unique(first.positions[:, 0])) > 3

    def test_selectors_choose_columns_and_their_ticks(self):
        x = Selector(["learning_rate", "optimizer"])
        plot = TableScatter(_runs(), x=x, y="loss", log_columns=["learning_rate"])
        backend = Figure(plot, renderer=PlotlyRenderer()).draw(embedded=[x])
        (menu,) = backend.layout.updatemenus
        assert [b.label for b in menu.buttons] == ["learning_rate", "optimizer"]
        x.state = 1
        assert plot.axis_ticks("x")[1] == ["adam", "lbfgs", "sgd"]

    def test_a_log_color_column_gets_a_log_color_scale(self):
        plot = TableScatter(_runs(), x="learning_rate", y="loss", color="error", log_columns=["error"])
        assert plot.color_scale.log


class TestControls:
    def test_a_control_draws_the_rows_of_one_value(self):
        runs = _runs()
        optimizer = SliderAxis("optimizer")
        plot = TableScatter(runs, x="learning_rate", y="loss", controls=[optimizer])
        assert [optimizer.label(s) for s in optimizer.states] == ["adam", "lbfgs", "sgd"]
        optimizer.state = 2
        (result,) = plot.evaluate()
        assert len(result.positions) == np.sum(runs["optimizer"] == "sgd")

    def test_a_numeric_control_is_labeled_by_its_values(self):
        width = SliderAxis("width")
        TableScatter(_runs(), x="learning_rate", y="loss", controls=[width])
        assert [width.label(s) for s in width.states] == ["8", "16", "32"]


class TestHeatMap:
    def test_the_mean_over_each_rectangle(self):
        table = {"a": [0.0, 0.0, 1.0], "b": [0.0, 0.0, 1.0], "c": [1.0, 3.0, 5.0]}
        plot = TableHeatMap(table, x="a", y="b", color="c", bins_x=2, bins_y=2)
        (result,) = plot.evaluate()
        assert isinstance(result, HeatmapResult)
        assert result.values[0, 0] == pytest.approx(2.0)
        assert result.values[1, 1] == pytest.approx(5.0)
        assert np.isnan(result.values[0, 1])

    def test_categorical_columns_cannot_be_rastered(self):
        with pytest.raises(ValueError, match="numeric columns only"):
            TableHeatMap(_runs(), x="optimizer", y="loss", color="error")


class TestParallelCoordinates:
    def test_one_dimension_per_column_in_order(self):
        plot = ParallelCoordinatesPlot(_runs(), columns=["loss", "learning_rate", "optimizer"])
        (result,) = plot.evaluate()
        assert isinstance(result, ParallelCoordinatesResult)
        assert [d.title for d in result.dimensions] == ["loss", "learning_rate", "optimizer"]
        assert result.dimensions[2].ticks == ([0, 1, 2], ["adam", "lbfgs", "sgd"])
        assert plot.dim == 0

    def test_a_linear_color_is_not_transformed(self):
        runs = _runs()
        plot = ParallelCoordinatesPlot(runs, columns=["loss"], color="loss")
        (trace,) = Figure(plot, renderer=PlotlyRenderer()).draw().data
        np.testing.assert_allclose(trace.line.color, runs["loss"])
