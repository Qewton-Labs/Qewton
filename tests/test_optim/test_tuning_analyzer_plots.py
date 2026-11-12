"""The plots of a TuningAnalyzer: one table plot each, with the
parameters and objectives to choose from."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from qewton.optim.tuner.results.analyzer import TuningAnalyzer  # noqa: E402
from qewton.optim.tuner.results.tune_results import TuneResultCollector  # noqa: E402
from qewton.visualization import (
    FacetAxis,
    ParallelCoordinatesPlot,
    Selector,
    TableHeatMap,
    TableScatter,
)


@pytest.fixture
def analyzer():
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame(
        {
            "lr": rng.uniform(1e-4, 1e-1, n),
            "width": rng.choice([8, 16, 32], n),
            "optimizer": rng.choice(["adam", "sgd"], n),
            "loss": rng.uniform(0.01, 1.0, n),
            "error": rng.uniform(0.1, 2.0, n),
        }
    )
    meta = {
        TuneResultCollector.objective_names_key: ["loss", "error"],
        TuneResultCollector.objective_values_key: ["min", "min"],
        TuneResultCollector.hp_key: ["lr", "width", "optimizer"],
    }
    return TuningAnalyzer.from_df(df, meta)


def _plot(figure):
    (plot,) = figure.plots
    return plot


def test_parallel_coordinates_choose_between_the_objectives(analyzer):
    plot = _plot(analyzer.parallel_coordinates_plot(log_axes=["lr"]))
    assert isinstance(plot, ParallelCoordinatesPlot)
    assert plot.column_names == ["lr", "width", "optimizer", "loss", "error"]
    (selector,) = plot.selectors
    assert selector.candidates == ["loss", "error"]
    figure = analyzer.parallel_coordinates_plot()
    figure.draw()


def test_parallel_coordinates_take_controls_over_columns(analyzer):
    figure = analyzer.parallel_coordinates_plot(controls=[FacetAxis("optimizer")])
    assert len(figure.cells) == 2


def test_scatter_chooses_x_y_and_color(analyzer):
    plot = _plot(analyzer.scatter_table_plot(jitter=0.02))
    assert isinstance(plot, TableScatter)
    x, y, color = plot.selectors
    assert x.selected == "lr" and y.selected == "width"
    assert color.candidates == ["loss", "error", "combined_score"]


def test_heatmap_uses_numeric_parameters_only(analyzer):
    plot = _plot(analyzer.heatmap_plot(4, 4))
    assert isinstance(plot, TableHeatMap)
    assert plot.selectors[0].candidates == ["lr", "width"]
    with pytest.raises(ValueError, match="numeric"):
        analyzer.heatmap_plot(4, 4, axis_keys=["optimizer", "lr"])
