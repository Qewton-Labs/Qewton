import numpy as np
import pytest

from qewton.visualization.figure import Figure
from qewton.visualization.plots.spec import ColorSpec, FacetSpec, Scale
from qewton.visualization.plots.table.base import TablePlot
from qewton.visualization.plots.table.parallel_coordinates import ParallelCoordinatesPlot


def _tuning_columns():
    rng = np.random.default_rng(0)
    n = 20
    return {
        "learning_rate": rng.uniform(1e-4, 1e-1, n),
        "optimizer": rng.choice(["adam", "sgd", "lbfgs"], n),
        "loss": rng.uniform(0.01, 1.0, n),
    }


class TestTablePlotNormalization:
    def test_numeric_columns_stay_numeric(self):
        plot = TablePlot({"a": np.array([1.0, 2.0, 3.0])})
        assert plot.columns["a"].labels is None
        assert np.allclose(plot.columns["a"].values, [1.0, 2.0, 3.0])

    def test_categorical_columns_are_coded_to_integers_with_labels_preserved(self):
        plot = TablePlot({"kind": np.array(["b", "a", "b", "c"])})
        column = plot.columns["kind"]
        assert column.labels == ["a", "b", "c"]  # sorted unique
        # codes must round-trip back to the original strings via labels
        decoded = [column.labels[i] for i in column.values]
        assert decoded == ["b", "a", "b", "c"]

    def test_n_rows_matches_column_length(self):
        plot = TablePlot(_tuning_columns())
        assert plot.n_rows == 20

    def test_apply_controls_filters_rows_by_column_value(self):
        columns = _tuning_columns()
        facet = FacetSpec("optimizer")
        plot = TablePlot(columns, controls=[facet])
        assert set(facet.values) == {0, 1, 2}  # 3 coded optimizer categories
        facet.state = facet.values[0]
        rows = plot.apply_controls()
        expected_mask = plot.columns["optimizer"].values == facet.values[0]
        assert len(rows["learning_rate"]) == expected_mask.sum()


class TestParallelCoordinatesPlot:
    def test_evaluate_returns_requested_axes_in_order(self):
        columns = _tuning_columns()
        plot = ParallelCoordinatesPlot(
            columns, axes=["learning_rate", "optimizer", "loss"], color=ColorSpec("loss")
        )
        result = plot.evaluate()
        assert list(result.columns.keys()) == ["learning_rate", "optimizer", "loss"]
        assert result.columns["optimizer"].labels == ["adam", "lbfgs", "sgd"]
        assert result.color.shape == (20,)

    def test_missing_axis_column_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="not found in columns"):
            ParallelCoordinatesPlot(_tuning_columns(), axes=["nope"])

    def test_missing_color_column_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="not found in columns"):
            ParallelCoordinatesPlot(_tuning_columns(), axes=["loss"], color=ColorSpec("nope"))

    def test_embedding_dim_is_none_like_other_non_cartesian_plots(self):
        plot = ParallelCoordinatesPlot(_tuning_columns(), axes=["loss"])
        assert plot.embedding_dim is None

    def test_top_k_filtering_is_upstream_not_a_plot_feature(self):
        """Per the plan's node-layer criterion: selecting a row subset
        produces a different dataset, so it's the caller's job, not a
        constructor argument on the plot."""
        columns = _tuning_columns()
        top_5 = np.argsort(columns["loss"])[:5]
        filtered = {k: np.asarray(v)[top_5] for k, v in columns.items()}
        plot = ParallelCoordinatesPlot(filtered, axes=["learning_rate", "loss"])
        assert plot.n_rows == 5

    def test_shared_scale_trains_through_the_generic_color_values_path(self):
        """No TablePlot-specific code should be needed in Figure.draw() for
        this to work - color_values() is family-neutral."""
        scale = Scale()
        plot = ParallelCoordinatesPlot(
            _tuning_columns(), axes=["loss"], color=ColorSpec("loss", scale=scale)
        )
        Figure(plot).draw()
        assert scale.range is not None

    def test_facet_over_a_column_renders_one_trace_per_category(self):
        columns = _tuning_columns()
        facet = FacetSpec("optimizer", orientation="col")
        plot = ParallelCoordinatesPlot(columns, axes=["learning_rate", "loss"], controls=[facet])
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.data) == 3
        assert all(trace.type == "parcoords" for trace in backend_figure.data)

    def test_draws_as_parcoords(self):
        plot = ParallelCoordinatesPlot(_tuning_columns(), axes=["learning_rate", "loss"])
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "parcoords"
