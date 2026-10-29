"""Test matrix of the table plots: plot × drawn column × control × output.

A drawn column is numeric, logarithmic, categorical or boolean; a
control steps through the values of a categorical or a numeric column, or
a Selector chooses a column. TABLE gives the pairs that do not fit with
their reason; every other combination draws, follows its control and
embeds it.
"""

import warnings

import numpy as np
import pytest

from qewton.visualization import (
    DashApplication,
    FacetAxis,
    Figure,
    FixedAxis,
    ParallelCoordinatesPlot,
    Selector,
    SliderAxis,
    TableHeatMap,
    TableScatter,
    TimeAxis,
)
from qewton.visualization.plots.result import (
    HeatmapResult,
    MarkerResult,
    ParallelCoordinatesResult,
)
from qewton.visualization.renderers import PlotlyRenderer

from recording_renderer import RecordingRenderer

N_ROWS = 24


def _runs():
    rng = np.random.default_rng(0)
    return {
        "lr": rng.uniform(1e-4, 1e-1, N_ROWS),
        "width": np.tile([8.0, 16.0, 32.0], N_ROWS // 3),
        "optimizer": np.tile(["adam", "sgd", "lbfgs", "adam"], N_ROWS // 4),
        "flag": np.tile([True, False], N_ROWS // 2),
        "loss": rng.uniform(0.01, 1.0, N_ROWS),
        "error": rng.uniform(0.1, 5.0, N_ROWS),
    }


#: The drawn column of each kind, and the log columns that go with it.
COLUMNS = {
    "numeric": ("width", []),
    "log": ("lr", ["lr", "error"]),
    "categorical": ("optimizer", []),
    "boolean": ("flag", []),
}

#: Control name: (control factory, number of states, cells).
CONTROLS = {
    "none": (lambda: None, None, 1),
    "slider_categorical": (lambda: SliderAxis("optimizer"), 3, 1),
    "slider_numeric": (lambda: SliderAxis("width"), 3, 1),
    "time": (lambda: TimeAxis("width"), 3, 1),
    "facet": (lambda: FacetAxis("optimizer"), 3, 3),
    "fixed": (lambda: FixedAxis("optimizer", init_state=1), None, 1),
    "selector_x": (None, 2, 1),
    "selector_color": (None, 2, 1),
}

PLOTS = {
    "scatter": MarkerResult,
    "heatmap": HeatmapResult,
    "parallel": ParallelCoordinatesResult,
}

#: (plot, column kind or control) -> why the combination raises.
RAISES = {
    ("heatmap", "categorical"): "a raster needs numeric columns",
    ("heatmap", "boolean"): "a raster needs numeric columns",
}

#: (plot, control) -> why the combination is not built.
NOT_APPLICABLE = {
    ("parallel", "selector_x"): "parallel coordinates draw every listed column",
}

OUTPUTS = ["draw", "self_contained", "connected"]
RENDERERS = {"plotly": PlotlyRenderer, "recording": RecordingRenderer}


def build(plot: str, column_kind: str, control: str):
    """The plot of a matrix cell and its control (or Selector)."""
    name, log_columns = COLUMNS[column_kind]
    factory = CONTROLS[control][0]
    control_object = factory() if factory is not None else None
    x = Selector([name, "loss"]) if control == "selector_x" else name
    color = Selector(["error", "loss"]) if control == "selector_color" else "error"
    selector = x if control == "selector_x" else color if control == "selector_color" else None
    common = dict(
        color=color,
        log_columns=log_columns,
        controls=[control_object] if control_object is not None else [],
    )
    if plot == "scatter":
        made = TableScatter(_runs(), x=x, y="loss", jitter=0.02, **common)
    elif plot == "heatmap":
        made = TableHeatMap(_runs(), x=x, y="loss", bins_x=3, bins_y=3, **common)
    else:
        made = ParallelCoordinatesPlot(_runs(), columns=[name, "loss", "error"], **common)
    return made, control_object or selector


@pytest.mark.parametrize("renderer_name", list(RENDERERS))
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("control", list(CONTROLS))
@pytest.mark.parametrize("column_kind", list(COLUMNS))
@pytest.mark.parametrize("plot_name", list(PLOTS))
def test_table_matrix(plot_name, column_kind, control, output, renderer_name, tmp_path):
    if output == "connected" and renderer_name == "recording":
        pytest.skip("a connected output (Dash) shows Plotly figures")
    if (plot_name, control) in NOT_APPLICABLE:
        pytest.skip(NOT_APPLICABLE[(plot_name, control)])
    if (plot_name, column_kind) in RAISES:
        with pytest.raises(ValueError):
            build(plot_name, column_kind, control)
        return

    plot, control_object = build(plot_name, column_kind, control)
    results = plot.evaluate()
    assert all(isinstance(r, PLOTS[plot_name]) for r in results)
    n_states, n_cells = CONTROLS[control][1:]
    if n_states is not None and not isinstance(control_object, FacetAxis):
        _check_follows_state(plot, control_object)

    figure = Figure(plot, renderer=RENDERERS[renderer_name]())
    assert len(figure.cells) == n_cells
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if output == "draw":
            figure.draw()
        elif output == "self_contained":
            figure.save_html(tmp_path / "table.html")
        else:
            DashApplication.create(figure)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def _check_follows_state(plot, control_object):
    # The drawn data changes with the state of the control.
    snapshots = []
    for state in control_object.states[:2]:
        control_object.state = state
        (result,) = plot.evaluate()
        snapshots.append(_drawn(result))
    control_object.state = control_object.states[0]
    assert snapshots[0].shape != snapshots[1].shape or not np.allclose(
        *snapshots, equal_nan=True
    )


def _drawn(result) -> np.ndarray:
    if isinstance(result, ParallelCoordinatesResult):
        arrays = [d.values for d in result.dimensions]
        if result.values is not None:
            arrays.append(result.values)
        return np.concatenate(arrays)
    if isinstance(result, HeatmapResult):
        return result.values.ravel()
    arrays = [result.positions.ravel()]
    if result.values is not None:
        arrays.append(result.values)
    return np.concatenate(arrays)
