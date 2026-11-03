"""Test matrix of the table plots: plot × drawn column × control × output.

A drawn column is numeric, logarithmic, categorical or boolean; a
control steps through the values of a categorical or a numeric column, or
a Selector chooses a column. RAISES gives the pairs that do not fit with
their reason, NOT_DRAWN the plots a renderer does not draw; every other
combination draws, follows its control and embeds it.
"""

import warnings

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.collections import QuadMesh
from plotly.colors import sample_colorscale, unlabel_rgb

from qewton.visualization import (
    DashApplication,
    WidgetApplication,
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
from qewton.visualization.renderers import MatplotlibRenderer, PlotlyRenderer
from qewton.visualization.renderers.matplotlib.primitives import colorbar_axes
from qewton.visualization.themes import LIGHT_THEME

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

#: (plot, renderer) -> why the renderer does not draw the plot.
NOT_DRAWN = {
    ("parallel", "matplotlib"): "Matplotlib has no native parallel coordinates",
}

#: (renderer, control) -> why a self-contained output does not embed the
#: control; it keeps its current state, with a warning.
NOT_EMBEDDED = {
    ("matplotlib", "selector_x"): "Matplotlib embeds one control axis as frames",
    ("matplotlib", "selector_color"): "Matplotlib embeds one control axis as frames",
}

#: The application showing each renderer's figures as a connected output.
APPLICATIONS = {"plotly": DashApplication, "matplotlib": WidgetApplication}

OUTPUTS = ["draw", "self_contained", "connected"]
RENDERERS = {
    "plotly": PlotlyRenderer,
    "matplotlib": MatplotlibRenderer,
    "recording": RecordingRenderer,
}


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
        pytest.skip("an application shows the figures of a real renderer")
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
    if (plot_name, renderer_name) in NOT_DRAWN:
        with pytest.raises(ValueError, match="cannot draw"):
            figure.draw()
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if output == "draw":
            figure.draw()
        elif output == "self_contained":
            figure.save_html(tmp_path / "table.html")
        else:
            APPLICATIONS[renderer_name].create(figure)
    warned = [w for w in caught if issubclass(w.category, UserWarning)]
    not_embedded = output == "self_contained" and (renderer_name, control) in NOT_EMBEDDED
    assert len(warned) == (1 if not_embedded else 0)


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


def _colorbar_labels(backend, renderer_name) -> list[str]:
    if renderer_name == "plotly":
        return list(backend.data[-1].marker.colorbar.ticktext)
    (colorbar,) = colorbar_axes(backend)
    return [t.get_text() for t in colorbar.get_yticklabels()]


def _colorbar_colors(backend, renderer_name, codes) -> list[tuple]:
    # The RGB colors of the colorbar at the given codes.
    if renderer_name == "plotly":
        marker = backend.data[-1].marker
        positions = [(c - marker.cmin) / (marker.cmax - marker.cmin) for c in codes]
        colors = sample_colorscale([list(s) for s in marker.colorscale], positions)
        return [tuple(round(v / 255, 3) for v in unlabel_rgb(c)) for c in colors]
    (colorbar,) = colorbar_axes(backend)
    (solids,) = [c for c in colorbar.collections if isinstance(c, QuadMesh)]
    rgba = solids.cmap(solids.norm(np.asarray(codes, dtype=float)))
    return [tuple(round(float(v), 3) for v in c[:3]) for c in rgba]


@pytest.mark.parametrize("renderer_name", ["plotly", "matplotlib"])
def test_a_categorical_color_column_gets_a_band_per_category(renderer_name):
    # Labeled by name, in the palette's colors in the order plots are
    # colored; the band edges lie between the codes.
    plot = TableScatter(_runs(), x="lr", y="loss", color="optimizer")
    backend = Figure(plot, renderer=RENDERERS[renderer_name]()).draw()
    assert plot.color_scale.categorical
    assert _colorbar_labels(backend, renderer_name) == ["adam", "lbfgs", "sgd"]
    palette = [
        tuple(round(v, 3) for v in to_rgb(c)) for c in LIGHT_THEME.primary_color_palette[:3]
    ]
    assert _colorbar_colors(backend, renderer_name, [0, 1, 2]) == palette
    assert _colorbar_colors(backend, renderer_name, [0.45, 1.45]) == palette[:2]
