"""Test matrix over case × plot × control × variant.

A variant is an output (draw, self_contained, connected) with a renderer
and the backend of the data. Every output runs with numpy data on the
recording and the Plotly renderer; drawing also runs with torch data,
with Plotly without MathJax (texts as Plotly markup), with the dark
theme (the chrome of every cell follows the theme) and with Matplotlib
(rendered to an image, without warnings); self-contained outputs also run
with Matplotlib, which embeds controls as frames (NOT_EMBEDDED: the
controls it leaves in their current state), and connected outputs with
the application of each renderer (APPLICATIONS).

A supported combination (MODEL in matrix_cases.py) evaluates to the
expected Results, draws one artist per Result and cell, follows its
control in every output and realizes it (or warns); an excluded one raises
a ValueError at construction. Known defects are strict xfails in
KNOWN_GAPS.
"""

import dataclasses
import io
import re
import warnings

import matplotlib.colors

import numpy as np
import pytest

from qewton.visualization import DashApplication, Figure, WidgetApplication
from qewton.visualization.renderers import MatplotlibRenderer, PlotlyRenderer
from qewton.visualization.themes import DARK_THEME

from matrix_cases import (
    CASES,
    CONTROLS,
    PLOTS,
    EXCLUSIONS,
    NotApplicable,
    build,
    expected,
)
from recording_renderer import RecordingRenderer

OUTPUTS = ["draw", "self_contained", "connected"]
RENDERERS = {
    "plotly": PlotlyRenderer,
    "plotly_markup": lambda: PlotlyRenderer(mathjax=False),
    "plotly_dark": lambda: PlotlyRenderer(theme=DARK_THEME),
    "recording": RecordingRenderer,
    "matplotlib": MatplotlibRenderer,
    "matplotlib_dark": lambda: MatplotlibRenderer(theme=DARK_THEME),
}

#: (output, renderer, backend) of every variant.
VARIANTS = [
    *((output, renderer, "numpy") for output in OUTPUTS for renderer in ("plotly", "recording")),
    ("draw", "plotly_markup", "numpy"),
    ("draw", "plotly_dark", "numpy"),
    ("draw", "matplotlib", "numpy"),
    ("self_contained", "matplotlib", "numpy"),
    ("connected", "matplotlib", "numpy"),
    ("draw", "matplotlib_dark", "numpy"),
    ("draw", "plotly", "torch"),
    ("draw", "recording", "torch"),
]

#: The application showing each renderer's figures as a connected output.
APPLICATIONS = {"plotly": DashApplication, "matplotlib": WidgetApplication}

#: (renderer, control) -> why a self-contained output does not embed the
#: control; it keeps its current state, with a warning naming it.
NOT_EMBEDDED = {
    ("matplotlib", "selector"): "Matplotlib embeds one control axis as frames",
}

#: (case, plot, control, output, renderer) -> reason; None matches any.
KNOWN_GAPS: dict[tuple, str] = {}


def _gap(*key):
    for pattern, reason in KNOWN_GAPS.items():
        if all(p is None or p == k for p, k in zip(pattern, key)):
            return reason
    return None


def _arrays(result) -> list[np.ndarray]:
    # The float arrays of a Result.
    arrays = []
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            arrays.append(value)
    return arrays


def _n_copies(control: str) -> int:
    return {"overlay": 3, "tuple": 2}.get(control, 1)


def _n_cells(control: str) -> int:
    return 3 if control == "facet" else 1


def _check_results(plot, result_class, control):
    results = plot.evaluate()
    assert len(results) == _n_copies(control)
    for result in results:
        assert isinstance(result, result_class)
        for array in _arrays(result):
            assert np.all(
                np.isfinite(array)
            ), f"{type(result).__name__} has non-finite values"
    labels = [r.label for r in results]
    assert len(set(labels)) == len(labels), f"copies share a legend entry: {labels}"


def _check_follows_state(plot, control_object):
    # The drawn data changes with the state of the control.
    original = control_object.state
    snapshots = []
    for state in control_object.states[:2]:
        control_object.state = state
        snapshots.append(
            np.concatenate([a.ravel() for r in plot.evaluate() for a in _arrays(r)])
        )
    control_object.state = original
    assert snapshots[0].shape != snapshots[1].shape or not np.allclose(*snapshots)


def _check_drawn(figure, renderer_name, control):
    if renderer_name.startswith("matplotlib"):
        _check_drawn_matplotlib(figure, renderer_name, control)
        return
    backend = figure.draw()
    if renderer_name == "recording":
        assert len(backend["artists"]) == _n_copies(control) * _n_cells(control)
        assert len(backend["axes"]) == _n_cells(control)
    else:
        assert len(backend.data) > 0
    if renderer_name == "plotly_markup":
        assert "$" not in str(backend.layout.to_plotly_json()), "math left in a text"
    if renderer_name == "plotly_dark":
        _check_theme(figure.backend_figure, DARK_THEME)
    assert len(figure.cells) == _n_cells(control)


def _check_drawn_matplotlib(figure, renderer_name, control):
    # Every cell has Axes with something drawn; the figure renders without
    # warnings (e.g. math mathtext cannot parse).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = figure.draw()
        figure.renderer.save_image(io.BytesIO(), "png")
    assert not [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    cells = [a for a in backend.axes if a.get_label().startswith("cell-")]
    assert len(cells) == _n_cells(control) == len(figure.cells)
    assert all(a.collections or a.lines or a.patches or a.images for a in cells)
    if renderer_name == "matplotlib_dark":
        assert matplotlib.colors.same_color(
            backend.get_facecolor(), DARK_THEME.background_color
        )


def _check_theme(backend, theme):
    # The chrome of the figure and of every cell comes from the theme.
    layout = backend.layout
    assert layout.paper_bgcolor == theme.background_color
    assert layout.font.color == theme.text_color
    for axis in [*backend.select_xaxes(), *backend.select_yaxes()]:
        assert axis.gridcolor == theme.grid_color
    for scene in backend.select_scenes():
        assert scene.xaxis.backgroundcolor == theme.background_color


def _check_self_contained(figure, renderer_name, control, control_object, path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.save_html(path)
    backend = figure.backend_figure
    embedded = {"slider": 3, "time": 3, "selector": 2}.get(control)
    if embedded is None:
        return
    warned = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    if (renderer_name, control) in NOT_EMBEDDED:
        assert len(warned) == 1 and control_object.name in warned[0]
        return
    if renderer_name == "matplotlib":
        frames = re.findall(r'frames\[\d+\] = "data', path.read_text())
        assert len(frames) == embedded
    elif renderer_name == "recording":
        (snapshots,) = backend["embedded"].values()
        assert len(snapshots) == embedded
    elif control == "time":
        assert len(backend.frames) == embedded
    elif control == "slider":
        assert len(backend.layout.sliders[-1].steps) == embedded
    else:
        assert len(backend.layout.updatemenus[-1].buttons) == embedded
    assert not warned


def _check_connected(figure, renderer_name, control, control_object):
    # A control of a kind the application connects gets a widget and is
    # followed; Dash embeds a TimeAxis as an animation instead.
    application = APPLICATIONS[renderer_name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        application.create(figure)
        widgets = application.interactive_controls(figure)
        if control in ("slider", "time", "selector"):
            if type(control_object) in application.connected:
                assert widgets == [control_object]
                for state in control_object.states:
                    application.update(figure, widgets, [state])
            else:
                assert widgets == []
                assert len(figure.backend_figure.frames) == len(control_object.states)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize(
    "output, renderer_name, backend", VARIANTS, ids=["-".join(v) for v in VARIANTS]
)
@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("plot_name", PLOTS)
@pytest.mark.parametrize("case_name", list(CASES))
def test_matrix(case_name, plot_name, control, output, renderer_name, backend, tmp_path, request):
    if output == "connected" and renderer_name == "recording":
        pytest.skip("an application shows the figures of a real renderer")
    case = CASES[case_name]
    result_class = expected(plot_name, case_name)
    gap = _gap(case_name, plot_name, control, output, renderer_name)
    if gap is not None and result_class is not None:
        request.applymarker(pytest.mark.xfail(reason=gap, strict=True))

    try:
        if result_class is None:
            with pytest.raises(ValueError):
                build(plot_name, case, control, backend)
            return
        plot, control_object = build(plot_name, case, control, backend)
    except NotApplicable as reason:
        pytest.skip(str(reason))

    _check_results(plot, result_class, control)
    if control in ("slider", "time", "selector"):
        _check_follows_state(plot, control_object)

    figure = Figure(plot, renderer=RENDERERS[renderer_name]())
    if control == "fixed":
        assert "batch = 1" in figure.cells[0].title
    if output == "draw":
        _check_drawn(figure, renderer_name, control)
    elif output == "self_contained":
        _check_self_contained(
            figure, renderer_name, control, control_object, tmp_path / "figure.html"
        )
    else:
        _check_connected(figure, renderer_name, control, control_object)


def test_every_exclusion_has_a_reason():
    for plot in PLOTS:
        if any(expected(plot, case) is None for case in CASES):
            assert plot in EXCLUSIONS
