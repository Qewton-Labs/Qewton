"""Embedded and connected controls, alone and in pairs.

An embedded output must show, after any sequence of widget actions, what
``figure.draw()`` shows in the resulting states. The Plotly updates of the
embedded widgets (slider steps, dropdown buttons, animation frames) are
replayed on the figure's JSON by `_Replay`, as plotly.js applies them.
"""

import base64
import copy
import itertools
import json
import warnings

import numpy as np
import pytest
from plotly.utils import PlotlyJSONEncoder

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.visualization import (
    ColorScale,
    DashApplication,
    FacetAxis,
    Figure,
    FixedAxis,
    HeatmapPlot,
    LinePlot,
    Overlay,
    Row,
    ScatterPlot,
    Selector,
    SliderAxis,
    TimeAxis,
)
from qewton.visualization.renderers import PlotlyRenderer
from qewton.visualization.themes.base import Theme

from recording_renderer import RecordingRenderer

X, T = Variable("x", 2), Variable("t", 1)
U, P, Q = Variable("u", 1), Variable("p", 1), Variable("q", 1)
_ENCODE = PlotlyJSONEncoder().encode


def _grid_data(*steps, seed=0, offsets=(0.0, 10.0, 20.0)):
    batches = [BatchAxes(n) for n in steps]
    config = DataConfiguration(
        *batches, GeometryAxes(GridGeometry.build_unit_grid(X, 4)), FeatureAxes(U * P * Q)
    )
    data = np.random.default_rng(seed).random(config.shape) + np.asarray(offsets)
    return data, config, batches


def _curve_data(*steps, seed=1):
    batches = [BatchAxes(n) for n in steps]
    config = DataConfiguration(
        *batches, GeometryAxes(GridGeometry.build_unit_grid(T, 5)), FeatureAxes(U * P * Q)
    )
    data = np.random.default_rng(seed).random(config.shape) + 0.5
    return data, config, batches


# --- replaying embedded widgets -------------------------------------------------


def _set_path(node: dict, path: str, value) -> None:
    # As plotly.js restyle/relayout: None removes the property.
    *parents, leaf = path.split(".")
    for parent in parents:
        if value is None and parent not in node:
            return
        node = node.setdefault(parent, {})
    if value is None:
        node.pop(leaf, None)
    else:
        node[leaf] = value


def _merge(node: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(node.get(key), dict):
            _merge(node[key], value)
        else:
            node[key] = value


class _Replay:
    """The JSON of an embedded Plotly figure, changed by widget actions."""

    def __init__(self, figure: Figure):
        self.json = copy.deepcopy(figure.backend_figure.to_plotly_json())
        self.frames = {f["name"]: f for f in self.json.get("frames", [])}

    def _menu(self, kind: str, control, entries: str) -> list:
        for menu in self.json["layout"].get(kind, []):
            if menu.get("name", "").endswith(f"-{id(control)}"):
                return menu[entries]
        raise KeyError(f"{control.name} is not embedded")

    def _update(self, args) -> None:
        restyle, relayout, indices = args
        for key, values in restyle.items():
            for index, value in zip(indices, values):
                _set_path(self.json["data"][index], key, value)
        for key, value in relayout.items():
            _set_path(self.json["layout"], key, value)

    def act(self, control, state) -> None:
        if isinstance(control, TimeAxis):
            frame = self.frames[str(state)]
            for index, trace in zip(frame["traces"], frame["data"]):
                _merge(self.json["data"][index], trace)
            _merge(self.json["layout"], frame.get("layout", {}))
        elif isinstance(control, Selector):
            self._update(self._menu("updatemenus", control, "buttons")[state]["args"])
        else:
            self._update(self._menu("sliders", control, "steps")[state]["args"])


def _plain(value):
    # JSON values with arrays as lists, however Plotly encoded them.
    if isinstance(value, dict):
        if set(value) == {"dtype", "bdata"}:
            return np.frombuffer(
                base64.b64decode(value["bdata"]), value["dtype"]
            ).tolist()
        if set(value) == {"dtype", "bdata", "shape"}:
            flat = np.frombuffer(base64.b64decode(value["bdata"]), value["dtype"])
            shape = [int(n) for n in str(value["shape"]).split(",")]
            return flat.reshape(shape).tolist()
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _encode(value) -> str:
    return json.dumps(_plain(json.loads(_ENCODE(value))), sort_keys=True)


def _shown(figure_json: dict) -> tuple[list[str], dict]:
    # The visible traces (as JSON, in any order) and the axis layout.
    traces = sorted(
        _encode({k: v for k, v in t.items() if k not in ("visible", "uid")})
        for t in figure_json["data"]
        if t.get("visible", True) is not False
    )
    axes = {
        k: _encode(v)
        for k, v in figure_json["layout"].items()
        if k.startswith(("xaxis", "yaxis", "scene"))
    }
    return traces, axes


def _check_states(figure: Figure, embedded, actions) -> None:
    # After `actions` on the embedded output, it shows what draw() shows in
    # the resulting states.
    replay = _Replay(figure)
    for control, state in actions:
        replay.act(control, state)
    originals = [c.state for c in embedded]
    for control, state in actions:
        control.state = state
    expected = _shown(figure.draw().to_plotly_json())
    for control, state in zip(embedded, originals):
        control.state = state
    figure.draw(embedded=embedded)
    assert _shown(replay.json) == expected


def _check_every_combination(figure: Figure, controls) -> None:
    for states in itertools.product(*(c.states for c in controls)):
        for order in itertools.permutations(range(len(controls))):
            actions = [(controls[i], states[i]) for i in order]
            _check_states(figure, controls, actions)


# --- pairs of controls ----------------------------------------------------------


def _one_plot(pair):
    # One plot carrying the controls of `pair`, control axes first.
    kinds = [k for k in pair if k != "selector"]
    data, config, batches = _grid_data(*[3, 2][: len(kinds)])
    axis_kinds = {"slider": SliderAxis, "time": TimeAxis}
    controls = [axis_kinds[k](b) for k, b in zip(kinds, batches)]
    if pair.count("selector") == 2:
        x, color = Selector([U, P]), Selector([P, Q])
        plot = ScatterPlot(data, config, x=x, y=Q, color=color, controls=controls)
        return plot, controls + [x, color]
    if "selector" in pair:
        selector = Selector([U, P])
        plot = HeatmapPlot(data, config, position=X, color=selector, controls=controls)
        return plot, controls + [selector]
    plot = HeatmapPlot(data, config, position=X, color=U, controls=controls)
    return plot, controls


def _two_plots(pair):
    # Two plots with one control of `pair` each.
    plots, controls = [], []
    for seed, kind in enumerate(pair):
        data, config, (batch,) = _grid_data(3, seed=seed)
        if kind == "selector":
            selector = Selector([U, P])
            plot = HeatmapPlot(
                data, config, position=X, color=selector, controls=[FixedAxis(batch)]
            )
            controls.append(selector)
        else:
            control = {"slider": SliderAxis, "time": TimeAxis}[kind](batch)
            plot = HeatmapPlot(data, config, position=X, color=U, controls=[control])
            controls.append(control)
        plots.append(plot)
    return Row(*plots), controls


ONE_PLOT_CONSISTENT = [
    ("slider", "selector"),
    ("time", "selector"),
    ("selector", "selector"),  # different channels, each embedded by its data
]
ONE_PLOT_OVERWRITING = [("time", "slider"), ("slider", "selector", "selector")]
TWO_PLOTS = [
    ("slider", "slider"),
    ("slider", "time"),
    ("slider", "selector"),
    ("time", "selector"),
    ("selector", "selector"),
]


@pytest.mark.parametrize("pair", ONE_PLOT_CONSISTENT)
def test_two_controls_of_one_plot_combine(pair):
    plot, controls = _one_plot(pair)
    figure = Figure(plot)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure.draw(embedded=controls)
    _check_every_combination(figure, controls)


@pytest.mark.parametrize("pair", ONE_PLOT_OVERWRITING)
def test_controls_overwriting_each_other_on_one_plot_embed_the_first(pair):
    plot, controls = _one_plot(pair)
    figure = Figure(plot)
    with pytest.warns(UserWarning, match="cannot be embedded together"):
        figure.draw(embedded=controls)
    first, *_, skipped = controls
    replay = _Replay(figure)
    replay.act(first, 1)
    with pytest.raises(KeyError, match="not embedded"):
        replay.act(skipped, 1)


@pytest.mark.parametrize("pair", TWO_PLOTS)
def test_controls_of_two_plots_combine(pair):
    layout, controls = _two_plots(pair)
    figure = Figure(layout)
    figure.draw(embedded=controls)
    _check_every_combination(figure, controls)


def test_a_selector_shared_by_two_plots_switches_both():
    selector = Selector([U, P])
    plots = []
    for seed in range(2):
        data, config, (batch,) = _grid_data(3, seed=seed)
        plots.append(
            HeatmapPlot(
                data, config, position=X, color=selector, controls=[SliderAxis(batch)]
            )
        )
    figure = Figure(Row(*plots))
    figure.draw(embedded=[selector])
    _check_every_combination(figure, [selector])


def test_embedding_writes_only_what_changes():
    (d1, c1, (b1,)), (d2, c2, (b2,)) = _grid_data(3), _grid_data(3, seed=1)
    slider = SliderAxis(b1)
    moving = HeatmapPlot(d1, c1, position=X, color=U, controls=[slider])
    still = HeatmapPlot(d2, c2, position=X, color=U, controls=[FacetAxis(b2)])
    figure = Figure(Row(moving, still))
    backend = figure.draw(embedded=[slider])
    restyle, relayout, indices = backend.layout.sliders[0].steps[0].args
    assert list(indices) == [0]
    assert set(restyle) == {"z"}
    assert relayout == {}


# --- selectors and color scales -------------------------------------------------


class TestSelectorModes:
    def _plot(self, color_scale=None):
        data, config, (batch,) = _grid_data(3)
        return (
            HeatmapPlot(
                data,
                config,
                position=X,
                color=Selector([U, P]),
                color_scale=color_scale,
                controls=[SliderAxis(batch)],
            ),
            data,
        )

    def test_a_selector_next_to_a_control_axis_draws_every_candidate(self):
        plot, _ = self._plot()
        figure = Figure(plot, renderer=RecordingRenderer())
        backend = figure.draw()
        assert len([a for a in backend["artists"] if a.visible]) == 1
        backend = figure.draw(embedded=plot.controls + plot.selectors)
        assert len([a for a in backend["artists"] if a.visible]) == 1
        assert len(backend["artists"]) == 3  # the connected one, hidden, and two

    def test_a_selector_alone_is_embedded_by_its_data(self):
        plot, _ = self._plot()
        backend = Figure(plot, renderer=RecordingRenderer()).draw(embedded=plot.selectors)
        assert len(backend["artists"]) == 1


# --- connected outputs ----------------------------------------------------------


class TestDash:
    def test_a_slider_is_connected_and_a_time_axis_embedded(self):
        data, config, (runs, steps) = _curve_data(2, 3)
        slider, time = SliderAxis(runs), TimeAxis(steps)
        figure = Figure(LinePlot(data, config, x=T, y=U, controls=[slider, time]))
        DashApplication.create(figure)
        assert DashApplication.interactive_controls(figure) == [slider]
        assert DashApplication.embedded_controls(figure) == [time]
        for run in slider.states:
            DashApplication.update(figure, [slider], [run])
            for step in time.states:
                _check_states(figure, [time], [(time, step)])

    def test_a_connected_application_requires_plotly(self):
        data, config, (batch,) = _grid_data(3)
        plot = HeatmapPlot(
            data, config, position=X, color=U, controls=[SliderAxis(batch)]
        )
        with pytest.raises(ValueError, match="PlotlyRenderer"):
            DashApplication.create(Figure(plot, renderer=RecordingRenderer()))


# --- renderer, theme, facets ----------------------------------------------------


class TestRenderer:
    def test_a_renderer_draws_one_figure(self):
        data, config, (batch,) = _grid_data(3)
        plot = HeatmapPlot(
            data, config, position=X, color=U, controls=[SliderAxis(batch)]
        )
        renderer = PlotlyRenderer()
        Figure(plot, renderer=renderer)
        with pytest.raises(ValueError, match="one figure"):
            Figure(plot, renderer=renderer)

    def test_categorical_colors_come_from_the_renderers_theme(self):
        data, config, (runs,) = _curve_data(2)
        theme = Theme(primary_color_palette=["#123456", "#abcdef"])
        plot = LinePlot(data, config, x=T, y=(U, P), controls=[SliderAxis(runs)])
        backend = Figure(plot, renderer=PlotlyRenderer(theme=theme)).draw()
        assert [t.line.color for t in backend.data] == ["#123456", "#abcdef"]


class TestFacets:
    def test_a_plot_without_facets_is_drawn_in_every_facet_cell(self):
        data, config, (batch,) = _grid_data(3)
        facetted = HeatmapPlot(
            data, config, position=X, color=U, controls=[FacetAxis(batch)]
        )
        points, points_config, (single,) = _grid_data(1)
        outline = ScatterPlot(
            points, points_config, position=X, controls=[FixedAxis(single)]
        )
        figure = Figure(Overlay(facetted, outline), renderer=RecordingRenderer())
        backend = figure.draw()
        assert len(figure.cells) == 3
        cells_of_outline = {
            (a.cell.row, a.cell.col)
            for a in backend["artists"]
            if type(a.result).__name__ == "MarkerResult"
        }
        assert cells_of_outline == {(c.row, c.col) for c in figure.cells}

    def test_plots_with_different_facets_cannot_share_an_overlay(self):
        (d1, c1, (b1,)), (d2, c2, (b2,)) = _grid_data(3), _grid_data(3)
        first = HeatmapPlot(d1, c1, position=X, color=U, controls=[FacetAxis(b1)])
        second = HeatmapPlot(d2, c2, position=X, color=U, controls=[FacetAxis(b2)])
        with pytest.raises(ValueError, match="same"):
            Overlay(first, second)


class TestGif:
    def _figure(self):
        data, config, (batch,) = _grid_data(3)
        time = TimeAxis(batch)
        plot = HeatmapPlot(data, config, position=X, color=U, controls=[time])
        return Figure(plot, renderer=PlotlyRenderer()), time

    def test_every_frame_is_the_whole_figure_in_its_state(self):
        figure, time = self._figure()
        figure.draw(embedded=[time])
        frames = figure.renderer.frame_figures()
        assert len(frames) == len(time.states)
        for state, frame in zip(time.states, frames):
            time.state = state
            drawn = figure.draw()
            assert len(frame.data) == len(drawn.data)
            np.testing.assert_allclose(
                np.asarray(frame.data[0].z, dtype=float),
                np.asarray(drawn.data[0].z, dtype=float),
            )
            assert not frame.layout.sliders and not frame.frames

    def test_save_gif_needs_a_time_axis(self, tmp_path):
        data, config, (batch,) = _grid_data(3)
        figure = Figure(
            HeatmapPlot(data, config, position=X, color=U, controls=[SliderAxis(batch)]),
            renderer=PlotlyRenderer(),
        )
        with pytest.raises(ValueError, match="TimeAxis"):
            figure.save_gif(tmp_path / "out.gif")

    def test_frames_are_shown_without_transitions(self):
        """A transition would interpolate trace coordinates between frames,
        e.g. move the triangles of a mesh between value bins."""
        figure, time = self._figure()
        backend = figure.draw(embedded=[time])
        (slider,) = backend.layout.sliders
        assert slider.transition.duration == 0
        calls = [step.args[1] for step in slider.steps]
        calls += [button.args[1] for button in backend.layout.updatemenus[0].buttons]
        assert all(call["transition"]["duration"] == 0 for call in calls)

    def test_an_animation_does_not_play_on_load(self):
        """Playing on load would use Plotly's default transitions."""
        figure, time = self._figure()
        figure.draw(embedded=[time])
        page = figure.renderer.html(full_html=False)
        assert "Plotly.animate" not in page


class TestPhysicalTime:
    """A time axis over a geometry steps in its coordinates."""

    def _figure(self, times):
        time_grid = GridGeometry(T, np.asarray(times, dtype=float)[:, None])
        config = DataConfiguration(
            GeometryAxes(time_grid), GeometryAxes(GridGeometry.build_unit_grid(X, 3)),
            FeatureAxes(U),
        )
        data = np.random.default_rng(0).random(config.shape)
        time = TimeAxis(T)
        plot = HeatmapPlot(data, config, position=X, color=U, controls=[time])
        return Figure(plot, renderer=PlotlyRenderer()), time

    def test_frames_last_in_proportion_to_their_time_step(self):
        figure, time = self._figure([0.0, 0.1, 0.5, 2.0])
        # Steps 0.1, 0.4, 1.5 (mean 2/3, also taken for the last frame).
        assert time.frame_weights() == pytest.approx([0.15, 0.6, 2.25, 1.0])
        backend = figure.draw(embedded=[time])
        play = backend.layout.updatemenus[0].buttons[0].args[1]
        durations = [frame["duration"] for frame in play["frame"]]
        assert durations == pytest.approx([time.duration * w for w in time.frame_weights()])

    def test_uniform_time_steps_keep_one_duration(self):
        figure, time = self._figure([0.0, 0.5, 1.0])
        backend = figure.draw(embedded=[time])
        play = backend.layout.updatemenus[0].buttons[0].args[1]
        assert play["frame"]["duration"] == time.duration

    def test_steps_are_labeled_with_their_time(self):
        figure, time = self._figure([0.0, 0.1, 0.5, 2.0])
        backend = figure.draw(embedded=[time])
        assert [s.label for s in backend.layout.sliders[0].steps] == ["0", "0.1", "0.5", "2"]


class TestDashWidgets:
    def test_the_figure_title_is_the_app_title(self):
        pytest.importorskip("dash")
        data, config, (batch,) = _grid_data(3)
        plot = HeatmapPlot(data, config, position=X, color=U, controls=[SliderAxis(batch)])
        app = DashApplication.create(Figure(plot, title="$u$ over x"))
        assert app.title == "u over x"

    def test_every_widget_gets_its_own_id(self):
        pytest.importorskip("dash")
        data, config, batches = _grid_data(3, 2)
        sliders = [SliderAxis(b) for b in batches]
        plot = HeatmapPlot(data, config, position=X, color=Selector([U, P]), controls=sliders)
        figure = Figure(plot)
        widgets = DashApplication.interactive_controls(figure)
        assert len(widgets) == 3
        app = DashApplication.create(figure)
        ids = [c.id for c in app.layout.children[1:] for c in c.children[1:]]
        assert len(set(ids)) == len(ids) == 3


def test_linked_3d_cells_move_their_cameras_together_in_dash():
    pytest.importorskip("dash")
    from qewton.geometries.continuous.domains_3d.box import Box
    from qewton.visualization import PositionAxis, SurfacePlot

    Y = Variable("y", 3)
    config = DataConfiguration(
        GeometryAxes(Box(Y, [0.0] * 3, 1, 1, 1).create_mesh(0.6)), FeatureAxes(U)
    )
    shared = dict(x=PositionAxis(), y=PositionAxis(), z=PositionAxis())
    plots = [SurfacePlot(np.zeros(config.shape), config, position=Y, color=U) for _ in range(2)]
    app = DashApplication.create(Figure(Row(*(Overlay(p, **shared) for p in plots))))
    stores = [c for c in app.layout.children if getattr(c, "id", None) == "camera-sync"]
    assert len(stores) == 1


def test_controls_keep_the_view_of_the_viewer():
    """No embedded state sets a camera or the zoom, and the figure keeps
    the viewer's view when redrawn (Dash), so the view survives a control."""
    from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
    from qewton.visualization import SurfacePlot

    time, selector = TimeAxis(BatchAxes(3)), Selector([U, P])
    config = DataConfiguration(
        time.variable_or_axes,
        GeometryAxes(Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)),
        FeatureAxes(U * P),
    )
    data = np.random.default_rng(0).random(config.shape)
    plot = SurfacePlot(data, config, position=X, z=selector, color=selector, controls=[time])
    backend = Figure(plot, renderer=PlotlyRenderer()).draw(embedded=[time, selector])
    assert backend.layout.uirevision
    written = [str(f.layout.to_plotly_json()) for f in backend.frames]
    written += [str(b.args) for m in backend.layout.updatemenus for b in m.buttons]
    written += [str(s.args) for sl in backend.layout.sliders for s in sl.steps]
    assert not any("camera" in w or "range" in w for w in written)
