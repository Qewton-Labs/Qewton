"""Controls over a dimension with physical coordinates (e.g. the time
factor of space-time data): labels, playback timing and slider placement."""

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.mesh import Mesh
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.visualization.applications.dash_app import DashApplication
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.spec import SliderSpec, TimeSpec
from qewton.visualization.renderers.plotly import PlotlyRenderer

T, X, U = Variable("t", 1), Variable("x", 2), Variable("u", 1)
TIMES = [0.0, 0.1, 0.5, 2.0]


def _space_time_plot(control, times=TIMES):
    """A MeshFieldPlot over a square with one GeometryAxes per time step."""
    time = MeshGeometry(
        T,
        Mesh(
            vertices=np.array(times)[:, None],
            cells=[[i, i + 1] for i in range(len(times) - 1)],
        ),
    )
    square = Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)
    n_space = len(square.mesh.vertices)
    data = np.array(times)[:, None, None] * np.ones((1, n_space, 1))
    config = DataConfiguration(GeometryAxes(time), GeometryAxes(square), FeatureAxes(U))
    return MeshFieldPlot(data, config, color=U, controls=[control])


class TestControlCoordinates:
    def test_a_control_over_a_1d_geometry_gets_its_coordinates(self):
        spec = TimeSpec(T)
        _space_time_plot(spec)
        assert spec.coordinates == pytest.approx(TIMES)
        assert spec.label(2) == "0.5"

    def test_a_control_over_a_batch_axis_has_none(self):
        steps = BatchAxes(3)
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0).create_mesh(0.5)
        data = np.zeros((3, len(square.mesh.vertices), 1))
        config = DataConfiguration(steps, GeometryAxes(square), FeatureAxes(U))
        spec = TimeSpec(steps)
        MeshFieldPlot(data, config, color=U, controls=[spec])
        assert spec.coordinates is None
        assert spec.label(2) == "2"

    def test_state_at_finds_the_nearest_coordinate(self):
        spec = SliderSpec(T)
        spec.set_coordinates(TIMES)
        assert spec.state_at(0.4) == 2
        assert spec.state_at(1.9) == 3

    def test_explicit_coordinates_are_kept(self):
        spec = TimeSpec(T)
        spec.set_coordinates([10.0, 11.0, 12.0, 13.0])
        _space_time_plot(spec)
        assert spec.coordinates == [10.0, 11.0, 12.0, 13.0]


class TestFrameWeights:
    def test_proportional_to_the_time_steps_with_mean_one(self):
        spec = TimeSpec(T)
        _space_time_plot(spec)
        weights = spec.frame_weights()
        steps = np.array([0.1, 0.4, 1.5, np.mean([0.1, 0.4, 1.5])])
        assert weights == pytest.approx(steps / steps.mean())
        assert np.mean(weights) == pytest.approx(1.0)

    def test_uniform_without_coordinates(self):
        spec = TimeSpec(T, values=[0, 1, 2])
        assert spec.frame_weights() == [1.0, 1.0, 1.0]


class TestPlotlyAnimation:
    def test_slider_steps_are_labeled_with_time(self):
        fig = Figure(_space_time_plot(TimeSpec(T))).draw()
        slider = fig.layout.sliders[0]
        assert [step.label for step in slider.steps] == ["0", "0.1", "0.5", "2"]
        assert slider.currentvalue.prefix == "t = "

    def test_non_uniform_steps_play_in_physical_time(self):
        spec = TimeSpec(T, duration=100)
        fig = Figure(_space_time_plot(spec)).draw()
        play_args = fig.layout.updatemenus[0].buttons[0].args[1]
        durations = [frame["duration"] for frame in play_args["frame"]]
        assert durations == pytest.approx([100 * w for w in spec.frame_weights()])
        assert play_args["fromcurrent"] is False

    def test_uniform_steps_keep_one_duration(self):
        spec = TimeSpec(T, duration=100)
        fig = Figure(_space_time_plot(spec, times=[0.0, 1.0, 2.0])).draw()
        play_args = fig.layout.updatemenus[0].buttons[0].args[1]
        assert play_args["frame"]["duration"] == 100
        assert play_args["fromcurrent"] is True

    def test_frames_update_every_fill_trace_of_a_2d_mesh_field(self):
        plot = _space_time_plot(TimeSpec(T))
        fig = Figure(plot).draw()
        # every value bin and the colorbar - the static wireframe is left out
        assert all(len(frame.traces) == plot.n_bins + 1 for frame in fig.frames)


class TestSaveGif:
    def test_frame_weights_are_passed_to_the_renderer(self, tmp_path, monkeypatch):
        captured = {}

        def fake_save_gif(backend_figure, path, fps=10, frame_weights=None):
            captured["frame_weights"] = frame_weights

        monkeypatch.setattr(PlotlyRenderer, "save_gif", staticmethod(fake_save_gif))
        spec = TimeSpec(T)
        Figure(_space_time_plot(spec)).save_gif(str(tmp_path / "out.gif"))
        assert captured["frame_weights"] == pytest.approx(spec.frame_weights())


class TestDashSlider:
    def test_slider_runs_over_the_physical_coordinates(self):
        spec = SliderSpec(T)
        _space_time_plot(spec)
        slider = DashApplication.create_slider(spec, id="slider-spec-0")
        assert slider.min == 0.0 and slider.max == 2.0
        assert slider.step is None
        assert sorted(slider.marks) == pytest.approx(TIMES)
        assert slider.marks[0.5] == "0.5"

    def test_slider_without_coordinates_is_unchanged(self):
        steps = BatchAxes(3)
        spec = SliderSpec(steps)
        spec.resolve(range(3))
        slider = DashApplication.create_slider(spec, id="slider-spec-0")
        assert (slider.min, slider.max, slider.step) == (0, 2, 1)


class TestColorRangeOverFrames:
    @staticmethod
    def _colorbar_ranges(fig):
        return {
            (frame.data[-1]["marker"]["cmin"], frame.data[-1]["marker"]["cmax"])
            for frame in fig.frames
        }

    def test_without_a_scale_each_frame_has_its_own_range(self):
        fig = Figure(_space_time_plot(TimeSpec(T))).draw()
        assert len(self._colorbar_ranges(fig)) > 1

    def test_a_scale_is_trained_over_every_frame(self):
        from qewton.visualization.plots.spec import Scale

        plot = _space_time_plot(TimeSpec(T))
        plot.color.scale = Scale()
        fig = Figure(plot).draw()
        assert self._colorbar_ranges(fig) == {(min(TIMES), max(TIMES))}

    def test_explicit_bounds_fix_the_range(self):
        from qewton.visualization.plots.spec import Scale

        plot = _space_time_plot(TimeSpec(T))
        plot.color.scale = Scale(vmin=-1.0, vmax=5.0)
        fig = Figure(plot).draw()
        assert self._colorbar_ranges(fig) == {(-1.0, 5.0)}
