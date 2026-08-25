import sys

import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.curve import LinePlot
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.plots.spec import FacetSpec, TimeSpec


def _scatter_with_facet(n_facets, n_samples):
    X, Y = Variable("x", 1), Variable("y", 1)
    facet_axis = BatchAxes(n_facets)
    sample_axis = BatchAxes(n_samples)
    data = np.random.randn(n_facets, n_samples, 2)
    config = DataConfiguration(facet_axis, sample_axis, FeatureAxes(X * Y))
    facet = FacetSpec(facet_axis, orientation="col")
    plot = ScatterPlot(data, config, x=X, y=Y, controls=[facet])
    return plot, facet


class TestFacets:
    def test_column_facet_produces_one_trace_per_value_and_the_right_grid_shape(self):
        plot, facet = _scatter_with_facet(3, 5)
        fig = Figure(plot)
        backend_figure = fig.draw()
        assert fig.grid_shape() == (1, 3)
        assert len(backend_figure.data) == 3

    def test_redraw_does_not_duplicate_traces(self):
        plot, facet = _scatter_with_facet(2, 5)
        fig = Figure(plot)
        backend_figure = fig.draw()
        fig.draw()
        assert len(backend_figure.data) == 2

    def test_facet_state_is_restored_after_drawing(self):
        plot, facet = _scatter_with_facet(3, 5)
        original = facet.state
        Figure(plot).draw()
        assert facet.state == original

    def test_unlabeled_facet_produces_no_subplot_titles(self):
        plot, facet = _scatter_with_facet(3, 5)
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.annotations == ()

    def test_labeled_facet_produces_matching_subplot_titles(self):
        plot, facet = _scatter_with_facet(3, 5)
        facet.labels = ["A", "B", "C"]
        backend_figure = Figure(plot).draw()
        assert [a.text for a in backend_figure.layout.annotations] == ["A", "B", "C"]

    def test_facet_labels_must_match_the_number_of_values(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        facet_axis = BatchAxes(3)
        sample_axis = BatchAxes(5)
        data = np.random.randn(3, 5, 2)
        config = DataConfiguration(facet_axis, sample_axis, FeatureAxes(X * Y))
        facet = FacetSpec(facet_axis, orientation="col", labels=["only one"])
        with pytest.raises(AssertionError, match="labels"):
            ScatterPlot(data, config, x=X, y=Y, controls=[facet])


class _TimeSpecFixture:
    @staticmethod
    def animated_line_plot(n_steps=4, n_samples=10, duration=300):
        Y = Variable("y", 1)
        step_axis = BatchAxes(n_steps)
        sample_axis = BatchAxes(n_samples)
        t = np.linspace(0, 6, n_samples)
        data = np.stack(
            [np.sin(t - step / n_steps) for step in range(n_steps)]
        )[..., None]
        config = DataConfiguration(step_axis, sample_axis, FeatureAxes(Y))
        spec = TimeSpec(step_axis, duration=duration)
        plot = LinePlot(data, config, x=sample_axis, y=Y, controls=[spec])
        return plot, spec, data


class TestTimeSpecAnimation:
    def test_produces_one_frame_per_state(self):
        plot, spec, data = _TimeSpecFixture.animated_line_plot(n_steps=5)
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.frames) == 5
        assert [f.name for f in backend_figure.frames] == ["0", "1", "2", "3", "4"]

    def test_live_trace_shows_the_initial_state_after_animating(self):
        plot, spec, data = _TimeSpecFixture.animated_line_plot(n_steps=4)
        backend_figure = Figure(plot).draw()
        assert np.allclose(backend_figure.data[0].y, data[0, :, 0])

    def test_a_later_frame_matches_that_states_data(self):
        plot, spec, data = _TimeSpecFixture.animated_line_plot(n_steps=4)
        backend_figure = Figure(plot).draw()
        frame_2 = backend_figure.frames[2]
        assert np.allclose(np.array(frame_2.data[0]["y"]), data[2, :, 0])

    def test_play_pause_and_slider_controls_are_present(self):
        plot, spec, data = _TimeSpecFixture.animated_line_plot()
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.layout.updatemenus) == 1
        assert len(backend_figure.layout.sliders) == 1

    def test_redraw_does_not_duplicate_traces_or_frames(self):
        plot, spec, data = _TimeSpecFixture.animated_line_plot()
        fig = Figure(plot)
        backend_figure = fig.draw()
        fig.draw()
        assert len(backend_figure.data) == 1
        assert len(backend_figure.frames) == 4

    def test_a_figure_without_timespec_is_unaffected(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y)
        backend_figure = Figure(plot).draw()
        assert backend_figure.frames == ()

    def test_timespec_combined_with_facetspec_on_the_same_plot_is_rejected(self):
        """update() has no notion of "which facet cell" outside
        Figure._draw_plot()'s own per-cell state loop - animating would
        silently reuse whatever facet state was left over after the normal
        draw pass for every cell, so this is an explicit, loud rejection
        instead."""
        X, Y = Variable("x", 1), Variable("y", 1)
        facet_axis = BatchAxes(2)
        step_axis = BatchAxes(3)
        sample_axis = BatchAxes(5)
        data = np.zeros((2, 3, 5, 1))
        config = DataConfiguration(facet_axis, step_axis, sample_axis, FeatureAxes(Y))
        plot = LinePlot(
            data, config, x=sample_axis, y=Y,
            controls=[TimeSpec(step_axis), FacetSpec(facet_axis)],
        )
        with pytest.raises(NotImplementedError):
            Figure(plot).draw()


class TestSaveGif:
    def test_raises_a_clear_error_with_no_frames(self, tmp_path):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(5)
        data = np.random.rand(5, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y)
        with pytest.raises(ValueError, match="no animation frames"):
            Figure(plot).save_gif(str(tmp_path / "out.gif"))

    def test_raises_a_clear_error_when_kaleido_is_missing(self, tmp_path, monkeypatch):
        plot, spec, data = _TimeSpecFixture.animated_line_plot()
        monkeypatch.setitem(sys.modules, "kaleido", None)
        with pytest.raises(ImportError, match="kaleido"):
            Figure(plot).save_gif(str(tmp_path / "out.gif"))


class TestSaveImage:
    def _scatter_plot(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        return ScatterPlot(data, config, x=X, y=Y)

    def test_save_png_writes_a_nonempty_file(self, tmp_path):
        path = tmp_path / "out.png"
        Figure(self._scatter_plot()).save_png(str(path))
        assert path.stat().st_size > 0

    def test_save_svg_writes_a_nonempty_file(self, tmp_path):
        path = tmp_path / "out.svg"
        Figure(self._scatter_plot()).save_svg(str(path))
        assert path.stat().st_size > 0

    def test_save_png_raises_a_clear_error_when_kaleido_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "kaleido", None)
        with pytest.raises(ImportError, match="kaleido"):
            Figure(self._scatter_plot()).save_png(str(tmp_path / "out.png"))
