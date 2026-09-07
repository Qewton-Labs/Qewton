import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.spec import ColorSpec, TimeSpec, VariableSpec


def _mesh_field_plot_with_selector(small_mesh_geometry, **kwargs):
    temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
    n = small_mesh_geometry.mesh.vertices.shape[0]
    data = np.stack([np.full(n, 1.0), np.full(n, 2.0)], axis=-1)
    config = DataConfiguration(
        GeometryAxes(small_mesh_geometry), FeatureAxes(temperature * pressure)
    )
    selector = VariableSpec([temperature, pressure])
    plot = MeshFieldPlot(data, config, color=ColorSpec(selector), show_edges=False, **kwargs)
    return plot, selector


class TestApplyVariableSelector:
    def test_adds_one_restyle_button_per_candidate(self, small_mesh_geometry):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        fig = Figure(plot)
        fig.draw()
        fig._apply_variable_selectors()
        menus = fig.backend_figure.layout.updatemenus
        assert len(menus) == 1
        assert [b.label for b in menus[0].buttons] == ["temperature", "pressure"]
        assert all(b.method == "restyle" for b in menus[0].buttons)

    def test_leaves_the_live_trace_at_its_original_selection(self, small_mesh_geometry):
        """_apply_variable_selectors() only adds the dropdown menu - it must
        not change selector.state or the data actually drawn. FilledMeshArtist
        has no single trace carrying a continuous per-vertex value to
        inspect directly (it's binned across several flat-fill traces), so
        this checks the plot's own evaluated data instead."""
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        fig = Figure(plot)
        fig.draw()
        fig._apply_variable_selectors()
        assert selector.state.name == "temperature"
        assert np.all(np.asarray(plot.evaluate().color) == 1.0)

    def test_repeated_calls_do_not_duplicate_the_menu(self, small_mesh_geometry):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        fig = Figure(plot)
        fig.draw()
        fig._apply_variable_selectors()
        fig._apply_variable_selectors()
        assert len(fig.backend_figure.layout.updatemenus) == 1

    def test_coexists_with_a_timespec_animation_menu(self, small_mesh_geometry):
        temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
        step_axis = BatchAxes(2)
        n = small_mesh_geometry.mesh.vertices.shape[0]
        data = np.stack([
            np.stack([np.full(n, float(s)), np.full(n, float(s) * 10)], axis=-1)
            for s in range(2)
        ])
        config = DataConfiguration(
            step_axis, GeometryAxes(small_mesh_geometry), FeatureAxes(temperature * pressure)
        )
        selector = VariableSpec([temperature, pressure])
        plot = MeshFieldPlot(
            data, config, color=ColorSpec(selector),
            controls=[TimeSpec(step_axis)], show_edges=False,
        )
        fig = Figure(plot)
        fig.draw()  # runs animate() internally, adding the Play/Pause menu
        fig._apply_variable_selectors()
        menus = fig.backend_figure.layout.updatemenus
        assert len(menus) == 2
        labels = {tuple(b.label for b in m.buttons) for m in menus}
        assert ("Play", "Pause") in labels
        assert ("temperature", "pressure") in labels

    def test_no_op_without_any_variable_spec(self):
        from qewton.visualization.plots.data.samples import ScatterPlot

        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(5)
        data = np.random.randn(5, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y)
        fig = Figure(plot)
        fig.draw()
        fig._apply_variable_selectors()  # must not raise
        assert fig.backend_figure.layout.updatemenus == ()

    def test_show_and_save_html_apply_the_selector(self, small_mesh_geometry, tmp_path, monkeypatch):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        fig = Figure(plot)
        monkeypatch.setattr(fig.renderer, "show", lambda backend_figure: None)
        fig.show()
        assert len(fig.backend_figure.layout.updatemenus) == 1

        path = tmp_path / "out.html"
        fig2 = Figure(_mesh_field_plot_with_selector(small_mesh_geometry)[0])
        fig2.save_html(str(path))
        assert len(fig2.backend_figure.layout.updatemenus) == 1
        assert path.exists()
