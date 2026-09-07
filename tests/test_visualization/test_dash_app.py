import numpy as np
from dash import dcc

from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.applications.dash_app import DashApplication
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.spec import ColorSpec, VariableSpec


def _mesh_field_plot_with_selector(small_mesh_geometry):
    temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
    n = small_mesh_geometry.mesh.vertices.shape[0]
    data = np.stack([np.full(n, 1.0), np.full(n, 2.0)], axis=-1)
    config = DataConfiguration(
        GeometryAxes(small_mesh_geometry), FeatureAxes(temperature * pressure)
    )
    selector = VariableSpec([temperature, pressure])
    plot = MeshFieldPlot(data, config, color=ColorSpec(selector), show_edges=False)
    return plot, selector


class TestAppTitle:
    def test_a_figure_with_no_title_renders_the_index_page_without_error(self, small_mesh_geometry):
        """Regression: Dash interpolates app.title into its index HTML via
        str.replace(), which crashes on None - Figure's own default title
        is None, so this is the common case, not an edge case."""
        plot, _ = _mesh_field_plot_with_selector(small_mesh_geometry)
        app = DashApplication.create(Figure(plot))
        assert app.index()  # must not raise

    def test_an_explicit_figure_title_is_used_as_the_app_title(self, small_mesh_geometry):
        plot, _ = _mesh_field_plot_with_selector(small_mesh_geometry)
        app = DashApplication.create(Figure(plot, title="My Figure"))
        assert app.title == "My Figure"


class TestCreateDropdown:
    def test_options_are_candidate_indices_labeled_by_name(self):
        temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
        spec = VariableSpec([temperature, pressure])
        dropdown = DashApplication.create_dropdown(spec)
        assert dropdown.options == [
            {"label": "temperature", "value": 0}, {"label": "pressure", "value": 1}
        ]
        assert dropdown.value == 0
        assert dropdown.clearable is False

    def test_value_reflects_a_non_default_current_selection(self):
        temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
        spec = VariableSpec([temperature, pressure], init_index=1)
        dropdown = DashApplication.create_dropdown(spec)
        assert dropdown.value == 1


class TestVariableSpecNotInFigureControls:
    def test_variable_spec_is_collected_separately_from_controls(self, small_mesh_geometry):
        """The whole point of Plot.variable_specs: a VariableSpec must never
        end up in figure.controls, since apply_controls()/_resolve_controls()
        never learned to skip it - it isn't a whole-axis control."""
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        fig = Figure(plot)
        assert selector not in fig.controls
        assert fig.variable_specs == [selector]


class TestDashLayout:
    def test_layout_includes_a_dropdown_for_a_variable_spec(self, small_mesh_geometry):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        app = DashApplication.create(Figure(plot))
        widgets = [getattr(c, "children", None) for c in app.layout.children]
        dropdowns = [w[1] for w in widgets if isinstance(w, list) and isinstance(w[1], dcc.Dropdown)]
        assert len(dropdowns) == 1
        assert dropdowns[0].id == selector.name

    def test_callback_inputs_include_the_variable_spec(self, small_mesh_geometry):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        app = DashApplication.create(Figure(plot))
        inputs = app.callback_map["figure.figure"]["inputs"]
        assert {"id": selector.name, "property": "value"} in inputs
