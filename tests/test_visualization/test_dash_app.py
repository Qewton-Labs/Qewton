import numpy as np
from dash import dcc

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.applications.dash_app import DashApplication
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.plots.spec import ColorSpec, SelectorSpec, SliderSpec


def _mesh_field_plot_with_selector(small_mesh_geometry):
    temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
    n = small_mesh_geometry.mesh.vertices.shape[0]
    data = np.stack([np.full(n, 1.0), np.full(n, 2.0)], axis=-1)
    config = DataConfiguration(
        GeometryAxes(small_mesh_geometry), FeatureAxes(temperature * pressure)
    )
    selector = SelectorSpec([temperature, pressure])
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
        spec = SelectorSpec([temperature, pressure])
        dropdown = DashApplication.create_dropdown(spec)
        assert dropdown.options == [
            {"label": "temperature", "value": 0}, {"label": "pressure", "value": 1}
        ]
        assert dropdown.value == 0
        assert dropdown.clearable is False

    def test_value_reflects_a_non_default_current_selection(self):
        temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
        spec = SelectorSpec([temperature, pressure], init_index=1)
        dropdown = DashApplication.create_dropdown(spec)
        assert dropdown.value == 1


def _scatter_plot_with_batch_slider(n=5):
    """A ScatterPlot with an explicit SliderSpec over its own DataConfig's
    batch axis - one independent SliderSpec instance per call, each wrapping
    a distinct BatchAxes object of the same shape (as graph.visualize()'s
    own auto-resolved per-port sliders do), so PlotSpec.name's str(axis)
    fallback gives every one of them the identical name/label."""
    x, y = Variable("x", 1), Variable("y", 1)
    config = DataConfiguration(BatchAxes(n), FeatureAxes(x * y))
    data = np.random.randn(n, 2)
    batch_axis = config.axes[0]
    slider = SliderSpec(batch_axis)
    return ScatterPlot(data, config, x=x, y=y, controls=[slider]), slider


class TestCreateSlider:
    def test_id_defaults_to_the_spec_name(self):
        _, slider = _scatter_plot_with_batch_slider()
        slider.resolve(range(5))
        assert DashApplication.create_slider(slider).id == slider.name

    def test_id_can_be_overridden(self):
        _, slider = _scatter_plot_with_batch_slider()
        slider.resolve(range(5))
        assert DashApplication.create_slider(slider, id="slider-spec-0").id == "slider-spec-0"


class TestSliderIdCollision:
    def test_sliders_over_same_shaped_but_distinct_axes_share_a_name(self):
        """Confirms the premise of the regression below: two independently-
        resolved SliderSpecs over same-shaped BatchAxes objects really do
        collide by name - this is what made keying Dash component ids off
        SliderSpec.name a bug, not a hypothetical."""
        _, slider1 = _scatter_plot_with_batch_slider()
        _, slider2 = _scatter_plot_with_batch_slider()
        assert slider1 is not slider2
        assert slider1.name == slider2.name

    def test_three_panels_get_distinct_component_ids_and_independent_state(self):
        """Regression: graph.visualize([port1, port2, port3]) (no
        reference=) resolves one independent SliderSpec per port, over each
        port's own same-shaped batch axis - keying Dash's component id off
        SliderSpec.name collided all three into one shared id, so only the
        last slider was interactive and it drove every panel at once."""
        plots, sliders = zip(*(_scatter_plot_with_batch_slider() for _ in range(3)))
        fig = Figure(list(plots))
        app = DashApplication.create(fig)

        inputs = app.callback_map["figure.figure"]["inputs"]
        ids = [i["id"] for i in inputs]
        assert len(ids) == len(set(ids)) == 3

        for slider in sliders:
            slider.resolve(range(5))
        sliders[0].state, sliders[1].state, sliders[2].state = 0, 4, 2
        assert (sliders[0].state, sliders[1].state, sliders[2].state) == (0, 4, 2)

    def test_slider_labels_disambiguate_by_owning_plot(self):
        plots, sliders = zip(*(_scatter_plot_with_batch_slider() for _ in range(3)))
        fig = Figure(list(plots))
        labels = [DashApplication._slider_label(fig, s) for s in sliders]
        assert labels == [f"plot {i}: {sliders[i].name}" for i in range(3)]


class TestSelectorSpecNotInFigureControls:
    def test_selector_spec_is_collected_separately_from_controls(self, small_mesh_geometry):
        """The whole point of Plot.selector_specs: a SelectorSpec must never
        end up in figure.controls, since apply_controls()/_resolve_controls()
        never learned to skip it - it isn't a whole-axis control."""
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        fig = Figure(plot)
        assert selector not in fig.controls
        assert fig.selector_specs == [selector]


class TestDashLayout:
    def test_layout_includes_a_dropdown_for_a_selector_spec(self, small_mesh_geometry):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        app = DashApplication.create(Figure(plot))
        widgets = [getattr(c, "children", None) for c in app.layout.children]
        dropdowns = [w[1] for w in widgets if isinstance(w, list) and isinstance(w[1], dcc.Dropdown)]
        assert len(dropdowns) == 1
        assert dropdowns[0].id == "selector-spec-0"

    def test_callback_inputs_include_the_selector_spec(self, small_mesh_geometry):
        plot, selector = _mesh_field_plot_with_selector(small_mesh_geometry)
        app = DashApplication.create(Figure(plot))
        inputs = app.callback_map["figure.figure"]["inputs"]
        assert {"id": "selector-spec-0", "property": "value"} in inputs

    def test_specs_sharing_candidates_get_distinct_component_ids(self):
        """A scatter's x and y SelectorSpecs share their candidates, hence
        their name - keying components by name raised DuplicateIdError."""
        import numpy as np

        from qewton.visualization.plots.table.scatter_table import TableScatter

        rng = np.random.default_rng(0)
        data = {k: rng.random(8) for k in ["lr", "layers", "width", "loss", "acc"]}
        fig = Figure(TableScatter(data, ["lr", "layers", "width"], ["loss", "acc"]))
        app = DashApplication.create(fig)
        inputs = app.callback_map["figure.figure"]["inputs"]
        ids = [i["id"] for i in inputs]
        assert len(ids) == len(set(ids)) == 3
        assert fig.selector_specs[0].name == fig.selector_specs[1].name
