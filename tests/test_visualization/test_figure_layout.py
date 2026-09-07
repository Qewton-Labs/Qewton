import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.layout import Column, Overlay, Row
from qewton.visualization.plots.data.curve import LinePlot
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.plots.spec import FacetSpec


def _line_plot(n_samples=10, variable_name="y", **kwargs):
    Y = Variable(variable_name, 1)
    sample_axis = BatchAxes(n_samples)
    data = np.random.rand(n_samples, 1)
    config = DataConfiguration(sample_axis, FeatureAxes(Y))
    return LinePlot(data, config, x=sample_axis, y=Y, **kwargs)


def _scatter_with_facet(n_facets, n_samples, orientation="col", **kwargs):
    X, Y = Variable("x", 1), Variable("y", 1)
    facet_axis = BatchAxes(n_facets)
    sample_axis = BatchAxes(n_samples)
    data = np.random.randn(n_facets, n_samples, 2)
    config = DataConfiguration(facet_axis, sample_axis, FeatureAxes(X * Y))
    facet = FacetSpec(facet_axis, orientation=orientation)
    plot = ScatterPlot(data, config, x=X, y=Y, controls=[facet], **kwargs)
    return plot, facet


def _scatter(n_samples=10, **kwargs):
    X, Y = Variable("x", 1), Variable("y", 1)
    sample_axis = BatchAxes(n_samples)
    data = np.random.randn(n_samples, 2)
    config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
    return ScatterPlot(data, config, x=X, y=Y, **kwargs)


class TestNonFacetedRow:
    def test_row_of_three_produces_a_clean_1x3_grid(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure(Row(a, b, c))
        assert fig.grid_shape() == (1, 3)

    def test_each_plot_draws_into_its_own_cell_not_overlaid(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure(Row(a, b, c))
        fig.draw()
        assert len(fig.artists[a]) == 1
        assert len(fig.artists[b]) == 1
        assert len(fig.artists[c]) == 1
        # distinct cells: (None, None) keys don't disambiguate position,
        # but distinct plots must end up in distinct backend_figure traces
        assert len({fig.artists[p][(None, None)].figure_idx for p in (a, b, c)}) == 3


class TestFacetedSinglePlotUnchanged:
    """a single faceted plot's grid must be byte-identical before and after the panel
    system."""

    def test_grid_shape_matches_the_facet_length_alone(self):
        plot, facet = _scatter_with_facet(3, 5)
        fig = Figure(plot)
        assert fig.grid_shape() == (1, 3)

    def test_produces_one_trace_per_facet_value(self):
        plot, facet = _scatter_with_facet(3, 5)
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.data) == 3


class TestColumn:
    def test_column_of_two_produces_a_2x1_grid(self):
        a, b = _line_plot(), _line_plot()
        fig = Figure(Column(a, b))
        assert fig.grid_shape() == (2, 1)


class TestOverlayInFigure:
    def test_overlaid_plots_share_one_cell(self):
        a, b = _line_plot(), _line_plot()
        fig = Figure(Overlay(a, b))
        assert fig.grid_shape() == (1, 1)
        backend_figure = fig.draw()
        assert len(backend_figure.data) == 2


class TestNestedLayoutBlocks:
    def test_2x2_from_column_of_rows(self):
        a, b, c, d = (_line_plot() for _ in range(4))
        fig = Figure(Column(Row(a, b), Row(c, d)))
        assert fig.grid_shape() == (2, 2)

    def test_padded_cell_draws_without_error_and_adds_no_trace(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure(Column(Row(a, b), c))
        backend_figure = fig.draw()
        # 3 real plots -> 3 traces, the padded cell contributes none
        assert len(backend_figure.data) == 3


class TestPanelsAndFacetsMultiply:
    def test_facet_extent_is_shared_across_every_panel(self):
        """facet extent is the max across all panels -
        an unfaceted panel still gets a full-size block, using its first
        cell."""
        faceted, facet = _scatter_with_facet(3, 5)
        plain = _line_plot()
        fig = Figure(Row(faceted, plain))
        assert fig.grid_shape() == (1, 6)  # 2 panels * 3 facet cols each

    def test_unfaceted_panel_only_occupies_its_blocks_first_cell(self):
        faceted, facet = _scatter_with_facet(3, 5)
        plain = _line_plot()
        fig = Figure(Row(faceted, plain))
        fig.draw()
        assert len(fig.artists[plain]) == 1

    def test_unfaceted_panel_spans_its_whole_block(self):
        """The plain panel's block is 3 facet columns wide (matching
        `faceted`'s FacetSpec) - its single cell must span all 3, not sit
        narrow in just the first."""
        faceted, facet = _scatter_with_facet(3, 5)
        plain = _line_plot()
        backend_figure = Figure(Row(faceted, plain)).draw()
        facet_col_width = backend_figure.layout.xaxis.domain[1]
        plain_domain = backend_figure.layout.xaxis4.domain
        plain_width = plain_domain[1] - plain_domain[0]
        # Spans all 3 facet columns plus their inner gaps - noticeably more
        # than 3x a single column's width, not squeezed into just one.
        assert plain_width > 3 * facet_col_width


class TestAddPlotAppendsANewPanel:
    def test_add_plot_creates_a_separate_row_each_time(self):
        fig = Figure()
        a, b = _line_plot(), _line_plot()
        fig.add_plot(a)
        fig.add_plot(b)
        assert fig.grid_shape() == (2, 1)
        assert fig.panels == [[fig.panels[0][0]], [fig.panels[1][0]]]

    def test_color_index_follows_add_order(self):
        fig = Figure()
        a, b = _line_plot(), _line_plot()
        fig.add_plot(a)
        fig.add_plot(b)
        assert a.color_index == 0
        assert b.color_index == 1


class TestListInputMeansRow:
    def test_list_of_plots_behaves_like_row(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure([a, b, c])
        assert fig.grid_shape() == (1, 3)

    def test_color_index_follows_reading_order(self):
        a, b = _line_plot(), _line_plot()
        Figure(Row(Overlay(a, b)))
        assert a.color_index == 0
        assert b.color_index == 1


class TestEmptyFigure:
    def test_no_plots_defaults_to_a_1x1_grid(self):
        fig = Figure()
        assert fig.grid_shape() == (1, 1)


class TestPanelSubplotTitles:
    """panel title from plot.title, facet title
    appended when both are present."""

    def test_each_panel_gets_its_own_plot_title(self):
        a = _scatter(title="Predicted")
        b = _scatter(title="Exact")
        backend_figure = Figure(Row(a, b)).draw()
        assert [ann.text for ann in backend_figure.layout.annotations] == [
            "Predicted",
            "Exact",
        ]

    def test_facet_label_is_appended_to_the_panel_title(self):
        plot, facet = _scatter_with_facet(2, 5, title="Samples")
        facet.labels = ["low", "high"]
        backend_figure = Figure(plot).draw()
        assert [ann.text for ann in backend_figure.layout.annotations] == [
            "Samples, low",
            "Samples, high",
        ]

    def test_a_plot_with_no_title_and_no_facet_label_gets_no_annotation(self):
        a, b = _scatter(), _scatter()
        backend_figure = Figure(Row(a, b)).draw()
        assert backend_figure.layout.annotations == ()

    def test_unfaceted_panel_title_appears_once_not_once_per_shared_block_cell(self):
        """A panel with no FacetSpec sharing a Row with a faceted one still
        gets a full-size block, but only draws into
        its first cell - its title must not be broadcast across the rest
        of that block."""
        faceted, facet = _scatter_with_facet(3, 5)
        plain = _scatter(title="Reference")
        backend_figure = Figure(Row(faceted, plain)).draw()
        titles = [ann.text for ann in backend_figure.layout.annotations]
        assert titles.count("Reference") == 1


class TestPanelYAxisTitles:
    """Several plots sharing one Overlay via a `.y: AxisSpec` (LinePlot,
    ScatterPlot, StructuredGridPlot/HeatmapPlot/...) only get a y-axis
    title when they all name the same quantity - a mix must not
    misleadingly show just one of them."""

    def test_single_curve_keeps_its_own_y_title(self):
        a = _line_plot(variable_name="u")
        backend_figure = Figure(a).draw()
        assert backend_figure.layout.yaxis.title.text == "$u$"

    def test_overlay_of_the_same_quantity_keeps_the_shared_title(self):
        a = _line_plot(variable_name="u")
        b = _line_plot(variable_name="u", label="Run 2")
        backend_figure = Figure(Overlay(a, b)).draw()
        assert backend_figure.layout.yaxis.title.text == "$u$"

    def test_overlay_of_different_quantities_clears_the_title(self):
        a = _line_plot(variable_name="f")
        b = _line_plot(variable_name="u")
        backend_figure = Figure(Overlay(a, b)).draw()
        assert backend_figure.layout.yaxis.title.text == ""

    def test_a_non_line_plot_in_the_overlay_is_ignored(self):
        """Only LinePlots' own y quantity participates - a PointCloudPlot
        sharing the Overlay (no quantity of its own, baselined at y=0)
        must not force the title blank on its own."""
        from qewton.geometries.discrete.point_cloud import PointCloud
        from qewton.visualization.plots.data.points import PointCloudPlot

        a = _line_plot(variable_name="u")
        T = Variable("t", 1)
        points = np.linspace(0.0, 1.0, 10).reshape(-1, 1).astype(np.float32)
        geometry = PointCloud(T, points)
        point_cloud_config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(T))
        b = PointCloudPlot(points, point_cloud_config, color=None)
        backend_figure = Figure(Overlay(a, b)).draw()
        assert backend_figure.layout.yaxis.title.text == "$u$"

    def test_each_panel_in_a_row_gets_its_own_reconciled_title(self):
        panel_a = Overlay(_line_plot(variable_name="f"), _line_plot(variable_name="u"))
        panel_b = Overlay(_line_plot(variable_name="u"), _line_plot(variable_name="u"))
        backend_figure = Figure(Row(panel_a, panel_b)).draw()
        assert backend_figure.layout.yaxis.title.text == ""
        assert backend_figure.layout.yaxis2.title.text == "$u$"

    def test_applies_to_any_plot_family_with_a_y_axisspec_not_just_lineplot(self):
        """ScatterPlot has the same `.y: AxisSpec`/math_name-titled y-axis
        as LinePlot, via a completely separate renderer - the same
        conflict must be caught there too, not just for LinePlot."""
        X, F, U = Variable("x", 1), Variable("f", 1), Variable("u", 1)
        sample_axis = BatchAxes(10)
        config_f = DataConfiguration(sample_axis, FeatureAxes(X * F))
        config_u = DataConfiguration(sample_axis, FeatureAxes(X * U))
        a = ScatterPlot(np.random.rand(10, 2), config_f, x=X, y=F)
        b = ScatterPlot(np.random.rand(10, 2), config_u, x=X, y=U)
        backend_figure = Figure(Overlay(a, b)).draw()
        assert backend_figure.layout.yaxis.title.text == ""


class TestFigureTopTitle:
    """The whole-figure title (layout.title, distinct from a panel's own
    subplot annotation - see TestPanelSubplotTitles) comes only from an
    explicit Figure(title=...) - never from any individual Plot.title,
    which used to leak through from whichever plot's artist drew last."""

    def test_no_title_set_shows_no_title(self):
        backend_figure = Figure(_line_plot()).draw()
        assert backend_figure.layout.title.text is None

    def test_explicit_figure_title_is_shown(self):
        backend_figure = Figure(_line_plot(), title="My Figure").draw()
        assert backend_figure.layout.title.text == "My Figure"

    def test_a_plots_own_title_does_not_become_the_figure_title(self):
        backend_figure = Figure(_scatter(title="Predicted")).draw()
        assert backend_figure.layout.title.text is None

    def test_last_drawn_plots_title_does_not_win_without_a_figure_title(self):
        a, b = _scatter(title="First"), _scatter(title="Second")
        backend_figure = Figure(Row(a, b)).draw()
        assert backend_figure.layout.title.text is None


class TestRemovePlot:
    def test_removes_the_plot_from_its_panel(self):
        a, b = _line_plot(), _line_plot()
        fig = Figure(Row(a, b))
        fig.remove_plot(a)
        assert fig.panels[0][0].plots == []
        assert fig.panels[0][1].plots == [b]
        assert fig.plots == [b]

    def test_removes_a_plot_from_a_shared_overlay(self):
        a, b = _line_plot(), _line_plot()
        fig = Figure(Overlay(a, b))
        fig.remove_plot(a)
        assert fig.panels[0][0].plots == [b]

    def test_drawing_after_removal_does_not_include_the_removed_plot(self):
        a, b = _line_plot(), _line_plot()
        fig = Figure(Row(a, b))
        fig.remove_plot(a)
        backend_figure = fig.draw()
        assert len(backend_figure.data) == 1
        assert a not in fig.artists

    def test_removing_a_plot_not_in_the_figure_raises(self):
        fig = Figure(_line_plot())
        with pytest.raises(ValueError, match="not part of this Figure"):
            fig.remove_plot(_line_plot())

    def test_orphaned_control_is_dropped(self):
        from qewton.visualization.plots.spec import SliderSpec

        X, Y = Variable("x", 1), Variable("y", 1)
        step_axis = BatchAxes(3)
        sample_axis = BatchAxes(5)
        data = np.random.randn(3, 5, 2)
        config = DataConfiguration(step_axis, sample_axis, FeatureAxes(X * Y))
        slider = SliderSpec(step_axis, init_state=0, minimum=0, maximum=2)
        with_slider = ScatterPlot(data, config, x=X, y=Y, controls=[slider])
        other = _line_plot()

        fig = Figure(Row(with_slider, other))
        assert slider in fig.controls
        fig.remove_plot(with_slider)
        assert slider not in fig.controls


class TestReplacePlot:
    def test_replaces_in_the_same_panel_position(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure(Row(a, b))
        fig.replace_plot(a, c)
        assert fig.panels[0][0].plots == [c]
        assert fig.panels[0][1].plots == [b]
        assert fig.plots == [c, b]

    def test_new_plot_inherits_the_old_ones_color_index(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure(Row(a, b))
        old_index = a.color_index
        fig.replace_plot(a, c)
        assert c.color_index == old_index

    def test_drawing_after_replacement_uses_the_new_plot(self):
        a, b, c = _line_plot(), _line_plot(), _line_plot()
        fig = Figure(Row(a, b))
        fig.replace_plot(a, c)
        backend_figure = fig.draw()
        assert len(backend_figure.data) == 2
        assert a not in fig.artists
        assert c in fig.artists

    def test_replacing_a_plot_not_in_the_figure_raises(self):
        fig = Figure(_line_plot())
        with pytest.raises(ValueError, match="not part of this Figure"):
            fig.replace_plot(_line_plot(), _line_plot())

    def test_replacement_rejects_mismatched_embedding_dim(self):
        """Swapping in a plot from a different embedding_dim family must
        raise the same Overlay validation as constructing one directly -
        needs a sibling in the same Overlay to actually conflict with."""
        from qewton.geometries.discrete.mesh import Mesh
        from qewton.geometries.discrete.mesh_geometry import MeshGeometry
        from qewton.config.axes import GeometryAxes
        from qewton.visualization.plots.data.mesh import MeshFieldPlot
        from qewton.visualization.plots.spec import ColorSpec

        a, sibling = _line_plot(), _line_plot()  # both embedding_dim 2
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        cells = np.array([[0, 1, 2], [1, 3, 2]])
        geometry = MeshGeometry(Variable("p", 3), Mesh(vertices=vertices, cells=cells))
        U = Variable("u", 1)
        data = np.random.rand(4, 1)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        mesh_3d = MeshFieldPlot(data, config, color=ColorSpec(U))  # embedding_dim 3

        fig = Figure(Overlay(a, sibling))
        with pytest.raises(ValueError, match="embedding_dim"):
            fig.replace_plot(a, mesh_3d)
        assert fig.panels[0][0].plots == [a, sibling]  # unchanged on rejection
