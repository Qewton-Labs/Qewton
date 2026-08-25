import numpy as np
import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.data.samples import ScatterPlot
from qewton.visualization.plots.spec import ColorSpec, FacetSpec
from qewton.visualization.themes import DARK_THEME, DEFAULT_THEME, LIGHT_THEME, Theme
from qewton.visualization.themes.base import Theme as ThemeClass


class TestThemeDataclass:
    def test_default_cmap_is_a_real_overridable_field(self):
        """Regression: default_cmap used to have no type annotation, so it
        wasn't part of the dataclass at all - Theme(default_cmap=...) would
        raise TypeError."""
        custom = Theme(default_cmap="cividis")
        assert custom.default_cmap == "cividis"

    def test_light_and_dark_are_distinct_presets(self):
        assert LIGHT_THEME.background_color != DARK_THEME.background_color
        assert LIGHT_THEME.text_color != DARK_THEME.text_color
        assert LIGHT_THEME.geometry_color != DARK_THEME.geometry_color

    def test_default_theme_is_the_light_preset(self):
        assert DEFAULT_THEME is LIGHT_THEME

    def test_theme_default_and_dark_classmethods_still_work(self):
        assert ThemeClass.default().background_color == LIGHT_THEME.background_color
        assert ThemeClass.dark().background_color == DARK_THEME.background_color

    def test_get_color_cycles_through_the_palette(self):
        theme = Theme()
        n = len(theme.primary_color_palette)
        assert theme.get_color(0) == theme.get_color(n)  # wraps around

    def test_every_family_specific_color_field_exists(self):
        """These used to be informally getattr(theme, "x", fallback)'d from
        renderer code, not real fields - a real Theme must declare all of
        them so a corporate theme can override any of them."""
        theme = Theme()
        for field in (
            "geometry_color", "vector_color", "line_color", "grid_color",
            "node_color_by_type", "node_color_default", "cluster_outline_color",
        ):
            assert hasattr(theme, field)


class TestThemeAppliedToFigure:
    def _scatter_plot(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        return ScatterPlot(data, config, x=X, y=Y)

    def test_paper_and_plot_background_match_the_theme(self):
        backend_figure = Figure(self._scatter_plot(), theme=DARK_THEME).draw()
        assert backend_figure.layout.paper_bgcolor == DARK_THEME.background_color
        assert backend_figure.layout.plot_bgcolor == DARK_THEME.background_color

    def test_font_color_matches_the_theme(self):
        backend_figure = Figure(self._scatter_plot(), theme=DARK_THEME).draw()
        assert backend_figure.layout.font.color == DARK_THEME.text_color

    def test_light_and_dark_figures_differ(self):
        light = Figure(self._scatter_plot(), theme=LIGHT_THEME).draw()
        dark = Figure(self._scatter_plot(), theme=DARK_THEME).draw()
        assert light.layout.paper_bgcolor != dark.layout.paper_bgcolor

    def test_axis_gridline_color_matches_the_theme(self):
        backend_figure = Figure(self._scatter_plot(), theme=DARK_THEME).draw()
        assert backend_figure.layout.xaxis.gridcolor == DARK_THEME.grid_color

    def test_scene_background_matches_the_theme_for_3d_plots(self, circle_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        backend_figure = Figure(plot, theme=DARK_THEME).draw()
        assert backend_figure.layout.scene.bgcolor == DARK_THEME.background_color

    def test_theme_chrome_survives_a_faceted_grid(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        facet_axis = BatchAxes(2)
        sample_axis = BatchAxes(5)
        data = np.random.randn(2, 5, 2)
        config = DataConfiguration(facet_axis, sample_axis, FeatureAxes(X * Y))
        plot = ScatterPlot(
            data, config, x=X, y=Y, controls=[FacetSpec(facet_axis, orientation="col")]
        )
        backend_figure = Figure(plot, theme=DARK_THEME).draw()
        assert backend_figure.layout.paper_bgcolor == DARK_THEME.background_color
        assert backend_figure.layout.xaxis.gridcolor == DARK_THEME.grid_color
        assert backend_figure.layout.xaxis2.gridcolor == DARK_THEME.grid_color

    def test_per_artist_colors_come_from_the_theme_not_a_hardcoded_default(self, cylinder_mesh_geometry):
        """geometry_color used to be accessed via getattr(theme, "x",
        "lightgray") with the fallback doing the real work - now it's the
        theme's own value that must show up on the drawn trace."""
        from qewton.visualization.plots.geometry import GeometryPlot

        plot = GeometryPlot(cylinder_mesh_geometry)
        backend_figure = Figure(plot, theme=DARK_THEME).draw()
        assert backend_figure.data[0].color == DARK_THEME.geometry_color


class TestOpacityWiring:
    """opacity_default/surface_opacity/wireframe_opacity used to be declared
    on Theme but never read anywhere - setting them was a silent no-op.
    A custom theme with distinctive values makes the wiring unambiguous:
    default Theme()'s own defaults could coincidentally match Plotly's
    built-in default (1.0) and pass without actually being applied."""

    @staticmethod
    def _theme():
        return Theme(opacity_default=0.42, surface_opacity=0.55, wireframe_opacity=0.13)

    def test_scatter_marker_opacity(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        data = np.random.randn(10, 2)
        config = DataConfiguration(BatchAxes(10), FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y)
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.42

    def test_bar_opacity(self):
        from qewton.visualization.plots.data.samples import BarPlot

        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = BarPlot(data, config, x=sample_axis, y=Y)
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.42

    def test_line_opacity(self):
        from qewton.visualization.plots.data.curve import LinePlot

        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y)
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.42

    def test_arrow_field_opacity(self, cylinder_mesh_geometry):
        from qewton.visualization.plots.data.mesh import MeshVectorPlot
        from qewton.visualization.plots.spec import VectorSpec

        V = Variable("v", 3)
        n = len(np.asarray(cylinder_mesh_geometry.mesh.vertices))
        data = np.random.randn(n, 3) * 0.1
        config = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(V))
        plot = MeshVectorPlot(data, config, vector=VectorSpec(V))
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.42

    def test_mesh_field_surface_and_wireframe_opacity(self, circle_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U), show_edges=True)
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.55  # the surface (Mesh3d)
        assert backend_figure.data[1].opacity == 0.13  # its wireframe overlay

    def test_geometry_plot_3d_surface_and_wireframe_opacity(self, cylinder_mesh_geometry):
        from qewton.visualization.plots.geometry import GeometryPlot

        plot = GeometryPlot(cylinder_mesh_geometry, show_edges=True)
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.55
        assert backend_figure.data[1].opacity == 0.13

    def test_geometry_plot_2d_fill_and_wireframe_opacity_but_not_boundary(self, small_mesh_geometry):
        from qewton.visualization.plots.geometry import GeometryPlot

        plot = GeometryPlot(small_mesh_geometry, show_edges=True)
        backend_figure = Figure(plot, theme=self._theme()).draw()
        assert backend_figure.data[0].opacity == 0.55  # filled triangulation
        assert backend_figure.data[1].opacity == 0.13  # interior wireframe
        # boundary outline is a real geometric feature, not a stylistic
        # overlay - stays fully opaque regardless of the theme
        assert backend_figure.data[2].opacity == 1.0


class TestColorCycling:
    """get_color()/the palettes used to be built but never actually wired
    up - every LinePlot/ScatterPlot in a Figure got an identical, unthemed
    color regardless of how many were overlaid."""

    @staticmethod
    def _line_plot():
        from qewton.visualization.plots.data.curve import LinePlot

        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        return LinePlot(data, config, x=sample_axis, y=Y)

    def test_plot_color_index_is_none_before_being_added_to_a_figure(self):
        plot = self._line_plot()
        assert plot.color_index is None

    def test_figure_assigns_sequential_color_indices_in_add_order(self):
        plots = [self._line_plot(), self._line_plot(), self._line_plot()]
        Figure(plots)
        assert [p.color_index for p in plots] == [0, 1, 2]

    def test_two_line_plots_get_different_colors(self):
        theme = Theme()
        plot_a, plot_b = self._line_plot(), self._line_plot()
        backend_figure = Figure([plot_a, plot_b], theme=theme).draw()
        assert backend_figure.data[0].line.color == theme.get_color(0)
        assert backend_figure.data[1].line.color == theme.get_color(1)
        assert backend_figure.data[0].line.color != backend_figure.data[1].line.color

    def test_scatter_plot_without_a_color_spec_cycles_too(self):
        theme = Theme()
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = ScatterPlot(data, config, x=X, y=Y)
        Figure(plot)  # not used directly - just to occupy color_index 0
        plot2 = ScatterPlot(data, config, x=X, y=Y)
        backend_figure = Figure(plot2, theme=theme).draw()
        assert backend_figure.data[0].marker.color == theme.get_color(0)

    def test_scatter_plot_with_a_color_spec_is_not_overridden_by_cycling(self):
        """Data-driven coloring (a real ColorSpec) always wins - cycling is
        only ever a fallback for plots with no other coloring at all."""
        X, Y, C = Variable("x", 1), Variable("y", 1), Variable("c", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 3)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y * C))
        plot = ScatterPlot(data, config, x=X, y=Y, color=ColorSpec(C))
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].marker.color is not None
        # it's the per-point data array, not a single palette color string
        assert len(backend_figure.data[0].marker.color) == 10

    def test_bar_plot_cycles(self):
        from qewton.visualization.plots.data.samples import BarPlot

        theme = Theme()
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot_a = BarPlot(data, config, x=sample_axis, y=Y)
        plot_b = BarPlot(data, config, x=sample_axis, y=Y)
        backend_figure = Figure([plot_a, plot_b], theme=theme).draw()
        assert backend_figure.data[0].marker.color == theme.get_color(0)
        assert backend_figure.data[1].marker.color == theme.get_color(1)
