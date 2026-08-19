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
