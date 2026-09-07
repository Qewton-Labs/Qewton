import numpy as np

from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.layout import Row
from qewton.visualization.plots.data.mesh import MeshFieldPlot
from qewton.visualization.plots.spec import ColorSpec, Scale


def _field_plot(geometry, fill_value=1.0, scale=None, **kwargs):
    U = Variable("u", 1)
    vertices = np.asarray(geometry.mesh.vertices)
    field = np.full((len(vertices), 1), fill_value)
    config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
    return MeshFieldPlot(field, config, color=ColorSpec(U, scale=scale), **kwargs)


class TestColorbarPositioning:
    """Several panels each showing a colorbar (own Scale, or a shared one
    only its claimant shows) must not collide at one fixed page position -
    see graphs.py's Reference/Prediction/Error Row, or any FacetSpec."""

    def test_two_panels_each_own_scale_get_distinct_x_positions(
        self, circle_mesh_geometry
    ):
        """2D mesh fields draw via FilledMeshArtist (xaxis/yaxis domain,
        not a 3D scene) - the other branch of _colorbar_position."""
        a = _field_plot(circle_mesh_geometry, fill_value=1.0)
        b = _field_plot(circle_mesh_geometry, fill_value=2.0)
        backend_figure = Figure(Row(a, b)).draw()
        colorbar_traces = [
            tr for tr in backend_figure.data if getattr(tr, "marker", None) is not None
            and tr.marker.showscale
        ]
        assert len(colorbar_traces) == 2
        x_positions = {tr.marker.colorbar.x for tr in colorbar_traces}
        assert len(x_positions) == 2

    def test_two_3d_panels_each_own_scale_get_distinct_positions(
        self, cylinder_mesh_geometry
    ):
        """3D mesh fields draw via SurfaceMeshArtist/go.Mesh3d into a
        `scene` subplot - the domain-attribute branch of
        _colorbar_position."""
        a = _field_plot(cylinder_mesh_geometry, fill_value=1.0)
        b = _field_plot(cylinder_mesh_geometry, fill_value=2.0)
        backend_figure = Figure(Row(a, b)).draw()
        colorbar_traces = [tr for tr in backend_figure.data if tr.type == "mesh3d"]
        assert len(colorbar_traces) == 2
        assert all(tr.showscale for tr in colorbar_traces)
        x_positions = [tr.colorbar.x for tr in colorbar_traces]
        assert x_positions[0] < x_positions[1]

    def test_colorbar_sits_in_the_gap_not_on_the_next_panel(self, cylinder_mesh_geometry):
        """The colorbar's x is the actual midpoint of the gap between this
        cell and the next one (read from Plotly's own resolved domains),
        not a fixed offset that can land past the next cell's own left
        edge - and its thickness is a small, fixed pixel width so it can't
        outgrow that gap regardless of the figure's overall size."""
        a = _field_plot(cylinder_mesh_geometry, fill_value=1.0)
        b = _field_plot(cylinder_mesh_geometry, fill_value=2.0)
        backend_figure = Figure(Row(a, b)).draw()
        cell_a_domain = backend_figure.get_subplot(1, 1).domain.x
        cell_b_domain = backend_figure.get_subplot(1, 2).domain.x
        trace_a = next(tr for tr in backend_figure.data if tr.type == "mesh3d" and tr.showscale)
        assert cell_a_domain[1] < trace_a.colorbar.x < cell_b_domain[0]
        assert trace_a.colorbar.thickness == 14
        assert trace_a.colorbar.thicknessmode == "pixels"

    def test_single_panel_keeps_the_old_default_position(self, circle_mesh_geometry):
        """No grid to place within - the pre-existing fixed position is
        left exactly as it was."""
        shared = Scale()
        plot = _field_plot(circle_mesh_geometry, scale=shared)
        backend_figure = Figure(plot).draw()
        colorbar_trace = next(
            tr for tr in backend_figure.data
            if getattr(tr, "marker", None) is not None and tr.marker.showscale
        )
        assert colorbar_trace.marker.colorbar.x == 1.02

    def test_a_scale_shared_by_the_first_two_of_three_panels_sits_after_the_second(
        self, cylinder_mesh_geometry
    ):
        """A Scale shared by panels 1 and 2 (but not 3) belongs after panel
        2, not wedged between 1 and 2 just because panel 1 is whichever one
        happens to claim the colorbar first."""
        shared = Scale()
        a = _field_plot(cylinder_mesh_geometry, fill_value=1.0, scale=shared)
        b = _field_plot(cylinder_mesh_geometry, fill_value=2.0, scale=shared)
        c = _field_plot(cylinder_mesh_geometry, fill_value=3.0)
        backend_figure = Figure(Row(a, b, c)).draw()

        cell2_domain = backend_figure.get_subplot(1, 2).domain.x
        cell3_domain = backend_figure.get_subplot(1, 3).domain.x
        mesh_traces = [tr for tr in backend_figure.data if tr.type == "mesh3d"]
        showing = [tr for tr in mesh_traces if tr.showscale]
        assert len(showing) == 2  # the shared claimant, plus panel 3's own
        shared_trace = showing[0]  # panel 1's trace claims the shared scale first
        assert cell2_domain[1] < shared_trace.colorbar.x < cell3_domain[0]

    def test_a_shared_scales_non_claimant_still_shows_no_colorbar_in_a_grid(
        self, circle_mesh_geometry
    ):
        """Positioning is orthogonal to Scale.claim_colorbar()'s first-
        come-first-served rule - a panel that doesn't claim the shared
        Scale still shows none, even though it's in its own grid cell."""
        shared = Scale()
        a = _field_plot(circle_mesh_geometry, fill_value=1.0, scale=shared)
        b = _field_plot(circle_mesh_geometry, fill_value=2.0, scale=shared)
        backend_figure = Figure(Row(a, b)).draw()
        fill_traces = [
            tr for tr in backend_figure.data
            if getattr(tr, "marker", None) is not None and hasattr(tr.marker, "showscale")
        ]
        showscale_flags = [bool(tr.marker.showscale) for tr in fill_traces]
        assert showscale_flags.count(True) == 1
