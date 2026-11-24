"""GeometryPlot: a geometry drawn without data."""

import numpy as np
import pytest

from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.mesh_geometry import MeshGeometry
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.visualization import Figure, GeometryPlot, Overlay, auto_layout
from qewton.visualization.plots.result import LineResult, MarkerResult, SurfaceResult
from qewton.visualization.renderers import PlotlyRenderer

from recording_renderer import RecordingRenderer

X = Variable("x", 2)


def _square():
    return Rectangle(X, [0.0, 0.0], 1.0, 1.0)


class TestPrimitives:
    def test_points_are_markers(self):
        cloud = PointCloud(X, np.random.default_rng(0).random((7, 2)))
        (result,) = GeometryPlot(cloud).evaluate()
        assert isinstance(result, MarkerResult)
        assert result.positions.shape == (7, 2)

    def test_a_curve_is_a_line(self):
        boundary = MeshGeometry(X, _square().create_mesh(0.5).mesh.get_boundary_mesh())
        (result,) = GeometryPlot(boundary).evaluate()
        assert isinstance(result, LineResult)

    def test_a_surface_is_filled_with_edges(self):
        (result,) = GeometryPlot(_square().create_mesh(0.5)).evaluate()
        assert isinstance(result, SurfaceResult)
        assert result.show_edges and result.values is None

    def test_a_volume_is_drawn_by_its_boundary(self, cylinder_mesh_geometry):
        plot = GeometryPlot(cylinder_mesh_geometry)
        (result,) = plot.evaluate()
        assert plot.dim == 3
        np.testing.assert_array_equal(result.cells, cylinder_mesh_geometry.boundary_cells)

    def test_a_grid_is_drawn_by_its_cells(self):
        (result,) = GeometryPlot(GridGeometry.build_unit_grid(X, 4)).evaluate()
        assert isinstance(result, SurfaceResult)
        assert len(result.cells) == 18

    def test_a_continuous_geometry_is_meshed(self):
        plot = GeometryPlot(_square(), max_vertex_distance=0.25)
        assert len(plot.geometry.coordinates()) > len(_square().create_mesh(0.5).coordinates())

    def test_a_geometry_with_one_coordinate_raises(self):
        with pytest.raises(ValueError, match="2 or 3 coordinates"):
            GeometryPlot(Interval(Variable("t", 1), 0.0, 1.0).create_mesh(0.25))


class TestDrawing:
    def test_titles_and_label_come_from_the_variable(self, small_mesh_geometry):
        plot = GeometryPlot(small_mesh_geometry)
        assert [plot.axis_title(c) for c in "xy"] == ["$p_1$", "$p_2$"]
        assert plot.evaluate()[0].label == "$p$"

    def test_the_axes_are_trained_on_the_points(self, small_mesh_geometry):
        figure = Figure(GeometryPlot(small_mesh_geometry), renderer=RecordingRenderer())
        backend = figure.draw()
        (axes,) = backend["axes"].values()
        assert axes["x"][1] == (0.0, 1.0)

    @pytest.mark.parametrize("fixture", ["small_mesh_geometry", "cylinder_mesh_geometry"])
    def test_drawn_with_plotly_without_a_colorbar(self, fixture, request):
        backend = Figure(
            GeometryPlot(request.getfixturevalue(fixture)), renderer=PlotlyRenderer()
        ).draw()
        assert len(backend.data) > 0
        assert not any(getattr(t, "showscale", False) for t in backend.data)

    def test_points_go_over_a_filled_geometry(self, small_mesh_geometry):
        variable = small_mesh_geometry.variable
        cloud = PointCloud(variable, np.random.default_rng(0).random((5, 2)))
        points, domain = GeometryPlot(cloud), GeometryPlot(small_mesh_geometry)
        layout = auto_layout([points, domain])
        assert isinstance(layout, Overlay)
        assert layout.plots == [domain, points]
