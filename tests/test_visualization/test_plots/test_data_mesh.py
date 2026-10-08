import numpy as np
import pytest

from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import (
    MeshFieldPlot,
    MeshSurfacePlot,
    MeshVectorPlot,
)
from qewton.visualization.plots.spec import ColorSpec, VectorSpec


class TestMeshFieldPlot:
    def test_evaluate_returns_one_value_per_vertex(self, circle_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = (np.sin(vertices[:, 0]) * np.cos(vertices[:, 1]))[:, None]
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        result = plot.evaluate()
        assert result.color.shape == (len(vertices),)
        assert result.vertices.shape == (len(vertices), 2)

    def test_rejects_a_vector_color_spec(self, circle_mesh_geometry):
        V = Variable("v", 2)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        data = np.tile(vertices, 1)
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(V))
        with pytest.raises(ValueError, match="must be scalar"):
            MeshFieldPlot(data, config, color=ColorSpec(V))

    def test_embedding_dim_is_2_for_a_2d_mesh(self, circle_mesh_geometry):
        """A 2D mesh field draws as a genuinely cartesian triangulation
        (FilledMeshArtist) - it can sit beside a HeatmapPlot in one Figure
        row, unlike go.Mesh3d's `scene` cell."""
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        assert plot.embedding_dim == 2

    def test_embedding_dim_is_3_for_a_3d_mesh(self, cylinder_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        assert plot.embedding_dim == 3

    def test_axis_titles_from_the_mesh_geometry_variable(self, circle_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.xaxis.title.text == "$x_1$"
        assert backend_figure.layout.yaxis.title.text == "$x_2$"


class TestFilledMeshArtist:
    """The 2D FilledMeshArtist rendering path - one go.Scatter(fill="toself")
    trace per value bin, plus an invisible colorbar-carrier trace, plus an
    optional wireframe trace."""

    @staticmethod
    def _field_plot(circle_mesh_geometry, **kwargs):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = (vertices[:, 0] ** 2 + vertices[:, 1] ** 2)[:, None]
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        return MeshFieldPlot(field, config, color=ColorSpec(U), **kwargs)

    def test_draws_n_bins_fill_traces_plus_edges_and_colorbar(self, circle_mesh_geometry):
        plot = self._field_plot(circle_mesh_geometry, show_edges=True)
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.data) == plot.n_bins + 2
        assert backend_figure.data[plot.n_bins].line.width == 0.5  # the wireframe

    def test_without_edges_omits_the_wireframe_trace(self, circle_mesh_geometry):
        plot = self._field_plot(circle_mesh_geometry, show_edges=False)
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.data) == plot.n_bins + 1

    def test_2d_fills_are_outlined_in_their_own_color(self, circle_mesh_geometry):
        plot = self._field_plot(circle_mesh_geometry, show_edges=False)
        backend_figure = Figure(plot).draw()
        for trace in backend_figure.data[: plot.n_bins]:
            assert trace.line.color == trace.fillcolor

    def test_n_bins_is_configurable(self, circle_mesh_geometry):
        plot = self._field_plot(circle_mesh_geometry, show_edges=False, n_bins=8)
        backend_figure = Figure(plot).draw()
        assert plot.n_bins == 8
        assert len(backend_figure.data) == 9

    def test_colorbar_carrier_trace_reflects_the_value_range(self, circle_mesh_geometry):
        plot = self._field_plot(circle_mesh_geometry, show_edges=False)
        backend_figure = Figure(plot).draw()
        result = plot.evaluate()
        # The bin range is over per-triangle means, not raw per-vertex
        # values - narrower than the vertex range, same as the artist.
        cells = np.asarray(plot.render_cells())
        triangle_values = np.asarray(result.color)[cells].mean(axis=1)
        colorbar_trace = backend_figure.data[-1]
        assert colorbar_trace.marker.cmin == pytest.approx(float(triangle_values.min()))
        assert colorbar_trace.marker.cmax == pytest.approx(float(triangle_values.max()))
        assert colorbar_trace.marker.showscale is True

    def test_redraw_does_not_change_trace_count(self, circle_mesh_geometry):
        plot = self._field_plot(circle_mesh_geometry)
        fig = Figure(plot)
        backend_figure = fig.draw()
        n_traces = len(backend_figure.data)
        fig.draw()
        assert len(backend_figure.data) == n_traces

    def test_all_triangles_land_in_one_bin_for_a_uniform_field(
        self, circle_mesh_geometry
    ):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.ones((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U), show_edges=False)
        backend_figure = Figure(plot).draw()
        fill_traces = backend_figure.data[: plot.n_bins]
        non_empty = [trace for trace in fill_traces if len(trace.x) > 0]
        assert len(non_empty) == 1

    def test_composes_with_a_heatmap_plot_in_one_row(self, circle_mesh_geometry):
        """The motivating case: a 2D mesh field is
        genuinely cartesian, so it can share a Row with a HeatmapPlot."""
        from qewton.config.axes import BatchAxes
        from qewton.visualization.layout import Row
        from qewton.visualization.plots.data.grid import HeatmapPlot

        mesh_plot = self._field_plot(circle_mesh_geometry, show_edges=False)

        X, Y, C = Variable("x", 1), Variable("y", 1), Variable("c", 1)
        x_axis, y_axis = BatchAxes(4), BatchAxes(4)
        heatmap_data = np.random.rand(4, 4, 1)
        heatmap_config = DataConfiguration(x_axis, y_axis, FeatureAxes(C))
        heatmap_plot = HeatmapPlot(
            heatmap_data, heatmap_config, x=x_axis, y=y_axis, color=C
        )

        fig = Figure(Row(mesh_plot, heatmap_plot))
        assert fig.grid_shape() == (1, 2)
        backend_figure = fig.draw()
        assert len(backend_figure.data) > 0


class TestMeshSurfacePlot:
    def test_defaults_color_to_height_when_unset(self, circle_mesh_geometry):
        Z = Variable("z", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        z_values = (vertices[:, 0] ** 2)[:, None]
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(Z))
        plot = MeshSurfacePlot(z_values, config, z=Z)
        result = plot.evaluate()
        assert np.allclose(result.color, result.vertices[:, 2])

    def test_z_title_comes_from_its_own_axis_spec_not_the_mesh_variable(
        self, circle_mesh_geometry
    ):
        """MeshSurfacePlot's z is a data value elevating a 2D mesh, unlike
        MeshFieldPlot where z genuinely is the mesh's own 3rd coordinate."""
        Height = Variable("elevation", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        z_values = np.zeros((len(vertices), 1))
        config = DataConfiguration(
            GeometryAxes(circle_mesh_geometry), FeatureAxes(Height)
        )
        plot = MeshSurfacePlot(z_values, config, z=Height)
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.scene.xaxis.title.text == "$x_1$"
        assert backend_figure.layout.scene.zaxis.title.text == "$elevation$"


class TestMeshVectorPlot:
    def test_evaluate_returns_one_vector_per_vertex(self, cylinder_mesh_geometry):
        V3 = Variable("v", 3)
        vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        vec = np.stack([-vertices[:, 1], vertices[:, 0], np.zeros(len(vertices))], axis=1)
        config = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(V3))
        plot = MeshVectorPlot(vec, config, vector=VectorSpec(V3))
        result = plot.evaluate()
        assert result.vectors.shape == (len(vertices), 3)
        assert result.magnitude.shape == (len(vertices),)

    def test_component_count_must_match_mesh_dimension(self, cylinder_mesh_geometry):
        V2 = Variable("v", 2)  # wrong: mesh is 3D
        vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        data = np.zeros((len(vertices), 2))
        config = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(V2))
        with pytest.raises(ValueError, match="components but the mesh is"):
            MeshVectorPlot(data, config, vector=VectorSpec(V2))

    def test_subsample_decimates_after_scale_and_normalize(self, cylinder_mesh_geometry):
        V3 = Variable("v", 3)
        vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        vec = np.stack([-vertices[:, 1], vertices[:, 0], np.zeros(len(vertices))], axis=1)
        config = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(V3))

        full = MeshVectorPlot(vec, config, vector=VectorSpec(V3)).evaluate()
        sub = MeshVectorPlot(vec, config, vector=VectorSpec(V3, subsample=3)).evaluate()
        assert len(sub.positions) == len(full.positions[::3])
        assert np.allclose(sub.positions, full.positions[::3])

    def test_draws_as_cone_in_3d_and_lines_in_2d(
        self, cylinder_mesh_geometry, circle_mesh_geometry
    ):
        V3 = Variable("v", 3)
        vertices3 = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        vec3 = np.stack(
            [-vertices3[:, 1], vertices3[:, 0], np.zeros(len(vertices3))], axis=1
        )
        cfg3 = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(V3))
        b3 = Figure(MeshVectorPlot(vec3, cfg3, vector=VectorSpec(V3))).draw()
        assert b3.data[0].type == "cone"

        V2 = Variable("v", 2)
        vertices2 = np.asarray(circle_mesh_geometry.mesh.vertices)
        vec2 = np.stack([-vertices2[:, 1], vertices2[:, 0]], axis=1)
        cfg2 = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(V2))
        b2 = Figure(MeshVectorPlot(vec2, cfg2, vector=VectorSpec(V2))).draw()
        assert b2.data[0].type == "scatter"


class TestCurveField:
    """MeshFieldPlot on meshes of line segments, e.g. a domain boundary."""

    @staticmethod
    def _boundary_plot(**kwargs):
        from qewton.geometries.continuous.domains_2d.rectangle import Rectangle
        from qewton.geometries.discrete.mesh_geometry import MeshGeometry

        U = Variable("u", 1)
        X = Variable("x", 2)
        square = Rectangle(X, [0.0, 0.0], 1.0, 1.0)
        boundary = MeshGeometry(X, square.boundary.create_mesh(0.25).mesh)
        vertices = np.asarray(boundary.mesh.vertices)
        field = vertices.sum(axis=1, keepdims=True)
        config = DataConfiguration(GeometryAxes(boundary), FeatureAxes(U))
        return MeshFieldPlot(field, config, color=ColorSpec(U), **kwargs)

    @staticmethod
    def _curve_3d_plot():
        from qewton.geometries.discrete.mesh import Mesh
        from qewton.geometries.discrete.mesh_geometry import MeshGeometry

        U = Variable("u", 1)
        s = np.linspace(0, 2 * np.pi, 9)
        vertices = np.stack([np.cos(s), np.sin(s), s], axis=1)
        helix = MeshGeometry(
            Variable("y", 3),
            Mesh(vertices=vertices, cells=[[i, i + 1] for i in range(8)]),
        )
        config = DataConfiguration(GeometryAxes(helix), FeatureAxes(U))
        return MeshFieldPlot(s[:, None], config, color=ColorSpec(U))

    def test_line_cells_have_cell_dim_one(self):
        assert self._boundary_plot().cell_dim == 1

    def test_2d_curve_draws_one_line_trace_per_bin_plus_colorbar(self):
        plot = self._boundary_plot(n_bins=8)
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.data) == 9
        assert all(trace.mode == "lines" for trace in backend_figure.data[:8])
        drawn = sum(trace.x.count(None) for trace in backend_figure.data[:8])
        assert drawn == len(plot.mesh.cells)

    def test_2d_curve_bins_follow_the_values(self):
        plot = self._boundary_plot(n_bins=4)
        backend_figure = Figure(plot).draw()
        lowest = backend_figure.data[0]
        points = np.array(
            [(x, y) for x, y in zip(lowest.x, lowest.y) if x is not None]
        )
        # u = x_1 + x_2 ranges over [0, 2]; the lowest bin holds the corner (0, 0)
        assert points.sum(axis=1).max() <= 0.75

    def test_3d_curve_colors_one_line_per_vertex(self):
        plot = self._curve_3d_plot()
        backend_figure = Figure(plot).draw()
        assert len(backend_figure.data) == 1
        trace = backend_figure.data[0]
        assert trace.type == "scatter3d"
        assert len(trace.line.color) == len(trace.x)
