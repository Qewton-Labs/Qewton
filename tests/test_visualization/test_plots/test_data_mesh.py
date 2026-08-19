import numpy as np
import pytest

from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.mesh import MeshFieldPlot, MeshSurfacePlot, MeshVectorPlot
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

    def test_embedding_dim_is_always_3(self, circle_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        assert plot.embedding_dim == 3  # even for a 2D mesh - drawn as a colored surface

    def test_axis_titles_from_the_mesh_geometry_variable(self, circle_mesh_geometry):
        U = Variable("u", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        field = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(U))
        plot = MeshFieldPlot(field, config, color=ColorSpec(U))
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.scene.xaxis.title.text == "x_0"
        assert backend_figure.layout.scene.yaxis.title.text == "x_1"


class TestMeshSurfacePlot:
    def test_defaults_color_to_height_when_unset(self, circle_mesh_geometry):
        Z = Variable("z", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        z_values = (vertices[:, 0] ** 2)[:, None]
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(Z))
        plot = MeshSurfacePlot(z_values, config, z=Z)
        result = plot.evaluate()
        assert np.allclose(result.color, result.vertices[:, 2])

    def test_z_title_comes_from_its_own_axis_spec_not_the_mesh_variable(self, circle_mesh_geometry):
        """MeshSurfacePlot's z is a data value elevating a 2D mesh, unlike
        MeshFieldPlot where z genuinely is the mesh's own 3rd coordinate."""
        Height = Variable("elevation", 1)
        vertices = np.asarray(circle_mesh_geometry.mesh.vertices)
        z_values = np.zeros((len(vertices), 1))
        config = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(Height))
        plot = MeshSurfacePlot(z_values, config, z=Height)
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.scene.xaxis.title.text == "x_0"
        assert backend_figure.layout.scene.zaxis.title.text == "elevation"


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

    def test_draws_as_cone_in_3d_and_lines_in_2d(self, cylinder_mesh_geometry, circle_mesh_geometry):
        V3 = Variable("v", 3)
        vertices3 = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        vec3 = np.stack([-vertices3[:, 1], vertices3[:, 0], np.zeros(len(vertices3))], axis=1)
        cfg3 = DataConfiguration(GeometryAxes(cylinder_mesh_geometry), FeatureAxes(V3))
        b3 = Figure(MeshVectorPlot(vec3, cfg3, vector=VectorSpec(V3))).draw()
        assert b3.data[0].type == "cone"

        V2 = Variable("v", 2)
        vertices2 = np.asarray(circle_mesh_geometry.mesh.vertices)
        vec2 = np.stack([-vertices2[:, 1], vertices2[:, 0]], axis=1)
        cfg2 = DataConfiguration(GeometryAxes(circle_mesh_geometry), FeatureAxes(V2))
        b2 = Figure(MeshVectorPlot(vec2, cfg2, vector=VectorSpec(V2))).draw()
        assert b2.data[0].type == "scatter"
