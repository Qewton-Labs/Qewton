import numpy as np
import pytest

from qewton.config.axes import FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.points import PointCloudPlot
from qewton.visualization.plots.spec import ColorSpec


def _point_cloud_setup(dim):
    U = Variable("u", 1)
    points = np.random.rand(6, dim)
    geometry = PointCloud(Variable("x", dim), points)
    field = np.random.rand(6, 1)
    config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
    return field, config, geometry, U


class TestPointCloudPlot:
    def test_evaluate_returns_one_value_per_point_2d(self):
        field, config, geometry, U = _point_cloud_setup(2)
        plot = PointCloudPlot(field, config, color=ColorSpec(U))
        result = plot.evaluate()
        assert result.positions.shape == (6, 2)
        assert result.color.shape == (6,)

    def test_evaluate_returns_one_value_per_point_3d(self):
        field, config, geometry, U = _point_cloud_setup(3)
        plot = PointCloudPlot(field, config, color=ColorSpec(U))
        result = plot.evaluate()
        assert result.positions.shape == (6, 3)

    def test_embedding_dim_follows_the_geometry(self):
        field, config, geometry, U = _point_cloud_setup(3)
        plot = PointCloudPlot(field, config, color=ColorSpec(U))
        assert plot.embedding_dim == 3

    def test_rejects_a_vector_color_spec(self):
        V = Variable("v", 2)
        points = np.random.rand(4, 2)
        geometry = PointCloud(Variable("x", 2), points)
        data = np.random.rand(4, 2)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(V))
        with pytest.raises(ValueError, match="must be scalar"):
            PointCloudPlot(data, config, color=ColorSpec(V))

    def test_draws_as_scatter_in_2d_and_scatter3d_in_3d(self):
        field2, config2, _, U2 = _point_cloud_setup(2)
        b2 = Figure(PointCloudPlot(field2, config2, color=ColorSpec(U2))).draw()
        assert b2.data[0].type == "scatter"

        field3, config3, _, U3 = _point_cloud_setup(3)
        b3 = Figure(PointCloudPlot(field3, config3, color=ColorSpec(U3))).draw()
        assert b3.data[0].type == "scatter3d"

    def test_draws_with_axis_titles_from_the_geometry_variable(self):
        field, config, geometry, U = _point_cloud_setup(3)
        backend_figure = Figure(PointCloudPlot(field, config, color=ColorSpec(U))).draw()
        titles = {
            backend_figure.layout.scene.xaxis.title.text,
            backend_figure.layout.scene.yaxis.title.text,
            backend_figure.layout.scene.zaxis.title.text,
        }
        assert titles == {"$x_0$", "$x_1$", "$x_2$"}
