import numpy as np
import pytest

from qewton.algorithms.building_blocks.geometry import MeshInterpolationNode
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.discrete.volume_grid import VolumeGridGeometry
from qewton.visualization.figure import Figure
from qewton.visualization.plots.data.grid import (
    EmbeddedGridPlot,
    HeatmapPlot,
    ImagePlot,
    QuiverPlot,
    SurfacePlot,
)
from qewton.visualization.plots.spec import AxisSpec, ColorSpec, FixedSpec, VectorSpec


def _heatmap_setup():
    X, Y, C = Variable("x", 1), Variable("y", 1), Variable("c", 1)
    x_axis, y_axis = BatchAxes(8), BatchAxes(6)
    data = np.random.rand(8, 6, 1)
    config = DataConfiguration(x_axis, y_axis, FeatureAxes(C))
    return data, config, x_axis, y_axis, C


class TestHeatmapPlot:
    def test_evaluate_orients_values_y_then_x(self):
        data, config, x_axis, y_axis, C = _heatmap_setup()
        plot = HeatmapPlot(data, config, x=x_axis, y=y_axis, color=C)
        result = plot.evaluate()
        assert result.values.shape == (6, 8, 1)  # (y, x, channel) after the x/y-first transpose

    def test_x_and_y_must_differ(self):
        data, config, x_axis, y_axis, C = _heatmap_setup()
        plot = HeatmapPlot(data, config, x=x_axis, y=x_axis, color=C)
        with pytest.raises(ValueError):
            plot.evaluate()

    def test_draws_with_correct_axis_titles_and_log_scale(self):
        data, config, x_axis, y_axis, C = _heatmap_setup()
        plot = HeatmapPlot(data, config, x=AxisSpec(x_axis, log_scale=True), y=y_axis, color=C)
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.xaxis.title.text == str(x_axis)
        assert backend_figure.layout.xaxis.type == "log"
        assert backend_figure.layout.yaxis.type == "linear"


class TestSurfacePlot:
    def test_embedding_dim_is_3(self):
        data, config, x_axis, y_axis, C = _heatmap_setup()
        plot = SurfacePlot(data, config, x=x_axis, y=y_axis, z=C)
        assert plot.embedding_dim == 3

    def test_xy_titles_land_on_the_scene_not_cartesian_axes(self):
        """Regression: SurfaceArtist used to call update_xaxes/update_yaxes
        for a trace that lives in a `scene` subplot (go.Surface), which
        silently drops those titles - only update_scenes reaches the right
        place."""
        data, config, x_axis, y_axis, C = _heatmap_setup()
        plot = SurfacePlot(data, config, x=x_axis, y=y_axis, z=C)
        backend_figure = Figure(plot).draw()
        assert backend_figure.layout.scene.xaxis.title.text == str(x_axis)
        assert backend_figure.layout.scene.yaxis.title.text == str(y_axis)
        assert backend_figure.layout.scene.zaxis.title.text == "c"


class TestImagePlot:
    def test_evaluate_and_draw(self):
        image = np.random.rand(10, 10, 3)
        x_axis, y_axis = BatchAxes(10), BatchAxes(10)
        config = DataConfiguration(x_axis, y_axis, FeatureAxes(Variable("rgb", 3)))
        plot = ImagePlot(image, config, x=x_axis, y=y_axis)
        backend_figure = Figure(plot).draw()
        assert backend_figure.data[0].type == "image"


@pytest.fixture
def resampled_scalar_field(cylinder_mesh_geometry):
    mesh_geometry = cylinder_mesh_geometry
    U = Variable("u", 1)
    i, j, k = Variable("i", 1), Variable("j", 1), Variable("k", 1)
    grid = VolumeGridGeometry(mesh_geometry, i * j * k, resolution=(6, 6, 6))
    node = MeshInterpolationNode(mesh_geometry, U, grid, backend=mesh_geometry.backend)
    vertices = np.asarray(mesh_geometry.mesh.vertices)
    field = mesh_geometry.backend.build_tensor((vertices**2).sum(axis=1))
    resampled = node.forward(field)[..., None]
    config = DataConfiguration(GeometryAxes(grid), FeatureAxes(U))
    return resampled, config, grid, U, i


class TestEmbeddedGridPlot:
    def test_requires_exactly_two_grid_dimensions_after_controls(self, resampled_scalar_field):
        resampled, config, grid, U, i = resampled_scalar_field
        with pytest.raises(ValueError, match="needs exactly 2 grid dimensions"):
            EmbeddedGridPlot(resampled, config, color=ColorSpec(U))

    def test_evaluate_positions_and_masks_outside_points_as_nan(self, resampled_scalar_field):
        resampled, config, grid, U, i = resampled_scalar_field
        plot = EmbeddedGridPlot(
            resampled, config, color=ColorSpec(U),
            controls=[FixedSpec(init_state=3, n_dimensions=1, variable_or_axes=i)],
        )
        result = plot.evaluate()
        assert result.values.shape == (6, 6)
        assert result.x.shape == (6, 6)
        point_filter = np.asarray(grid.point_filter)[3, :, :, 0]
        assert np.all(np.isnan(np.asarray(result.color)[~point_filter]))

    def test_draws_with_axis_titles_from_the_source_mesh_variable(self, resampled_scalar_field):
        """EmbeddedGridPlot's own geometry.variable names the grid's
        parametrization (i/j/k), not the space the surface is drawn in -
        titles must come from the source mesh's variable instead."""
        resampled, config, grid, U, i = resampled_scalar_field
        plot = EmbeddedGridPlot(
            resampled, config, color=ColorSpec(U),
            controls=[FixedSpec(init_state=3, n_dimensions=1, variable_or_axes=i)],
        )
        backend_figure = Figure(plot).draw()
        titles = {
            backend_figure.layout.scene.xaxis.title.text,
            backend_figure.layout.scene.yaxis.title.text,
            backend_figure.layout.scene.zaxis.title.text,
        }
        assert titles == {"x_0", "x_1", "x_2"}  # Cylinder's Variable("x", 3)


class TestQuiverPlot:
    def test_requires_geometry_with_3_coordinate_components(self):
        X, Y, C = Variable("x", 1), Variable("y", 1), Variable("c", 1)
        x_axis, y_axis = BatchAxes(4), BatchAxes(4)
        data = np.random.rand(4, 4, 1)
        config = DataConfiguration(x_axis, y_axis, FeatureAxes(C))
        with pytest.raises(AttributeError):
            # no GeometryAxes at all in this config
            QuiverPlot(data, config, vector=VectorSpec(Variable("v", 3)))

    def test_drops_points_outside_the_source_mesh(self, cylinder_mesh_geometry):
        mesh_geometry = cylinder_mesh_geometry
        V3 = Variable("v", 3)
        i, j, k = Variable("i", 1), Variable("j", 1), Variable("k", 1)
        grid = VolumeGridGeometry(mesh_geometry, i * j * k, resolution=(5, 5, 5))
        node = MeshInterpolationNode(mesh_geometry, V3, grid, backend=mesh_geometry.backend)
        vertices = np.asarray(mesh_geometry.mesh.vertices)
        vec = np.stack([-vertices[:, 1], vertices[:, 0], np.zeros(len(vertices))], axis=1)
        resampled = node.forward(mesh_geometry.backend.build_tensor(vec))
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(V3))
        plot = QuiverPlot(resampled, config, vector=VectorSpec(V3, scale=0.1))
        result = plot.evaluate()

        n_valid = int(np.asarray(grid.point_filter).astype(bool).sum())
        assert len(result.positions) == n_valid
        assert not np.isnan(result.vectors).any()
