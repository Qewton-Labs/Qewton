import numpy as np
import pytest

from qewton.algorithms.building_blocks.geometry import MeshInterpolationNode
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.geometries.discrete.volume_grid import VolumeGridGeometry
from qewton.visualization.auto import auto_plot
from qewton.visualization.plots.data.curve import LinePlot
from qewton.visualization.plots.data.grid import EmbeddedGridPlot, QuiverPlot
from qewton.visualization.plots.data.mesh import MeshFieldPlot, MeshVectorPlot
from qewton.visualization.plots.data.points import PointCloudPlot
from qewton.visualization.plots.data.samples import BarPlot, ScatterPlot
from qewton.visualization.plots.spec import FixedSpec, SliderSpec, VariableSpec


class TestLeftoverAxesBecomeSliders:
    def test_leftover_batch_axis_next_to_a_mesh_field_gets_a_default_slider(self, small_mesh_geometry):
        U = Variable("u", 1)
        step_axis = BatchAxes(5)
        n_vertices = small_mesh_geometry.mesh.vertices.shape[0]
        config = DataConfiguration(step_axis, GeometryAxes(small_mesh_geometry), FeatureAxes(U))
        data = np.zeros((5, n_vertices, 1))
        plot = auto_plot(data, config)
        assert isinstance(plot, MeshFieldPlot)
        assert len(plot.controls) == 1
        assert isinstance(plot.controls[0], SliderSpec)
        assert plot.controls[0].variable_or_axes is step_axis
        assert plot.controls[0].minimum == 0 and plot.controls[0].maximum == 4

    def test_an_explicit_control_for_the_axis_is_not_duplicated(self, small_mesh_geometry):
        U = Variable("u", 1)
        step_axis = BatchAxes(5)
        n_vertices = small_mesh_geometry.mesh.vertices.shape[0]
        config = DataConfiguration(step_axis, GeometryAxes(small_mesh_geometry), FeatureAxes(U))
        data = np.zeros((5, n_vertices, 1))
        own_control = FixedSpec(init_state=2, n_dimensions=1, variable_or_axes=step_axis)
        plot = auto_plot(data, config, controls=[own_control])
        assert plot.controls == [own_control]

    def test_line_plot_uses_the_first_leftover_axis_as_domain_and_sliders_the_rest(self):
        Y = Variable("y", 1)
        domain_axis, step_axis = BatchAxes(20), BatchAxes(3)
        data = np.random.rand(3, 20, 1)
        config = DataConfiguration(step_axis, domain_axis, FeatureAxes(Y))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        assert plot.x.variable_or_axes is step_axis
        assert len(plot.controls) == 1
        assert plot.controls[0].variable_or_axes is domain_axis

    def test_scatter_plot_leaves_remaining_axes_unresolved(self):
        """ScatterPlot flattens whatever's left into its implicit samples
        axis - it must not get an unwanted slider."""
        X, Y = Variable("x", 1), Variable("y", 1)
        batch_axis, sample_axis = BatchAxes(4), BatchAxes(10)
        data = np.random.randn(4, 10, 2)
        config = DataConfiguration(batch_axis, sample_axis, FeatureAxes(X * Y))
        plot = auto_plot(data, config)
        assert isinstance(plot, ScatterPlot)
        assert plot.controls == []


class TestMeshDispatch:
    def test_scalar_variable_becomes_mesh_field_plot(self, small_mesh_geometry):
        U = Variable("u", 1)
        config = DataConfiguration(GeometryAxes(small_mesh_geometry), FeatureAxes(U))
        field = np.zeros((small_mesh_geometry.mesh.vertices.shape[0], 1))
        plot = auto_plot(field, config)
        assert isinstance(plot, MeshFieldPlot)
        assert plot.color.variable_or_axes is U

    def test_vector_matching_mesh_dim_becomes_mesh_vector_plot(self, small_mesh_geometry):
        V = Variable("v", 2)  # small_mesh_geometry is 2D
        config = DataConfiguration(GeometryAxes(small_mesh_geometry), FeatureAxes(V))
        vectors = np.zeros((small_mesh_geometry.mesh.vertices.shape[0], 2))
        plot = auto_plot(vectors, config)
        assert isinstance(plot, MeshVectorPlot)
        assert plot.vector.variable_or_axes is V

    def test_wrong_dim_raises_a_clear_error(self, small_mesh_geometry):
        W = Variable("w", 3)  # neither scalar nor matching the 2D mesh
        config = DataConfiguration(GeometryAxes(small_mesh_geometry), FeatureAxes(W))
        data = np.zeros((small_mesh_geometry.mesh.vertices.shape[0], 3))
        with pytest.raises(ValueError, match="dim=3"):
            auto_plot(data, config)

    def test_two_distinct_scalar_variables_become_a_mesh_field_plot_with_a_selector(self, small_mesh_geometry):
        temperature, pressure = Variable("temperature", 1), Variable("pressure", 1)
        n = small_mesh_geometry.mesh.vertices.shape[0]
        config = DataConfiguration(
            GeometryAxes(small_mesh_geometry), FeatureAxes(temperature * pressure)
        )
        data = np.zeros((n, 2))
        plot = auto_plot(data, config)
        assert isinstance(plot, MeshFieldPlot)
        assert isinstance(plot.color.variable_or_axes, Variable)
        assert plot.color.variable_or_axes in (temperature, pressure)

    def test_distinct_variables_with_different_dims_raise_a_clear_error(self, small_mesh_geometry):
        """A scalar and a mesh-matching vector can't share one VariableSpec -
        selecting one would leave the Plot needing the other's role."""
        scalar, vector = Variable("s", 1), Variable("v", 2)
        n = small_mesh_geometry.mesh.vertices.shape[0]
        config = DataConfiguration(
            GeometryAxes(small_mesh_geometry), FeatureAxes(scalar * vector)
        )
        data = np.zeros((n, 3))
        with pytest.raises(ValueError, match="different dims"):
            auto_plot(data, config)


@pytest.fixture
def resampled_grid(cylinder_mesh_geometry):
    U = Variable("u", 1)
    i, j, k = Variable("i", 1), Variable("j", 1), Variable("k", 1)
    grid = VolumeGridGeometry(cylinder_mesh_geometry, i * j * k, resolution=(4, 4, 4))
    node = MeshInterpolationNode(cylinder_mesh_geometry, U, grid, backend=cylinder_mesh_geometry.backend)
    vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
    field = cylinder_mesh_geometry.backend.build_tensor((vertices**2).sum(axis=1))
    resampled = node.forward(field)[..., None]
    config = DataConfiguration(GeometryAxes(grid), FeatureAxes(U))
    return resampled, config, grid, U, i


class TestGridDispatch:
    def test_scalar_variable_becomes_embedded_grid_plot(self, resampled_grid):
        resampled, config, grid, U, i = resampled_grid
        plot = auto_plot(
            resampled, config,
            controls=[FixedSpec(init_state=2, n_dimensions=1, variable_or_axes=i)],
        )
        assert isinstance(plot, EmbeddedGridPlot)
        assert plot.color.variable_or_axes is U

    def test_three_component_variable_becomes_quiver_plot(self, cylinder_mesh_geometry):
        V3 = Variable("v", 3)
        i, j, k = Variable("i", 1), Variable("j", 1), Variable("k", 1)
        grid = VolumeGridGeometry(cylinder_mesh_geometry, i * j * k, resolution=(4, 4, 4))
        node = MeshInterpolationNode(cylinder_mesh_geometry, V3, grid, backend=cylinder_mesh_geometry.backend)
        vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        vec = np.stack([-vertices[:, 1], vertices[:, 0], np.zeros(len(vertices))], axis=1)
        resampled = node.forward(cylinder_mesh_geometry.backend.build_tensor(vec))
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(V3))
        plot = auto_plot(resampled, config)
        assert isinstance(plot, QuiverPlot)
        assert plot.vector.variable_or_axes is V3


class TestFlatDispatch:
    def test_one_scalar_leaf_and_one_other_axis_becomes_line_plot(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        assert plot.x.variable_or_axes is sample_axis
        assert plot.y.variable_or_axes is Y

    def test_two_scalar_leaves_becomes_scatter_plot(self):
        X, Y = Variable("x", 1), Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y))
        plot = auto_plot(data, config)
        assert isinstance(plot, ScatterPlot)
        assert plot.x.variable_or_axes is X
        assert plot.y.variable_or_axes is Y

    def test_three_distinct_scalar_variables_with_a_domain_axis_becomes_a_line_plot_with_a_selector(self):
        X, Y, Z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 3)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y * Z))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        assert plot.x.variable_or_axes is sample_axis
        assert isinstance(plot.y.variable_or_axes, Variable)
        assert plot.y.variable_or_axes in (X, Y, Z)  # whichever the selector defaults to

    def test_three_distinct_scalar_variables_without_a_domain_axis_raise_a_clear_error(self):
        """No BatchAxes at all - nothing to use as the LinePlot's x, so
        there's no fallback role to build a selector-driven plot around."""
        X, Y, Z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        data = np.random.randn(1, 3)
        config = DataConfiguration(FeatureAxes(X * Y * Z))
        with pytest.raises(ValueError, match="bundles multiple distinct"):
            auto_plot(data, config)

    def test_one_auto_expanded_two_dim_variable_becomes_scatter_plot(self):
        """Variable("p", 2) auto-expands into p_0/p_1 - one quantity, not
        two distinct ones, but still exactly the shape a ScatterPlot needs."""
        P = Variable("p", 2)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 2)
        config = DataConfiguration(sample_axis, FeatureAxes(P))
        plot = auto_plot(data, config)
        assert isinstance(plot, ScatterPlot)
        assert plot.x.variable_or_axes is P.leaves[0]
        assert plot.y.variable_or_axes is P.leaves[1]

    def test_no_feature_axes_raises_a_clear_error(self):
        config = DataConfiguration(BatchAxes(10), BatchAxes(5))
        with pytest.raises(ValueError, match="no FeatureAxes"):
            auto_plot(np.zeros((10, 5)), config)


class TestExplicitPlotType:
    def test_plot_type_is_a_plain_pass_through(self):
        """BarPlot and LinePlot share the same data shape, so this is the
        canonical case auto-selection deliberately leaves to an explicit
        choice - passing plot_type=BarPlot must still work end to end."""
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = auto_plot(data, config, plot_type=BarPlot, x=sample_axis, y=Y)
        assert isinstance(plot, BarPlot)

    def test_multiple_geometry_axes_raise_before_reaching_plot_type_none(self, small_mesh_geometry, cylinder_mesh_geometry):
        config = DataConfiguration(
            GeometryAxes(small_mesh_geometry), GeometryAxes(cylinder_mesh_geometry)
        )
        with pytest.raises(ValueError, match="multiple GeometryAxes"):
            auto_plot(np.zeros((1,)), config)


class TestPointCloudDispatch:
    """A DiscreteGeometry with neither a mesh nor a grid shape - e.g. a
    PointCloud - still has discretization_points, so auto_plot falls back
    to PointCloudPlot/QuiverPlot instead of raising, the same way MeshGeometry/
    GridGeometry are dispatched to."""

    def test_scalar_quantity_becomes_a_point_cloud_plot(self):
        U = Variable("u", 1)
        points = np.random.rand(5, 2)
        geometry = PointCloud(Variable("x", 2), points)
        data = np.random.rand(5, 1)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        plot = auto_plot(data, config)
        assert isinstance(plot, PointCloudPlot)

    def test_three_component_vector_quantity_becomes_a_quiver_plot(self):
        V3 = Variable("v", 3)
        points = np.random.rand(5, 3)
        geometry = PointCloud(Variable("x", 3), points)
        data = np.random.rand(5, 3)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(V3))
        plot = auto_plot(data, config)
        assert isinstance(plot, QuiverPlot)

    def test_wrong_dim_quantity_raises_a_clear_error(self):
        V2 = Variable("v", 2)
        points = np.random.rand(5, 3)
        geometry = PointCloud(Variable("x", 3), points)
        data = np.random.rand(5, 2)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(V2))
        with pytest.raises(ValueError, match="PointCloudPlot"):
            auto_plot(data, config)


class TestContinuousGeometryRaises:
    def test_a_continuous_geometry_with_no_discretization_is_the_only_error_case(self):
        """The one case auto_plot genuinely can't handle: a continuous
        Geometry that was never discretized has no known point positions at
        all - unlike any DiscreteGeometry, which always has
        discretization_points even without mesh/grid structure."""
        U = Variable("u", 1)
        circle = Circle(variable=Variable("x", 2), center=[0, 0], radius=1.0)
        data = np.random.rand(5, 1)
        config = DataConfiguration(GeometryAxes(circle), FeatureAxes(U))
        with pytest.raises(ValueError, match="no known discretization"):
            auto_plot(data, config)
