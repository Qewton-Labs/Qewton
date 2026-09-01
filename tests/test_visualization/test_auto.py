import numpy as np
import pytest

from qewton.algorithms.building_blocks.geometry import MeshInterpolationNode
from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.geometries.continuous.domains_1d.interval import Interval
from qewton.geometries.continuous.domains_2d.circle import Circle
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry
from qewton.geometries.discrete.point_cloud import PointCloud
from qewton.geometries.discrete.volume_grid import VolumeGridGeometry
from qewton.visualization.auto import auto_plot
from qewton.visualization.plots.data.curve import LinePlot
from qewton.visualization.plots.data.grid import EmbeddedGridPlot, HeatmapPlot, QuiverPlot
from qewton.visualization.plots.data.mesh import MeshFieldPlot, MeshVectorPlot
from qewton.visualization.plots.data.points import PointCloudPlot
from qewton.visualization.plots.data.samples import BarPlot, ScatterPlot
from qewton.visualization.plots.spec import FixedSpec, SliderSpec, VariableSpec


class TestLeftoverAxesBecomeSliders:
    def test_leftover_batch_axis_next_to_a_mesh_surface_gets_a_default_slider(
        self, small_mesh_geometry
    ):
        U = Variable("u", 1)
        step_axis = BatchAxes(5)
        n_vertices = small_mesh_geometry.mesh.vertices.shape[0]
        config = DataConfiguration(
            step_axis, GeometryAxes(small_mesh_geometry), FeatureAxes(U)
        )
        data = np.zeros((5, n_vertices, 1))
        plot = auto_plot(data, config)
        assert isinstance(plot, MeshFieldPlot)
        assert len(plot.controls) == 1
        assert isinstance(plot.controls[0], SliderSpec)
        assert plot.controls[0].variable_or_axes is step_axis
        assert plot.controls[0].minimum == 0 and plot.controls[0].maximum == 4

    def test_an_explicit_control_for_the_axis_is_not_duplicated(
        self, small_mesh_geometry
    ):
        U = Variable("u", 1)
        step_axis = BatchAxes(5)
        n_vertices = small_mesh_geometry.mesh.vertices.shape[0]
        config = DataConfiguration(
            step_axis, GeometryAxes(small_mesh_geometry), FeatureAxes(U)
        )
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
        """A scalar field on a mesh becomes MeshFieldPlot regardless of
        mesh.dim - it draws flat (FilledMeshArtist) in 2D and as a boundary
        surface (SurfaceMeshArtist) in 3D itself, so auto_plot needs no
        dim-specific branch. MeshSurfacePlot (elevating a 2D mesh into a
        height field) is an equally valid alternative, left explicit -
        same rule as BarPlot vs. LinePlot."""
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

    def test_two_distinct_scalar_variables_become_a_mesh_field_plot_with_a_selector(
        self, small_mesh_geometry
    ):
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

    def test_distinct_variables_with_different_dims_raise_a_clear_error(
        self, small_mesh_geometry
    ):
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
    node = MeshInterpolationNode(
        cylinder_mesh_geometry, U, grid, backend=cylinder_mesh_geometry.backend
    )
    vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
    field = cylinder_mesh_geometry.backend.build_tensor((vertices**2).sum(axis=1))
    resampled = node.forward(field)[..., None]
    config = DataConfiguration(GeometryAxes(grid), FeatureAxes(U))
    return resampled, config, grid, U, i


class TestGridDispatch:
    def test_scalar_variable_becomes_embedded_grid_plot(self, resampled_grid):
        resampled, config, grid, U, i = resampled_grid
        plot = auto_plot(
            resampled,
            config,
            controls=[FixedSpec(init_state=2, n_dimensions=1, variable_or_axes=i)],
        )
        assert isinstance(plot, EmbeddedGridPlot)
        assert plot.color.variable_or_axes is U

    def test_more_than_two_grid_axes_get_an_automatic_slider_on_the_surplus(
        self, resampled_grid
    ):
        """Without an explicit control for i, auto_plot must generate its
        own SliderSpec for it (the same _default_sliders mechanism used for
        non-geometry axes) rather than raising or guessing which axis to
        reduce - the last two leaves (j, k) stay as the drawn grid."""
        resampled, config, grid, U, i = resampled_grid
        plot = auto_plot(resampled, config)
        assert isinstance(plot, EmbeddedGridPlot)
        assert len(plot.controls) == 1
        assert isinstance(plot.controls[0], SliderSpec)
        assert plot.controls[0].variable_or_axes is i

    def test_three_component_variable_becomes_quiver_plot(self, cylinder_mesh_geometry):
        V3 = Variable("v", 3)
        i, j, k = Variable("i", 1), Variable("j", 1), Variable("k", 1)
        grid = VolumeGridGeometry(cylinder_mesh_geometry, i * j * k, resolution=(4, 4, 4))
        node = MeshInterpolationNode(
            cylinder_mesh_geometry, V3, grid, backend=cylinder_mesh_geometry.backend
        )
        vertices = np.asarray(cylinder_mesh_geometry.mesh.vertices)
        vec = np.stack([-vertices[:, 1], vertices[:, 0], np.zeros(len(vertices))], axis=1)
        resampled = node.forward(cylinder_mesh_geometry.backend.build_tensor(vec))
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(V3))
        plot = auto_plot(resampled, config)
        assert isinstance(plot, QuiverPlot)
        assert plot.vector.variable_or_axes is V3


class TestControlsClassOrInstance:
    """controls= accepts a ControlSpec class/instance/dict as the resolver
    for surplus axes. resampled_grid has
    exactly one surplus axis (i) beyond the drawn (j, k) grid."""

    def test_a_bare_class_is_instantiated_per_axis(self, resampled_grid):
        resampled, config, grid, U, i = resampled_grid
        plot = auto_plot(resampled, config, controls=FixedSpec)
        assert len(plot.controls) == 1
        assert isinstance(plot.controls[0], FixedSpec)
        assert plot.controls[0].variable_or_axes is i

    def test_a_bare_instance_is_resolved_for_the_one_surplus_axis(self, resampled_grid):
        resampled, config, grid, U, i = resampled_grid
        control = FixedSpec(init_state=2)
        plot = auto_plot(resampled, config, controls=control)
        assert plot.controls == [control]
        assert control.variable_or_axes is i
        assert control.state == 2

    def test_a_dict_resolves_the_named_axis(self, resampled_grid):
        resampled, config, grid, U, i = resampled_grid
        control = FixedSpec(init_state=1)
        plot = auto_plot(resampled, config, controls={i: control})
        assert plot.controls == [control]
        assert control.variable_or_axes is i

    def test_a_dict_falls_back_to_sliderspec_for_unnamed_axes(self, resampled_grid):
        resampled, config, grid, U, i = resampled_grid
        plot = auto_plot(resampled, config, controls={})
        assert len(plot.controls) == 1
        assert isinstance(plot.controls[0], SliderSpec)

    def test_a_shared_instance_across_two_auto_plot_calls_stays_one_instance(
        self, resampled_grid
    ):
        """One instance passed to several auto_plot() calls ends up shared
        (one widget, moving every panel together) - the same identity-based
        mechanism a shared Scale/VariableSpec already relies on."""
        resampled, config, grid, U, i = resampled_grid
        control = FixedSpec(init_state=2)
        plot_a = auto_plot(resampled, config, controls=control)
        plot_b = auto_plot(resampled, config, controls=control)
        assert plot_a.controls[0] is plot_b.controls[0] is control

    def test_an_instance_already_resolved_for_a_different_axis_raises(
        self, resampled_grid
    ):
        resampled, config, grid, U, i = resampled_grid
        j = grid.variable.leaves[1]
        control = FixedSpec(init_state=0, variable_or_axes=j)
        with pytest.raises(ValueError, match="already resolved"):
            auto_plot(resampled, config, controls=control)

    def test_a_bare_instance_for_more_than_one_surplus_axis_raises(self):
        i, j, k, l = (Variable(name, 1) for name in "ijkl")
        grid = IndexGridGeometry(i * j * k * l, shape=(2, 2, 2, 2))
        U = Variable("u", 1)
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(U))
        data = np.zeros((2, 2, 2, 2, 1))
        with pytest.raises(ValueError, match="can't resolve"):
            auto_plot(data, config, controls=FixedSpec(init_state=0))

    def test_a_dict_entry_for_an_already_covered_axis_raises(self, resampled_grid):
        """Exercises _default_sliders directly - the conflict (a dict entry
        for an axis some other, already-resolved control covers) can't
        arise through a single auto_plot(controls=...) call, since that
        kwarg is either a literal list of pre-resolved controls or a
        resolver, never both at once; the check still matters for
        _default_sliders as a building block other call sites reuse."""
        from qewton.visualization.auto import _default_sliders

        resampled, config, grid, U, i = resampled_grid
        own_control = FixedSpec(init_state=2, variable_or_axes=i)
        with pytest.raises(ValueError, match="already covered"):
            _default_sliders([i], [own_control], controls={i: SliderSpec})


class TestIndexGridDispatch:
    """A GridGeometry whose discretization_points are 2-component (index
    coordinates, not a spatial embedding) - coord_dim, not grid_dims,
    decides HeatmapPlot vs. EmbeddedGridPlot."""

    @staticmethod
    def _index_grid_2d():
        i, j = Variable("i", 1), Variable("j", 1)
        return IndexGridGeometry(i * j, shape=(4, 5))

    def test_scalar_variable_becomes_heatmap_plot(self):
        grid = self._index_grid_2d()
        U = Variable("u", 1)
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(U))
        data = np.zeros((4, 5, 1))
        plot = auto_plot(data, config)
        assert isinstance(plot, HeatmapPlot)
        assert plot.color.variable_or_axes is U

    def test_two_component_variable_becomes_a_2d_quiver_plot(self):
        grid = self._index_grid_2d()
        V2 = Variable("v", 2)
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(V2))
        data = np.zeros((4, 5, 2))
        plot = auto_plot(data, config)
        assert isinstance(plot, QuiverPlot)
        assert plot.embedding_dim == 2
        assert plot.vector.variable_or_axes is V2

    def test_one_grid_axis_becomes_a_line_plot(self):
        i = Variable("i", 1)
        grid = IndexGridGeometry(i, shape=(10,))
        U = Variable("u", 1)
        config = DataConfiguration(GeometryAxes(grid), FeatureAxes(U))
        data = np.zeros((10, 1))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        assert plot.x.variable_or_axes is i
        assert plot.y.variable_or_axes is U


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

    def test_three_distinct_scalar_variables_with_a_domain_axis_becomes_a_line_plot_with_a_selector(
        self,
    ):
        X, Y, Z = Variable("x", 1), Variable("y", 1), Variable("z", 1)
        sample_axis = BatchAxes(10)
        data = np.random.randn(10, 3)
        config = DataConfiguration(sample_axis, FeatureAxes(X * Y * Z))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        assert plot.x.variable_or_axes is sample_axis
        assert isinstance(plot.y.variable_or_axes, Variable)
        assert plot.y.variable_or_axes in (X, Y, Z)  # whichever the selector defaults to

    def test_three_distinct_scalar_variables_without_a_domain_axis_raise_a_clear_error(
        self,
    ):
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

    def test_multiple_geometry_axes_raise_before_reaching_plot_type_none(
        self, small_mesh_geometry, cylinder_mesh_geometry
    ):
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


class TestOneDGeometryDispatch:
    """A 1D geometry (mesh or point cloud) isn't a drawable 2D/3D surface
    or point cloud - it's a LinePlot, value vs. position along the domain,
    with real coordinates from the geometry's own discretization_points
    (not the plain sample index) - see LinePlot._geometry_x_values()."""

    def test_1d_mesh_scalar_becomes_a_line_plot_with_real_x_coordinates(self):
        T = Variable("t", 1)
        U = Variable("u", 1)
        interval = Interval(T, 0.0, 2.0)
        geometry = interval.create_mesh(max_vertex_distance=0.5)
        n = len(np.asarray(geometry.mesh.vertices))
        data = (np.asarray(geometry.mesh.vertices) ** 2).reshape(n, 1)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        result = plot.evaluate()
        assert np.allclose(np.sort(result.x), np.linspace(0.0, 2.0, n))

    def test_1d_mesh_vector_raises_a_clear_error(self):
        T = Variable("t", 1)
        V3 = Variable("v3", 3)
        interval = Interval(T, 0.0, 2.0)
        geometry = interval.create_mesh(max_vertex_distance=0.5)
        n = len(np.asarray(geometry.mesh.vertices))
        data = np.zeros((n, 3))
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(V3))
        with pytest.raises(ValueError, match="1D mesh"):
            auto_plot(data, config)

    def test_1d_point_cloud_scalar_becomes_a_line_plot_with_real_x_coordinates(self):
        U = Variable("u", 1)
        points = np.linspace(0.0, 1.0, 10).reshape(-1, 1).astype(np.float32)
        geometry = PointCloud(Variable("t", 1), points)
        data = (points**2).astype(np.float32)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        plot = auto_plot(data, config)
        assert isinstance(plot, LinePlot)
        result = plot.evaluate()
        assert np.allclose(result.x, points[:, 0])

    def test_1d_point_cloud_vector_raises_a_clear_error(self):
        V3 = Variable("v", 3)
        points = np.linspace(0.0, 1.0, 5).reshape(-1, 1).astype(np.float32)
        geometry = PointCloud(Variable("t", 1), points)
        data = np.zeros((5, 3), dtype=np.float32)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(V3))
        with pytest.raises(ValueError, match="1D point set"):
            auto_plot(data, config)


class TestLinePlotGeometryXValues:
    """LinePlot's x=geometry_axes/x=grid-leaf cases show real coordinates;
    everything else keeps the plain sample index unchanged."""

    def test_plain_axis_x_falls_back_to_sample_index(self):
        Y = Variable("y", 1)
        sample_axis = BatchAxes(10)
        data = np.random.rand(10, 1)
        config = DataConfiguration(sample_axis, FeatureAxes(Y))
        plot = LinePlot(data, config, x=sample_axis, y=Y)
        result = plot.evaluate()
        assert np.array_equal(result.x, np.arange(10))

    def test_grid_leaf_x_uses_the_grid_geometrys_real_coordinates(self):
        """auto_plot()'s grid_dims==1 case passes x=leaves[0] (the grid's
        own structural Variable), not the GeometryAxes itself - confirms
        _geometry_x_values() also recognizes that form."""
        i = Variable("i", 1)
        U = Variable("u", 1)
        geometry = IndexGridGeometry(i, shape=(6,))
        real_points = np.linspace(0.0, 5.0, 6).reshape(-1, 1).astype(np.float32)
        geometry.discretization_points = geometry.backend.build_tensor(real_points)
        data = np.zeros((6, 1), dtype=np.float32)
        config = DataConfiguration(GeometryAxes(geometry), FeatureAxes(U))
        plot = LinePlot(data, config, x=i, y=U)
        result = plot.evaluate()
        assert np.allclose(result.x, real_points[:, 0])


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
