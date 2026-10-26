import numpy as np
import pytest

from qewton.config.variables import Variable
from qewton.data.datasets.array_data.grid import GridDataSet
from qewton.geometries.discrete.grid_geometry import GridGeometry
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry


class TestGridDataSetIndexGeometry:
    def test_single_data_item_builds_an_index_grid_geometry(self):
        F = Variable("f", 1)
        data = np.random.rand(5, 8, 6, 1)
        ds = GridDataSet(data=[data], feature_variables=[F])
        assert len(ds) == 5
        geometry = ds.data_configs[0].geometry_axes.geometry
        assert isinstance(geometry, IndexGridGeometry)
        assert geometry._grid_shape == (8, 6)

    def test_a_single_data_item_need_not_be_wrapped_in_a_list(self):
        F = Variable("f", 1)
        data = np.random.rand(5, 8, 6, 1)
        ds = GridDataSet(data=data, feature_variables=F)
        assert len(ds) == 5

    def test_geometry_variable_defaults_when_omitted(self):
        """Regression: used to crash with AttributeError('NoneType' has no
        'dim') - geometry_variable=None (the documented default) was passed
        straight through to IndexGridGeometry without a fallback, unlike
        GridDataConfig's own equivalent."""
        F = Variable("f", 1)
        data = np.random.rand(5, 8, 6, 1)
        ds = GridDataSet(data=[data], feature_variables=[F])
        geometry = ds.data_configs[0].geometry_axes.geometry
        assert geometry.variable.name == "grid"
        assert geometry.variable.dim == 2

    def test_explicit_geometry_variable_is_used(self):
        F = Variable("f", 1)
        IJ = Variable("i", 1) * Variable("j", 1)
        data = np.random.rand(5, 8, 6, 1)
        ds = GridDataSet(data=[data], feature_variables=[F], geometry_variable=IJ)
        assert ds.data_configs[0].geometry_axes.geometry.variable is IJ

    def test_multiple_data_items_share_one_geometry(self):
        F, G = Variable("f", 1), Variable("g", 2)
        data_f = np.random.rand(5, 8, 6, 1)
        data_g = np.random.rand(5, 8, 6, 2)
        ds = GridDataSet(data=[data_f, data_g], feature_variables=[F, G])
        assert len(ds.data_configs) == 2
        geom_a = ds.data_configs[0].geometry_axes.geometry
        geom_b = ds.data_configs[1].geometry_axes.geometry
        assert geom_a is geom_b

    def test_mismatched_grid_shape_raises(self):
        F, G = Variable("f", 1), Variable("g", 1)
        data_f = np.random.rand(5, 8, 6, 1)
        data_g = np.random.rand(5, 8, 7, 1)  # different grid shape
        with pytest.raises(AssertionError, match="same grid shape"):
            GridDataSet(data=[data_f, data_g], feature_variables=[F, G])

    def test_number_of_feature_variables_must_match_number_of_data_items(self):
        F = Variable("f", 1)
        data = np.random.rand(5, 8, 6, 1)
        with pytest.raises(AssertionError, match="must match"):
            GridDataSet(data=[data, data], feature_variables=[F])


class TestGridDataSetExplicitPointGrid:
    def test_explicit_point_grid_builds_a_grid_geometry(self):
        F = Variable("f", 1)
        data = np.random.rand(5, 4, 3, 1)
        point_grid = np.random.rand(4, 3, 2)  # (grid..., coord_dim)
        ds = GridDataSet(data=[data], feature_variables=[F], point_grid=point_grid)
        geometry = ds.data_configs[0].geometry_axes.geometry
        assert isinstance(geometry, GridGeometry)
        assert not isinstance(geometry, IndexGridGeometry)

    def test_explicit_geometry_variable_is_used_with_a_point_grid_too(self):
        """Regression: the point_grid branch used to always build a fresh
        Variable("grid", ...), silently discarding a caller-supplied
        geometry_variable."""
        F = Variable("f", 1)
        XY = Variable("x", 1) * Variable("y", 1)
        data = np.random.rand(5, 4, 3, 1)
        point_grid = np.random.rand(4, 3, 2)
        ds = GridDataSet(
            data=[data], feature_variables=[F], point_grid=point_grid, geometry_variable=XY
        )
        assert ds.data_configs[0].geometry_axes.geometry.variable is XY

    def test_mismatched_point_grid_shape_raises(self):
        F = Variable("f", 1)
        data = np.random.rand(5, 4, 3, 1)
        point_grid = np.random.rand(4, 5, 2)  # wrong second grid dim
        with pytest.raises(AssertionError, match="point grid shape"):
            GridDataSet(data=[data], feature_variables=[F], point_grid=point_grid)
