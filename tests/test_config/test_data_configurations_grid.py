import pytest

from qewton.config.axes import BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import GridDataConfiguration
from qewton.config.data_configurations.grid import GridDataConfig
from qewton.config.variables import Variable
from qewton.geometries.discrete.index_grid_geometry import IndexGridGeometry


class TestGridDataConfig:
    def test_builds_batch_geometry_feature_axes_from_a_plain_shape(self):
        U = Variable("u", 1)
        config = GridDataConfig((10, 8, 6, 1), feature_variable=U)
        assert len(config.axes) == 3
        assert isinstance(config.axes[0], BatchAxes)
        assert isinstance(config.axes[1], GeometryAxes)
        assert isinstance(config.axes[2], FeatureAxes)
        assert config.shape == (10, 8, 6, 1)

    def test_geometry_is_an_index_grid_matching_the_middle_shape(self):
        U = Variable("u", 1)
        config = GridDataConfig((10, 8, 6, 1), feature_variable=U)
        geometry = config.geometry_axes.geometry
        assert isinstance(geometry, IndexGridGeometry)
        assert geometry._grid_shape == (8, 6)

    def test_geometry_variable_defaults_to_an_auto_named_grid_variable(self):
        U = Variable("u", 1)
        config = GridDataConfig((10, 8, 6, 1), feature_variable=U)
        geometry = config.geometry_axes.geometry
        assert geometry.variable.name == "grid"
        assert geometry.variable.dim == 2

    def test_explicit_geometry_variable_is_used(self):
        U = Variable("u", 1)
        XY = Variable("x", 1) * Variable("y", 1)
        config = GridDataConfig((10, 8, 6, 1), feature_variable=U, geometry_variable=XY)
        assert config.geometry_axes.geometry.variable is XY

    def test_feature_variable_dim_mismatch_raises(self):
        V2 = Variable("v", 2)  # data's last axis is 1, not 2
        with pytest.raises(AssertionError, match="does not match"):
            GridDataConfig((10, 8, 6, 1), feature_variable=V2)

    def test_dtype_is_forwarded(self):
        U = Variable("u", 1)
        config = GridDataConfig((10, 8, 6, 1), feature_variable=U, dtype="float32")
        assert config.dtype == "float32"

    def test_grid_data_configuration_is_the_same_class(self):
        assert GridDataConfiguration is GridDataConfig
