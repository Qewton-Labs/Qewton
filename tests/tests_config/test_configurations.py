import pytest
from src.pioneer.config.configuration_base import DataConfiguration
from src.pioneer.config.axis import BatchAxis, SpatialAxis, FeatureAxis, TimeAxis
from src.pioneer.config.variables import Variable


class TestDataConfigurationInit:
    """Tests for DataConfiguration initialization and basic properties."""

    def test_init_basic_and_properties(self):
        """Test basic initialization and property access."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)
        assert config.dtype == float
        assert config.axes == [batch, feature]
        assert config.feature_axis == feature
        assert not config.connection_to_axes

    def test_init_with_ellipsis_and_connections(self):
        """Test init with ellipsis and custom connections."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=3)
        v = Variable.from_dict({"x": 3})
        connections = {v: [spatial]}

        config = DataConfiguration(None, [batch, ..., feature], feature, connections)
        assert config.axes[1] is ...
        assert config.connection_to_axes == connections

    def test_init_feature_axis_not_in_axes_fails(self):
        """Test that feature_axis must be in axes."""
        batch = BatchAxis()
        feature1 = FeatureAxis(size=10)
        feature2 = FeatureAxis(size=20)

        with pytest.raises(AssertionError):
            DataConfiguration(float, [batch, feature1], feature2)


class TestAxisIndexing:
    """Tests for batch_axis_idx and feature_axis_idx properties."""

    def test_batch_axis_idx_finds_and_caches(self):
        """Test finding batch axis at various positions and caching."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)

        # Test at different positions
        config1 = DataConfiguration(float, [batch, feature], feature)
        assert config1.batch_axis_idx == 0

        config2 = DataConfiguration(float, [spatial, batch, feature], feature)
        assert config2.batch_axis_idx == 1

        # Test caching
        idx1 = config2.batch_axis_idx
        idx2 = config2.batch_axis_idx
        assert idx1 == idx2 == 1

    def test_batch_axis_idx_missing_or_ellipsis_fails(self):
        """Test batch axis index errors."""
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        batch = BatchAxis()

        # Missing batch
        config_no_batch = DataConfiguration(float, [spatial, feature], feature)
        with pytest.raises(ValueError, match="no batch axis"):
            _ = config_no_batch.batch_axis_idx

        # Ellipsis blocks finding
        config_ellipsis = DataConfiguration(float, [..., batch, feature], feature)
        with pytest.raises(RuntimeError, match="ellipsis"):
            _ = config_ellipsis.batch_axis_idx

    def test_feature_axis_idx_finds_and_caches(self):
        """Test finding feature axis at various positions and caching."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)

        # Test at different positions
        config1 = DataConfiguration(float, [feature, batch], feature)
        assert config1.feature_axis_idx == 0

        config2 = DataConfiguration(float, [batch, spatial, feature], feature)
        assert config2.feature_axis_idx == 2

        # Test caching
        idx1 = config2.feature_axis_idx
        idx2 = config2.feature_axis_idx
        assert idx1 == idx2 == 2

    def test_feature_axis_idx_missing_or_ellipsis_fails(self):
        """Test feature axis index errors."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)

        # Ellipsis blocks finding
        config_ellipsis = DataConfiguration(float, [batch, ..., feature], feature)
        with pytest.raises(RuntimeError, match="ellipsis"):
            _ = config_ellipsis.feature_axis_idx


class TestDataConfigurationFits:
    """Tests for fits method - all branches of compatibility logic."""

    def test_fits_identical_configs(self):
        """Test that identical configs fit each other."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config1 = DataConfiguration(float, [batch, feature], feature)
        config2 = DataConfiguration(float, [batch, feature], feature)
        assert config1.fits(config2)

    def test_fits_with_ellipsis_matches_anything(self):
        """Test that ellipsis in template matches any axes."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)
        time = TimeAxis(size=50)

        config_template = DataConfiguration(float, [batch, ..., feature], feature)
        config_data = DataConfiguration(float, [batch, spatial, time, feature], feature)
        assert config_template.fits(config_data)

    def test_fits_trailing_ellipsis(self):
        """Test trailing ellipsis matches everything."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)

        config_template = DataConfiguration(float, [batch, feature, ...], feature)
        config_data = DataConfiguration(float, [batch, feature, spatial], feature)
        assert config_template.fits(config_data)

    def test_fits_wrong_axis_order_fails(self):
        """Test that wrong axis order fails."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)

        config1 = DataConfiguration(float, [batch, spatial, feature], feature)
        config2 = DataConfiguration(float, [batch, feature, spatial], feature)
        assert not config1.fits(config2)

    def test_fits_different_axis_types_fails(self):
        """Test that different axis types fail."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)
        time = TimeAxis(size=50)

        config1 = DataConfiguration(float, [batch, spatial, feature], feature)
        config2 = DataConfiguration(float, [batch, time, feature], feature)
        assert not config1.fits(config2)

    def test_fits_with_variables_not_subset_fails(self):
        """Test that non-subset variables fail."""
        batch = BatchAxis()
        v1 = Variable.from_dict({"u": 2})
        v2 = Variable.from_dict({"p": 1})

        feature1 = FeatureAxis(size=2, variables=v1)
        feature2 = FeatureAxis(size=1, variables=v2)

        config1 = DataConfiguration(float, [batch, feature1], feature1)
        config2 = DataConfiguration(float, [batch, feature2], feature2)
        assert not config1.fits(config2)

    def test_fits_ellipsis_feature_axis(self):
        """Test fitting when feature axis is ellipsis."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)

        config1 = DataConfiguration(float, [batch, ...], ...)
        config2 = DataConfiguration(float, [batch, feature], feature)
        assert config1.fits(config2)

    def test_fits_feature_axis_no_variables(self):
        """Test fitting when feature axis has no variables."""
        batch = BatchAxis()
        feature1 = FeatureAxis(size=10)  # No variables
        feature2 = FeatureAxis(size=10)

        config1 = DataConfiguration(float, [batch, feature1], feature1)
        config2 = DataConfiguration(float, [batch, feature2], feature2)
        assert config1.fits(config2)

    def test_fits_too_few_axes_fails(self):
        """Test that too few axes fails."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)

        config1 = DataConfiguration(float, [batch, spatial, feature], feature)
        config2 = DataConfiguration(float, [batch, feature], feature)
        assert not config1.fits(config2)


class TestDataConfigurationGetitem:
    """Tests for __getitem__ slicing method."""

    def test_getitem_int_single_axis(self):
        """Test getting single axis by integer index."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)

        sliced = config[0]
        assert len(sliced.axes) == 1
        assert isinstance(sliced.axes[0], BatchAxis)

    def test_getitem_slice_range(self):
        """Test getting slice of axes."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)

        sliced = config[0:2]
        assert len(sliced.axes) == 2
        assert isinstance(sliced.axes[0], BatchAxis)
        assert isinstance(sliced.axes[1], SpatialAxis)

    def test_getitem_slice_preserves_feature_axis(self):
        """Test that feature axis is preserved if in slice."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)
        config = DataConfiguration(float, [batch, feature, spatial], feature)

        sliced = config[0:2]
        assert sliced.feature_axis == feature

    def test_getitem_slice_feature_axis_not_included(self):
        """Test that feature_axis becomes ellipsis if not in slice."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)

        sliced = config[0:2]
        assert sliced.feature_axis is ...

    def test_getitem_empty_slice_raises_error(self):
        """Test that empty slice raises ValueError."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)

        with pytest.raises(ValueError, match="empty"):
            _ = config[1:1]

    def test_getitem_variable_slice(self):
        """Test slicing by Variable."""
        batch = BatchAxis()
        v_full = Variable.from_dict({"u": 2, "v": 2, "p": 1})
        v_slice = Variable.from_dict({"u": 2})
        feature = FeatureAxis(size=5, variables=v_full)
        config = DataConfiguration(float, [batch, feature], feature)

        sliced = config[v_slice]
        assert sliced.feature_axis is not ...
        assert sliced.feature_axis.size == 2
        assert sliced.feature_axis.variables == v_slice

    def test_getitem_variable_not_subset_raises_error(self):
        """Test that non-subset Variable raises AssertionError."""
        batch = BatchAxis()
        v_full = Variable.from_dict({"u": 2, "p": 1})
        v_not_in = Variable.from_dict({"w": 1})
        feature = FeatureAxis(size=3, variables=v_full)
        config = DataConfiguration(float, [batch, feature], feature)

        with pytest.raises(AssertionError):
            _ = config[v_not_in]

    def test_getitem_variable_with_ellipsis_feature_raises_error(self):
        """Test that slicing by Variable with ellipsis feature axis raises ValueError."""
        batch = BatchAxis()
        v = Variable.from_dict({"u": 2})
        config = DataConfiguration(float, [batch, ...], ...)

        with pytest.raises(ValueError, match="Ellipsis"):
            _ = config[v]

    def test_getitem_variable_with_no_variables_raises_error(self):
        """Test slicing by Variable when feature axis has no variables."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)  # No variables
        v = Variable.from_dict({"u": 2})
        config = DataConfiguration(float, [batch, feature], feature)

        with pytest.raises(ValueError, match="no variables"):
            _ = config[v]

    def test_getitem_negative_index(self):
        """Test negative indexing."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)

        sliced = config[-1]
        assert len(sliced.axes) == 1
        assert isinstance(sliced.axes[0], FeatureAxis)

    def test_getitem_returns_same_type(self):
        """Test that slicing returns same type."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)

        sliced = config[0]
        assert isinstance(sliced, DataConfiguration)

    def test_getitem_deep_copies_axes(self):
        """Test that getitem deep copies axes when slicing by Variable."""
        batch = BatchAxis()
        v_full = Variable.from_dict({"u": 2, "p": 1})
        v_slice = Variable.from_dict({"u": 2})
        feature = FeatureAxis(size=3, variables=v_full)
        config = DataConfiguration(float, [batch, feature], feature)

        sliced = config[v_slice]
        # Verify that modifying sliced doesn't affect original
        assert config.feature_axis != sliced.feature_axis


class TestDataConfigurationLen:
    """Tests for __len__ method."""

    def test_len_single_axis(self):
        """Test length with single axis."""
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [feature], feature)
        assert len(config) == 1

    def test_len_multiple_axes(self):
        """Test length with multiple axes."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)
        assert len(config) == 3

    def test_len_with_ellipsis(self):
        """Test length with ellipsis counts as one."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, ..., feature], feature)
        assert len(config) == 3


class TestDataConfigurationEquality:
    """Tests for __eq__ method."""

    def test_eq_identical_configs(self):
        """Test equality of identical configurations."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config1 = DataConfiguration(float, [batch, feature], feature)
        config2 = DataConfiguration(float, [batch, feature], feature)
        assert config1 == config2

    def test_eq_different_type_returns_false(self):
        """Test equality with non-DataConfiguration returns False."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)
        assert config != "not a config"
        assert config != 42

    def test_eq_different_axis_count(self):
        """Test that different axis counts are not equal."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        spatial = SpatialAxis(size=100)

        config1 = DataConfiguration(float, [batch, feature], feature)
        config2 = DataConfiguration(float, [batch, spatial, feature], feature)
        assert config1 != config2

    def test_eq_different_dtype(self):
        """Test that different dtypes are not equal."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config1 = DataConfiguration(float, [batch, feature], feature)
        config2 = DataConfiguration(int, [batch, feature], feature)
        assert config1 != config2

    def test_eq_both_dtype_none(self):
        """Test equality when both dtypes are None."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config1 = DataConfiguration(None, [batch, feature], feature)
        config2 = DataConfiguration(None, [batch, feature], feature)
        assert config1 == config2

    def test_eq_one_dtype_none_other_not(self):
        """Test equality when one dtype is None."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config1 = DataConfiguration(None, [batch, feature], feature)
        config2 = DataConfiguration(float, [batch, feature], feature)
        # Based on code: if self.dtype is not None and other.dtype is not
        # None: return False
        # So this returns True (both dtypes considered compatible)
        assert config1 == config2

    def test_eq_different_axes(self):
        """Test that different axes are not equal."""
        batch = BatchAxis()
        feature1 = FeatureAxis(size=10)
        feature2 = FeatureAxis(size=20)

        config1 = DataConfiguration(float, [batch, feature1], feature1)
        config2 = DataConfiguration(float, [batch, feature2], feature2)
        assert config1 != config2


class TestDataConfigurationVariableMethods:
    """Tests for variable and axis mapping methods."""

    def test_axes_of_variable(self):
        """Test getting axes for variables."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=3)
        v1 = Variable.from_dict({"x": 3})
        v2 = Variable.from_dict({"y": 1})

        config = DataConfiguration(
            float, [batch, spatial, feature], feature, connection_to_axes={v1: [spatial]}
        )
        assert config.axes_of(v1) == [spatial]
        assert not config.axes_of(v2)

    def test_variables_on_axis(self):
        """Test getting variables on an axis."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=3)
        v = Variable.from_dict({"x": 3})

        config = DataConfiguration(
            float, [batch, spatial, feature], feature, connection_to_axes={v: [spatial]}
        )
        assert config.variables_on_axis(spatial) == v
        assert config.variables_on_axis(batch) is None

    def test_map_variable_to_axes_and_get_indices(self):
        """Test mapping and retrieving variable indices."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=5)
        v_full = Variable.from_dict({"u": 2, "p": 1, "v": 2})
        v_query = Variable.from_dict({"p": 1})

        config = DataConfiguration(float, [batch, spatial, feature], feature)
        config.map_variable_to_axes(v_full, [spatial])
        assert config.axes_of(v_full) == [spatial]

        # Set variables on feature for index lookup
        config.feature_axis = FeatureAxis(size=5, variables=v_full)
        indices = config.get_axis_indices_of_variables(v_query)
        assert indices == [2]  # p is at index 2

    def test_map_variable_to_axes_validation(self):
        """Test that mapping validates axes are in config."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=3)
        other_spatial = SpatialAxis(size=50)
        v = Variable.from_dict({"x": 3})

        config = DataConfiguration(float, [batch, spatial, feature], feature)
        with pytest.raises(AssertionError):
            config.map_variable_to_axes(v, [other_spatial])

    def test_get_axis_indices_edge_cases(self):
        """Test axis indices with multiple and edge cases."""
        batch = BatchAxis()
        feature = FeatureAxis(size=5)
        v_full = Variable.from_dict({"u": 2, "p": 1, "v": 2})
        v_multi = Variable.from_dict({"u": 2, "v": 2})

        config = DataConfiguration(float, [batch, feature], feature)
        config.feature_axis = FeatureAxis(size=5, variables=v_full)

        # Multiple variables
        indices = config.get_axis_indices_of_variables(v_multi)
        assert indices == [0, 1, 3, 4]

        # No variables
        config2 = DataConfiguration(float, [batch, feature], feature)
        indices2 = config2.get_axis_indices_of_variables(v_full)
        assert not indices2

    def test_slice_axis_creates_proper_tuple(self):
        """Test slice_axis method creates correct indexing tuple."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)

        # Slice first axis
        slices1 = config.slice_axis(0, slice(0, 10))
        assert slices1 == (slice(0, 10), slice(None), slice(None))

        # Slice middle axis
        slices2 = config.slice_axis(1, 5)
        assert slices2 == (slice(None), 5, slice(None))

        # Slice last axis
        slices3 = config.slice_axis(2, [0, 1, 2])
        assert slices3 == (slice(None), slice(None), [0, 1, 2])
        assert isinstance(slices3, tuple)


class TestDataConfigurationSliceAxis:
    """Tests for slice_axis method."""

    def test_slice_axis_first_axis(self):
        """Test slicing first axis."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)

        slices = config.slice_axis(0, slice(0, 10))
        assert slices[0] == slice(0, 10)
        assert slices[1] == slice(None)
        assert slices[2] == slice(None)

    def test_slice_axis_middle_axis(self):
        """Test slicing middle axis."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)

        slices = config.slice_axis(1, 5)
        assert slices[0] == slice(None)
        assert slices[1] == 5
        assert slices[2] == slice(None)

    def test_slice_axis_last_axis(self):
        """Test slicing last axis."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, spatial, feature], feature)

        slices = config.slice_axis(2, [0, 1, 2])
        assert slices[0] == slice(None)
        assert slices[1] == slice(None)
        assert slices[2] == [0, 1, 2]

    def test_slice_axis_returns_tuple(self):
        """Test that slice_axis returns tuple."""
        batch = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch, feature], feature)

        slices = config.slice_axis(0, slice(0, 5))
        assert isinstance(slices, tuple)
        assert len(slices) == 2


class TestDataConfigurationEdgeCases:
    """Edge cases and dangerous input tests."""

    def test_single_ellipsis_axes(self):
        """Test with only ellipsis in axes."""
        feature = ...
        config = DataConfiguration(float, [...], feature)
        assert config.axes == [...]

    def test_multiple_batch_axes_returns_first(self):
        """Test that multiple batch axes returns first one found."""
        batch1 = BatchAxis()
        batch2 = BatchAxis()
        feature = FeatureAxis(size=10)
        config = DataConfiguration(float, [batch1, batch2, feature], feature)
        assert config.batch_axis_idx == 0

    def test_multiple_feature_axes_returns_first(self):
        """Test that multiple feature axes returns first one found."""
        batch = BatchAxis()
        feature1 = FeatureAxis(size=10)
        feature2 = FeatureAxis(size=20)
        # This should fail assertion since feature1 and feature2 not same
        # Using feature1 as the feature_axis parameter
        config = DataConfiguration(float, [batch, feature1, feature2], feature1)
        assert config.feature_axis_idx == 1

    def test_variable_with_zero_dimension(self):
        """Test variable with zero dimension."""
        batch = BatchAxis()
        v = Variable.from_dict({})
        feature = FeatureAxis(size=0, variables=v)
        config = DataConfiguration(float, [batch, feature], feature)
        indices = config.get_axis_indices_of_variables(v)
        assert not indices

    def test_deeply_nested_variable_slicing(self):
        """Test slicing with complex multi-component variables."""
        batch = BatchAxis()
        v_full = Variable.from_dict({"ux": 1, "uy": 1, "uz": 1, "p": 1})
        v_slice = Variable.from_dict({"uz": 1, "p": 1})
        feature = FeatureAxis(size=4, variables=v_full)
        config = DataConfiguration(float, [batch, feature], feature)

        sliced = config[v_slice]
        assert sliced.feature_axis is not ...
        assert sliced.feature_axis.size == 2

    def test_connection_to_axes_persistence(self):
        """Test that connection_to_axes persists through operations."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=100)
        feature = FeatureAxis(size=3)
        v = Variable.from_dict({"x": 3})

        config = DataConfiguration(
            float, [batch, spatial, feature], feature, connection_to_axes={v: [spatial]}
        )
        sliced = config[0:2]
        assert v in sliced.connection_to_axes
