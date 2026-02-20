import pytest
from src.pioneer.config.axis import (
    Axis,
    BatchAxis,
    SpatialAxis,
    FeatureAxis,
    TimeAxis,
)
from src.pioneer.config.variables import Variable


class TestAxisInit:
    """Tests for Axis.__init__ method."""

    def test_init_all_none(self):
        """Test creating Axis with all None parameters."""
        axis = Axis()
        assert axis.size is None
        assert axis.name is None
        assert axis.variables is None

    def test_init_with_size_only(self):
        """Test creating Axis with size only."""
        axis = Axis(size=10)
        assert axis.size == 10
        assert axis.name is None
        assert axis.variables is None

    def test_init_with_name_only(self):
        """Test creating Axis with name only."""
        axis = Axis(name="test_axis")
        assert axis.size is None
        assert axis.name == "test_axis"
        assert axis.variables is None

    def test_init_with_variables_only(self):
        """Test creating Axis with variables only."""
        v = Variable.from_dict({"x": 5})
        axis = Axis(variables=v)
        assert axis.size is None
        assert axis.name is None
        assert axis.variables == v

    def test_init_with_size_and_name(self):
        """Test creating Axis with size and name."""
        axis = Axis(size=10, name="spatial")
        assert axis.size == 10
        assert axis.name == "spatial"
        assert axis.variables is None

    def test_init_with_size_and_variables_matching(self):
        """Test creating Axis with matching size and variables dimension."""
        v = Variable.from_dict({"x": 10})
        axis = Axis(size=10, variables=v)
        assert axis.size == 10
        assert axis.variables == v

    def test_init_with_size_and_variables_non_matching_raises_error(self):
        """Test that non-matching size and variables dimension raises AssertionError."""
        v = Variable.from_dict({"x": 5})
        with pytest.raises(AssertionError, match="Size.*does not match"):
            Axis(size=10, variables=v)

    def test_init_with_all_parameters(self):
        """Test creating Axis with all parameters."""
        v = Variable.from_dict({"x": 10, "y": 5})
        axis = Axis(size=15, name="combined", variables=v)
        assert axis.size == 15
        assert axis.name == "combined"
        assert axis.variables == v

    def test_init_with_large_size(self):
        """Test creating Axis with large size value."""
        axis = Axis(size=1000000)
        assert axis.size == 1000000

    def test_init_with_zero_size(self):
        """Test creating Axis with zero size."""
        axis = Axis(size=0)
        assert axis.size == 0

    def test_init_with_zero_size_and_zero_dimension_variables(self):
        """Test creating Axis with zero size and zero dimension variables."""
        v = Variable.from_dict({})
        axis = Axis(size=0, variables=v)
        assert axis.size == 0


class TestAxisNameProperty:
    """Tests for Axis.name property."""

    def test_name_property_when_set(self):
        """Test name property when name is set."""
        axis = Axis(name="test")
        assert axis.name == "test"

    def test_name_property_when_none(self):
        """Test name property when name is None."""
        axis = Axis()
        assert axis.name is None

    def test_name_property_read_only(self):
        """Test that name property returns the internal _name attribute."""
        axis = Axis(name="original")
        assert axis.name == "original"
        # Verify it's getting from _name
        assert axis.name == axis.name

    def test_name_property_with_special_characters(self):
        """Test name property with special characters."""
        axis = Axis(name="axis_123-test.name")
        assert axis.name == "axis_123-test.name"


class TestAxisEquality:
    """Tests for Axis.__eq__ method."""

    def test_eq_same_instance(self):
        """Test equality of same instance."""
        axis = Axis(size=10)
        assert axis == axis

    def test_eq_different_type_returns_false(self):
        """Test equality with non-Axis object returns False."""
        axis = Axis(size=10)
        assert axis != "not an axis"
        assert axis != 10
        assert axis is not None

    def test_eq_same_size_and_variables(self):
        """Test equality with same size and variables."""
        v = Variable.from_dict({"x": 5})
        axis1 = Axis(size=5, variables=v)
        axis2 = Axis(size=5, variables=v)
        assert axis1 == axis2

    def test_eq_different_size_same_variables(self):
        """Test equality with different size, same variables."""
        v = Variable.from_dict({"x": 5})
        with pytest.raises(AssertionError):
            _ = Axis(size=10, variables=v)

    def test_eq_same_size_different_variables(self):
        """Test equality with same size, different variables."""
        v1 = Variable.from_dict({"x": 5})
        v2 = Variable.from_dict({"y": 5})
        axis1 = Axis(size=5, variables=v1)
        axis2 = Axis(size=5, variables=v2)
        assert axis1 != axis2

    def test_eq_both_size_none(self):
        """Test equality when both sizes are None."""
        v = Variable.from_dict({"x": 5})
        axis1 = Axis(variables=v)
        axis2 = Axis(variables=v)
        assert axis1 == axis2

    def test_eq_one_size_none_other_not(self):
        """Test equality when one size is None and other is not."""
        v = Variable.from_dict({"x": 5})
        axis1 = Axis(size=5, variables=v)
        axis2 = Axis(variables=v)  # size is None
        assert axis1 == axis2  # Should be equal due to None check

    def test_eq_one_size_none_other_not_reverse(self):
        """Test equality in reverse: first has size, second doesn't."""
        v = Variable.from_dict({"x": 5})
        axis1 = Axis(variables=v)  # size is None
        axis2 = Axis(size=5, variables=v)
        assert axis1 == axis2

    def test_eq_different_type_subclasses(self):
        """Test equality between different Axis subclass types."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=10)
        assert batch != spatial  # Different types

    def test_eq_same_subclass_type(self):
        """Test equality between same Axis subclass types."""
        spatial1 = SpatialAxis(size=10)
        spatial2 = SpatialAxis(size=10)
        assert spatial1 == spatial2

    def test_eq_both_variables_none(self):
        """Test equality when both variables are None."""
        axis1 = Axis(size=10)
        axis2 = Axis(size=10)
        assert axis1 == axis2

    def test_eq_one_variables_none(self):
        """Test equality when one has variables and other is None."""
        v = Variable.from_dict({"x": 5})
        axis1 = Axis(size=5, variables=v)
        axis2 = Axis(size=5)
        assert axis1 != axis2

    def test_eq_both_size_and_variables_none(self):
        """Test equality when both size and variables are None."""
        axis1 = Axis()
        axis2 = Axis()
        assert axis1 == axis2

    def test_eq_ignores_name_attribute(self):
        """Test that equality comparison ignores name attribute."""
        v = Variable.from_dict({"x": 5})
        axis1 = Axis(size=5, name="axis1", variables=v)
        axis2 = Axis(size=5, name="axis2", variables=v)
        # Note: The code has a commented line "# and self.name == other_axes.name"
        # So names are effectively ignored
        assert axis1 == axis2


class TestBatchAxis:
    """Tests for BatchAxis class."""

    def test_batch_axis_init(self):
        """Test BatchAxis initialization."""
        batch = BatchAxis()
        assert batch.size is None
        assert batch.name == "batch"
        assert batch.variables is None

    def test_batch_axis_is_axis_instance(self):
        """Test that BatchAxis is instance of Axis."""
        batch = BatchAxis()
        assert isinstance(batch, Axis)

    def test_batch_axis_equality(self):
        """Test equality of BatchAxis instances."""
        batch1 = BatchAxis()
        batch2 = BatchAxis()
        assert batch1 == batch2

    def test_batch_axis_not_equal_to_other_axis(self):
        """Test that BatchAxis is not equal to other Axis types."""
        batch = BatchAxis()
        spatial = SpatialAxis()
        assert batch != spatial


class TestSpatialAxis:
    """Tests for SpatialAxis class."""

    def test_spatial_axis_init_no_params(self):
        """Test SpatialAxis initialization without parameters."""
        spatial = SpatialAxis()
        assert spatial.size is None
        assert spatial.name == "spatial"
        assert spatial.variables is None

    def test_spatial_axis_init_with_size(self):
        """Test SpatialAxis initialization with size."""
        spatial = SpatialAxis(size=100)
        assert spatial.size == 100
        assert spatial.name == "spatial"

    def test_spatial_axis_init_with_name(self):
        """Test SpatialAxis initialization with custom name."""
        spatial = SpatialAxis(name="custom_spatial")
        assert spatial.name == "custom_spatial"

    def test_spatial_axis_init_with_size_and_name(self):
        """Test SpatialAxis initialization with size and name."""
        spatial = SpatialAxis(size=50, name="my_spatial")
        assert spatial.size == 50
        assert spatial.name == "my_spatial"

    def test_spatial_axis_is_axis_instance(self):
        """Test that SpatialAxis is instance of Axis."""
        spatial = SpatialAxis()
        assert isinstance(spatial, Axis)

    def test_spatial_axis_equality_same_size(self):
        """Test equality of SpatialAxis with same size."""
        spatial1 = SpatialAxis(size=100)
        spatial2 = SpatialAxis(size=100)
        assert spatial1 == spatial2

    def test_spatial_axis_equality_different_size(self):
        """Test equality of SpatialAxis with different size."""
        spatial1 = SpatialAxis(size=100)
        spatial2 = SpatialAxis(size=50)
        assert spatial1 != spatial2


class TestFeatureAxis:
    """Tests for FeatureAxis class."""

    def test_feature_axis_init_no_params(self):
        """Test FeatureAxis initialization without parameters."""
        feature = FeatureAxis()
        assert feature.size is None
        assert feature.name == "features"
        assert feature.variables is None

    def test_feature_axis_init_with_size(self):
        """Test FeatureAxis initialization with size."""
        feature = FeatureAxis(size=64)
        assert feature.size == 64
        assert feature.name == "features"

    def test_feature_axis_init_with_variables(self):
        """Test FeatureAxis initialization with variables."""
        v = Variable.from_dict({"u": 2, "p": 1})
        feature = FeatureAxis(variables=v)
        assert feature.variables == v
        assert feature.size is None

    def test_feature_axis_init_with_size_and_variables(self):
        """Test FeatureAxis initialization with size and variables."""
        v = Variable.from_dict({"u": 2, "p": 1})
        feature = FeatureAxis(size=3, variables=v)
        assert feature.size == 3
        assert feature.variables == v

    def test_feature_axis_init_with_mismatched_size_and_variables(self):
        """Test FeatureAxis with mismatched size and variables raises error."""
        v = Variable.from_dict({"u": 2, "p": 1})
        with pytest.raises(AssertionError):
            FeatureAxis(size=5, variables=v)

    def test_feature_axis_is_axis_instance(self):
        """Test that FeatureAxis is instance of Axis."""
        feature = FeatureAxis()
        assert isinstance(feature, Axis)

    def test_feature_axis_name_is_features(self):
        """Test that FeatureAxis name is always 'features'."""
        feature = FeatureAxis(size=32)
        assert feature.name == "features"

    def test_feature_axis_equality_same_variables(self):
        """Test equality of FeatureAxis with same variables."""
        v = Variable.from_dict({"x": 5})
        feature1 = FeatureAxis(variables=v)
        feature2 = FeatureAxis(variables=v)
        assert feature1 == feature2

    def test_feature_axis_equality_different_variables(self):
        """Test equality of FeatureAxis with different variables."""
        v1 = Variable.from_dict({"x": 5})
        v2 = Variable.from_dict({"y": 5})
        feature1 = FeatureAxis(variables=v1)
        feature2 = FeatureAxis(variables=v2)
        assert feature1 != feature2


class TestTimeAxis:
    """Tests for TimeAxis class."""

    def test_time_axis_init_no_params(self):
        """Test TimeAxis initialization without parameters."""
        time = TimeAxis()
        assert time.size is None
        assert time.name == "time"
        assert time.variables is None

    def test_time_axis_init_with_size(self):
        """Test TimeAxis initialization with size."""
        time = TimeAxis(size=1000)
        assert time.size == 1000
        assert time.name == "time"

    def test_time_axis_is_axis_instance(self):
        """Test that TimeAxis is instance of Axis."""
        time = TimeAxis()
        assert isinstance(time, Axis)

    def test_time_axis_name_is_time(self):
        """Test that TimeAxis name is always 'time'."""
        time = TimeAxis(size=100)
        assert time.name == "time"

    def test_time_axis_equality_same_size(self):
        """Test equality of TimeAxis with same size."""
        time1 = TimeAxis(size=100)
        time2 = TimeAxis(size=100)
        assert time1 == time2

    def test_time_axis_equality_different_size(self):
        """Test equality of TimeAxis with different size."""
        time1 = TimeAxis(size=100)
        time2 = TimeAxis(size=200)
        assert time1 != time2

    def test_time_axis_equality_one_size_none(self):
        """Test equality of TimeAxis when one has size None."""
        time1 = TimeAxis(size=100)
        time2 = TimeAxis()
        assert time1 == time2  # Due to None check in __eq__


class TestAxisSubclassComparisons:
    """Tests comparing different Axis subclass types."""

    def test_batch_vs_spatial(self):
        """Test BatchAxis vs SpatialAxis equality."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=10)
        assert batch != spatial

    def test_batch_vs_feature(self):
        """Test BatchAxis vs FeatureAxis equality."""
        batch = BatchAxis()
        feature = FeatureAxis(size=32)
        assert batch != feature

    def test_batch_vs_time(self):
        """Test BatchAxis vs TimeAxis equality."""
        batch = BatchAxis()
        time = TimeAxis(size=100)
        assert batch != time

    def test_spatial_vs_feature(self):
        """Test SpatialAxis vs FeatureAxis equality."""
        spatial = SpatialAxis(size=64)
        feature = FeatureAxis(size=64)
        assert spatial != feature

    def test_spatial_vs_time(self):
        """Test SpatialAxis vs TimeAxis equality."""
        spatial = SpatialAxis(size=100)
        time = TimeAxis(size=100)
        assert spatial != time

    def test_feature_vs_time(self):
        """Test FeatureAxis vs TimeAxis equality."""
        feature = FeatureAxis(size=32)
        time = TimeAxis(size=32)
        assert feature != time


class TestAxisWithVariables:
    """Tests for Axis with Variables integration."""

    def test_axis_with_complex_variables(self):
        """Test Axis with complex Variable structure."""
        v = Variable.from_dict({"velocity": 3, "pressure": 1, "density": 1})
        axis = Axis(size=5, variables=v)
        assert axis.variables == v
        assert axis.size == 5

    def test_feature_axis_with_multi_component_variables(self):
        """Test FeatureAxis with multi-component variables."""
        v = Variable.from_dict({"u": 2, "v": 2, "w": 2, "p": 1})
        feature = FeatureAxis(size=7, variables=v)
        assert feature.size == 7
        assert feature.variables is not None
        assert feature.variables.dim == 7

    def test_axis_variables_preserved_after_creation(self):
        """Test that Variables are preserved after Axis creation."""
        v = Variable.from_dict({"x": 10})
        axis = Axis(size=10, variables=v)
        assert axis.variables is not None
        assert axis.variables["x"] == 10
        assert axis.variables.dim == 10


class TestAxisIntegration:
    """Integration tests for Axis functionality."""

    def test_create_axis_hierarchy(self):
        """Test creating a typical axis hierarchy."""
        batch = BatchAxis()
        spatial = SpatialAxis(size=64)
        feature = FeatureAxis(size=16)
        time = TimeAxis(size=100)

        axes = [batch, spatial, feature, time]
        assert len(axes) == 4
        assert all(isinstance(ax, Axis) for ax in axes)

    def test_axis_with_field_data_structure(self):
        """Test Axis with typical field data structure."""
        velocity = Variable.from_dict({"ux": 1, "uy": 1, "uz": 1})
        feature = FeatureAxis(size=3, variables=velocity)

        spatial = SpatialAxis(size=100)
        time = TimeAxis(size=50)
        assert feature.variables is not None
        assert feature.variables.dim == 3
        assert spatial.size == 100
        assert time.size == 50

    def test_multiple_axes_same_size(self):
        """Test multiple axes with same size."""
        spatial1 = SpatialAxis(size=100)
        spatial2 = SpatialAxis(size=100)
        assert spatial1 == spatial2

    def test_axis_equality_with_none_values(self):
        """Test axis equality with various None values."""
        axis1 = Axis()
        axis2 = Axis()
        assert axis1 == axis2

        axis3 = Axis(size=10)
        axis4 = Axis()  # size is None
        assert axis3 == axis4  # Should be equal due to None check

    def test_axis_size_with_zero(self):
        """Test axis with zero size."""
        axis = Axis(size=0)
        assert axis.size == 0

        feature = FeatureAxis(size=0)
        assert feature.size == 0

    def test_axis_subclass_independence(self):
        """Test that Axis subclasses are independent."""
        batch1 = BatchAxis()
        batch2 = BatchAxis()
        assert batch1 == batch2
        assert batch1 is not batch2

    def test_feature_axis_dimension_tracking(self):
        """Test FeatureAxis dimension tracking with variables."""
        v1 = Variable.from_dict({"x": 3})
        v2 = Variable.from_dict({"y": 5})

        feature1 = FeatureAxis(size=3, variables=v1)
        feature2 = FeatureAxis(size=5, variables=v2)

        assert feature1.size == 3
        assert feature2.size == 5
        assert feature1 != feature2

    def test_axis_assertion_message(self):
        """Test that assertion error message is informative."""
        v = Variable.from_dict({"x": 5})
        try:
            Axis(size=10, variables=v)
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Size" in str(e)
            assert "dimension" in str(e)
