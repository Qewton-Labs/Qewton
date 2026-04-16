from collections import OrderedDict
import pytest

from pioneer.config.variables import Variable


class TestVariableInit:
    """Tests for Variable.__init__ method."""

    def test_init_empty(self):
        """Test creating an empty Variable."""
        v = Variable()
        assert len(v) == 0
        assert isinstance(v, OrderedDict)

    def test_init_with_name_and_dim(self):
        """Test creating a Variable with name and dimension."""
        v = Variable(name="x", dim=2)
        assert v["x"] == 2
        assert len(v) == 1

    def test_init_with_name_without_dim_raises_error(self):
        """Test that providing name without dim raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be provided"):
            Variable(name="x", dim=None)

    def test_init_with_name_only_raises_error(self):
        """Test that providing only name raises ValueError."""
        with pytest.raises(ValueError):
            Variable(name="x")

    def test_init_with_dim_only_no_error(self):
        """Test that providing only dim without name doesn't raise error."""
        v = Variable(dim=2)
        assert len(v) == 0

    def test_init_preserves_order(self):
        """Test that Variable maintains order as OrderedDict."""
        v = Variable()
        v["x"] = 2
        v["y"] = 3
        v["z"] = 1
        assert list(v.keys()) == ["x", "y", "z"]


class TestVariableFromDict:
    """Tests for Variable.from_dict classmethod."""

    def test_from_dict_empty(self):
        """Test creating Variable from empty dict."""
        v = Variable.from_dict({})
        assert len(v) == 0

    def test_from_dict_single_entry(self):
        """Test creating Variable from dict with single entry."""
        v = Variable.from_dict({"x": 2})
        assert v["x"] == 2
        assert len(v) == 1

    def test_from_dict_multiple_entries(self):
        """Test creating Variable from dict with multiple entries."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        assert v["x"] == 2
        assert v["y"] == 3
        assert v["z"] == 1
        assert len(v) == 3

    def test_from_dict_preserves_order(self):
        """Test that from_dict preserves dictionary order."""
        input_dict = {"a": 1, "b": 2, "c": 3}
        v = Variable.from_dict(input_dict)
        assert list(v.keys()) == list(input_dict.keys())

    def test_from_dict_with_large_dimensions(self):
        """Test from_dict with large dimension values."""
        v = Variable.from_dict({"large": 1000000})
        assert v["large"] == 1000000

    def test_from_dict_with_zero_dimension(self):
        """Test from_dict with zero dimension."""
        v = Variable.from_dict({"zero": 0})
        assert v["zero"] == 0


class TestVariableMul:
    """Tests for Variable.__mul__ operator (combines variables)."""

    def test_mul_empty_variables(self):
        """Test multiplying two empty variables."""
        v1 = Variable()
        v2 = Variable()
        result = v1 * v2
        assert len(result) == 0

    def test_mul_one_empty_one_nonempty(self):
        """Test multiplying empty variable with non-empty."""
        v1 = Variable()
        v2 = Variable.from_dict({"x": 2})
        result = v1 * v2
        assert result["x"] == 2

    def test_mul_two_nonempty_no_overlap(self):
        """Test multiplying two variables with no overlapping keys."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"y": 3})
        result = v1 * v2
        assert result["x"] == 2
        assert result["y"] == 3
        assert len(result) == 2

    def test_mul_two_nonempty_with_overlap(self):
        """Test multiplying two variables with overlapping keys."""
        v1 = Variable.from_dict({"x": 2, "y": 1})
        v2 = Variable.from_dict({"y": 3, "z": 1})
        with pytest.raises(ValueError):
            _ = v1 * v2

    def test_mul_returns_variable_instance(self):
        """Test that multiplication returns Variable instance."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"y": 3})
        result = v1 * v2
        assert isinstance(result, Variable)

    def test_mul_chain(self):
        """Test chaining multiple multiplications."""
        v1 = Variable.from_dict({"x": 1})
        v2 = Variable.from_dict({"y": 2})
        v3 = Variable.from_dict({"z": 3})
        result = v1 * v2 * v3
        assert result["x"] == 1
        assert result["y"] == 2
        assert result["z"] == 3


class TestVariableAdd:
    """Tests for Variable.__add__ operator (should equal __mul__)."""

    def test_add_empty_variables(self):
        """Test adding two empty variables."""
        v1 = Variable()
        v2 = Variable()
        result = v1 + v2
        assert len(result) == 0

    def test_add_two_nonempty_no_overlap(self):
        """Test adding two variables with no overlapping keys."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"y": 3})
        result = v1 + v2
        assert result["x"] == 2
        assert result["y"] == 3

    def test_add_two_nonempty_with_overlap(self):
        """Test adding two variables with overlapping keys."""
        v1 = Variable.from_dict({"x": 2, "y": 1})
        v2 = Variable.from_dict({"y": 3, "z": 1})
        result = v1 + v2
        assert result["x"] == 2
        assert result["y"] == 4
        assert result["z"] == 1

    def test_add_equals_mul(self):
        """Test that addition behaves identically to multiplication."""
        v1 = Variable.from_dict({"a": 1, "b": 2})
        v2 = Variable.from_dict({"b": 3, "c": 4})
        assert dict(v1 + v2) == dict(v1 * v2)

    def test_add_chain(self):
        """Test chaining multiple additions."""
        v1 = Variable.from_dict({"x": 1})
        v2 = Variable.from_dict({"y": 2})
        v3 = Variable.from_dict({"z": 3})
        result = v1 + v2 + v3
        assert result["x"] == 1
        assert result["y"] == 2
        assert result["z"] == 3


class TestVariableDim:
    """Tests for Variable.dim property."""

    def test_dim_empty(self):
        """Test dim property on empty variable."""
        v = Variable()
        assert v.dim == 0

    def test_dim_single_entry(self):
        """Test dim property with single entry."""
        v = Variable.from_dict({"x": 5})
        assert v.dim == 5

    def test_dim_multiple_entries(self):
        """Test dim property with multiple entries."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 5})
        assert v.dim == 10

    def test_dim_with_zero_values(self):
        """Test dim property with zero-valued entries."""
        v = Variable.from_dict({"x": 0, "y": 5})
        assert v.dim == 5

    def test_dim_after_modification(self):
        """Test dim property after modifying variable."""
        v = Variable.from_dict({"x": 2})
        assert v.dim == 2
        v["y"] = 3
        assert v.dim == 5

    def test_dim_after_multiplication(self):
        """Test dim property after multiplication."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"y": 3})
        result = v1 * v2
        assert result.dim == 5


class TestVariableRepr:
    """Tests for Variable.__repr__ method."""

    def test_repr_empty(self):
        """Test repr of empty variable."""
        v = Variable()
        assert repr(v) == "Variable({})"

    def test_repr_single_entry(self):
        """Test repr with single entry."""
        v = Variable.from_dict({"x": 2})
        assert repr(v) == "Variable({'x': 2})"

    def test_repr_multiple_entries(self):
        """Test repr with multiple entries."""
        v = Variable.from_dict({"x": 2, "y": 3})
        assert repr(v) == "Variable({'x': 2, 'y': 3})"

    def test_repr_contains_class_name(self):
        """Test that repr contains class name."""
        v = Variable.from_dict({"x": 1})
        assert "Variable" in repr(v)


class TestVariableHash:
    """Tests for Variable.__hash__ method."""

    def test_hash_identical_variables_same(self):
        """Test that identical variables produce same hash."""
        v1 = Variable.from_dict({"x": 2, "y": 3})
        v2 = Variable.from_dict({"x": 2, "y": 3})
        assert hash(v1) == hash(v2)

    def test_hash_different_variables_different(self):
        """Test that different variables produce different hashes."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"y": 3})
        assert hash(v1) != hash(v2)


class TestVariableContains:
    """Tests for Variable.__contains__ method."""

    def test_contains_string_key_present(self):
        """Test contains with string key that is present."""
        v = Variable.from_dict({"x": 2, "y": 3})
        assert "x" in v
        assert "y" in v

    def test_contains_string_key_absent(self):
        """Test contains with string key that is absent."""
        v = Variable.from_dict({"x": 2})
        assert "z" not in v

    def test_contains_empty_variable(self):
        """Test contains on empty variable."""
        v = Variable()
        assert "x" not in v

    def test_contains_variable_exact_match(self):
        """Test contains with Variable that matches exactly."""
        v1 = Variable.from_dict({"x": 2, "y": 3})
        v2 = Variable.from_dict({"x": 2, "y": 3})
        assert v2 in v1

    def test_contains_variable_subset(self):
        """Test contains with Variable that is subset."""
        v1 = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        v2 = Variable.from_dict({"x": 2, "y": 3})
        assert v2 in v1

    def test_contains_variable_single_key_subset(self):
        """Test contains with Variable that is single-key subset."""
        v1 = Variable.from_dict({"x": 2, "y": 3})
        v2 = Variable.from_dict({"x": 2})
        assert v2 in v1

    def test_contains_variable_different_value(self):
        """Test contains with Variable with different value."""
        v1 = Variable.from_dict({"x": 2, "y": 3})
        v2 = Variable.from_dict({"x": 5, "y": 3})
        assert v2 not in v1

    def test_contains_variable_missing_key(self):
        """Test contains with Variable with missing key."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"x": 2, "y": 3})
        assert v2 not in v1

    def test_contains_variable_empty_in_nonempty(self):
        """Test if empty Variable is contained in non-empty Variable."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable()
        assert v2 in v1

    def test_contains_variable_nonempty_in_empty(self):
        """Test if non-empty Variable is contained in empty Variable."""
        v1 = Variable()
        v2 = Variable.from_dict({"x": 2})
        assert v2 not in v1

    def test_contains_variable_empty_in_empty(self):
        """Test if empty Variable is contained in empty Variable."""
        v1 = Variable()
        v2 = Variable()
        assert v2 in v1

    def test_contains_other_type(self):
        """Test contains with other types (not string or Variable)."""
        v = Variable.from_dict({"x": 2})
        assert 42 not in v
        assert None not in v
        assert 3.14 not in v


class TestVariableGetitem:
    """Tests for Variable.__getitem__ method with different access types."""

    def test_getitem_string_key(self):
        """Test getitem with string key."""
        v = Variable.from_dict({"x": 2, "y": 3})
        assert v["x"] == 2
        assert v["y"] == 3

    def test_getitem_string_key_not_found(self):
        """Test getitem with string key that doesn't exist."""
        v = Variable.from_dict({"x": 2})
        with pytest.raises(KeyError):
            _ = v["z"]

    def test_getitem_list_single_key(self):
        """Test getitem with list containing single key."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        result = v[["x"]]
        assert isinstance(result, Variable)
        assert result["x"] == 2
        assert len(result) == 1

    def test_getitem_list_multiple_keys(self):
        """Test getitem with list containing multiple keys."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        result = v[["x", "z"]]
        assert result["x"] == 2
        assert result["z"] == 1
        assert len(result) == 2
        assert "y" not in result

    def test_getitem_list_empty(self):
        """Test getitem with empty list."""
        v = Variable.from_dict({"x": 2, "y": 3})
        result = v[[]]
        assert len(result) == 0
        assert isinstance(result, Variable)

    def test_getitem_list_preserves_order(self):
        """Test that list getitem preserves order."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        result = v[["z", "x", "y"]]
        assert list(result.keys()) == ["z", "x", "y"]

    def test_getitem_tuple_single_key(self):
        """Test getitem with tuple containing single key."""
        v = Variable.from_dict({"x": 2, "y": 3})
        result = v[("x",)]
        assert isinstance(result, Variable)
        assert result["x"] == 2

    def test_getitem_slice_basic(self):
        """Test getitem with basic slice."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        result = v["x":"z"]
        assert "x" in result
        assert "y" in result
        # Note: slicing is exclusive at the end, similar to list slicing
        # Need to check actual behavior

    def test_getitem_slice_start_only(self):
        """Test getitem with slice start only."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        result = v["x":]
        assert isinstance(result, Variable)
        assert "x" in result

    def test_getitem_slice_stop_only(self):
        """Test getitem with slice stop only."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        result = v[:"y"]
        assert isinstance(result, Variable)

    def test_getitem_slice_with_step(self):
        """Test getitem with slice with step."""
        v = Variable.from_dict({"a": 1, "b": 2, "c": 3, "d": 4})
        result = v["a":"d":2]
        assert isinstance(result, Variable)

    def test_getitem_slice_returns_variable(self):
        """Test that slice getitem returns Variable instance."""
        v = Variable.from_dict({"x": 2, "y": 3})
        result = v["x":"y"]
        assert isinstance(result, Variable)

    def test_getitem_preserves_dimension_values(self):
        """Test that getitem preserves correct dimension values."""
        v = Variable.from_dict({"x": 2, "y": 3, "z": 5})
        result = v[["x", "z"]]
        assert result["x"] == 2
        assert result["z"] == 5
        assert result.dim == 7

    def test_getitem_list_converts_to_int(self):
        """Test that list getitem converts values to int."""
        v = Variable.from_dict({"x": 2, "y": 3})
        result = v[["x"]]
        assert isinstance(result["x"], int)

    def test_getitem_tuple_converts_to_int(self):
        """Test that tuple getitem converts values to int."""
        v = Variable.from_dict({"x": 2, "y": 3})
        result = v[("x",)]
        assert isinstance(result["x"], int)


class TestVariableIntegration:
    """Integration tests combining multiple operations."""

    def test_workflow_create_combine_access(self):
        """Test typical workflow: create, combine, and access variables."""
        v_spatial = Variable.from_dict({"x": 2, "y": 2, "z": 1})
        v_temporal = Variable.from_dict({"t": 1})
        v_combined = v_spatial * v_temporal

        assert v_combined.dim == 6
        assert "x" in v_combined
        assert "t" in v_combined

    def test_workflow_slice_and_multiply(self):
        """Test slicing a variable and then multiplying."""
        v1 = Variable.from_dict({"x": 2, "y": 3, "z": 1})
        v_sliced = v1[["x", "y"]]
        v2 = Variable.from_dict({"w": 1})
        result = v_sliced * v2

        assert result.dim == 6
        assert "z" not in result
        assert "w" in result

    def test_workflow_multiple_operations(self):
        """Test multiple operations in sequence."""
        v1 = Variable(name="u", dim=2)
        v2 = Variable.from_dict({"p": 1})
        v_combined = v1 + v2

        assert v_combined["u"] == 2
        assert v_combined["p"] == 1
        assert v_combined.dim == 3
        assert v1 in v_combined

    def test_contains_after_operations(self):
        """Test contains check after various operations."""
        v1 = Variable.from_dict({"x": 2, "y": 3})
        v2 = Variable.from_dict({"z": 1})
        v_combined = v1 * v2

        assert Variable.from_dict({"x": 2}) in v_combined
        assert Variable.from_dict({"y": 3, "z": 1}) in v_combined
        assert Variable.from_dict({"w": 1}) not in v_combined

    def test_repr_after_multiplication(self):
        """Test repr after multiplication."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"y": 3})
        result = v1 * v2
        assert "Variable" in repr(result)
        assert "x" in repr(result)
        assert "y" in repr(result)

    def test_dimension_accumulation(self):
        """Test that dimensions accumulate correctly across operations."""
        v1 = Variable.from_dict({"a": 1})
        v2 = Variable.from_dict({"b": 2})
        v3 = Variable.from_dict({"c": 3})

        result = v1 * v2 * v3
        assert result.dim == 6

    def test_overlapping_keys_accumulation(self):
        """Test dimension accumulation with overlapping keys."""
        v1 = Variable.from_dict({"x": 2})
        v2 = Variable.from_dict({"x": 3})
        v3 = Variable.from_dict({"x": 1})

        result = v1 * v2 * v3
        assert result["x"] == 6
        assert result.dim == 6

    def test_getitem_with_nonexistent_key_in_list(self):
        """Test getitem with list containing non-existent key."""
        v = Variable.from_dict({"x": 2, "y": 3})
        with pytest.raises(KeyError):
            _ = v[["x", "z"]]

    def test_complex_slicing_scenario(self):
        """Test complex slicing scenario."""
        v = Variable.from_dict({"x": 1, "y": 2, "z": 3, "w": 4})
        result = v["x":"w"]
        assert isinstance(result, Variable)
        # Verify keys are included based on slice logic
        assert "x" in result or "y" in result

    def test_ordered_dict_behavior(self):
        """Test that Variable preserves OrderedDict behavior."""
        v = Variable()
        v["z"] = 1
        v["a"] = 2
        v["m"] = 3
        assert list(v.keys()) == ["z", "a", "m"]

    def test_large_scale_operations(self):
        """Test operations with many variables."""
        variables = [Variable(name=f"v{i}", dim=i + 1) for i in range(10)]
        result = variables[0]
        for v in variables[1:]:
            result = result * v

        assert len(result) == 10
        assert result.dim == sum(range(1, 11))
