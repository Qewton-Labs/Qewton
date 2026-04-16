import math
import random
import pytest

from pioneer.optim.parameters.hyperparameter_base import (
    HyperParameter,
    HyperParameterState,
    # HyperParameterScale,
    # ContinuousHyperparameter,
    # DiscreteHyperparameter,
    # CategoricalHyperparameter,
    # BooleanHyperparameter,
)


class TestHyperParameterState:
    """Tests for HyperParameterState enum."""

    def test_fixed_state_value(self):
        assert HyperParameterState.FIXED.value == 1

    def test_optimize_state_value(self):
        assert HyperParameterState.OPTIMIZE.value == 2


class TestHyperParameterScale:
    """Tests for HyperParameterScale enum."""

    def test_linear_scale_value(self):
        assert HyperParameterScale.LINEAR.value == "linear"

    def test_log_scale_value(self):
        assert HyperParameterScale.LOG.value == "log"

    def test_power_scale_value(self):
        assert HyperParameterScale.POWER.value == "power"


class TestHyperParameterBase:
    """Tests for HyperParameter base class."""

    def test_init_with_defaults(self):
        hp = HyperParameter(parameter_range=(0, 10), initial_value=5, name="test_param")
        assert hp.parameter_range == (0, 10)
        assert hp.current_value == 5
        assert hp.state == HyperParameterState.OPTIMIZE
        assert hp.name == "test_param"

    def test_init_with_fixed_state(self):
        hp = HyperParameter(
            parameter_range=(0, 10),
            initial_value=5,
            state=HyperParameterState.FIXED,
            name="fixed_param",
        )
        assert hp.state == HyperParameterState.FIXED

    def test_init_with_list_range(self):
        hp = HyperParameter(parameter_range=[0, 10], initial_value=5)
        assert hp.parameter_range == [0, 10]

    def test_is_fixed_property_true(self):
        hp = HyperParameter(
            parameter_range=(0, 10), initial_value=5, state=HyperParameterState.FIXED
        )
        assert hp.is_fixed is True

    def test_is_fixed_property_false(self):
        hp = HyperParameter(
            parameter_range=(0, 10), initial_value=5, state=HyperParameterState.OPTIMIZE
        )
        assert hp.is_fixed is False

    def test_value_property(self):
        hp = HyperParameter(parameter_range=(0, 10), initial_value=5)
        assert hp.value == 5

    def test_set_value(self):
        hp = HyperParameter(parameter_range=(0, 10), initial_value=5)
        hp.set_value(7)
        assert hp.current_value == 7
        assert hp.value == 7

    def test_sample_parameter_random_not_implemented(self):
        hp = HyperParameter(parameter_range=(0, 10), initial_value=5)
        with pytest.raises(NotImplementedError):
            hp.sample_parameter_random()

    def test_sample_parameter_grid_single_point(self):
        hp = HyperParameter(parameter_range=(0, 10), initial_value=5)
        grid = hp.sample_parameter_grid(1)
        assert len(grid) == 1
        assert grid[0] == 5.0

    def test_sample_parameter_grid_multiple_points(self):
        hp = HyperParameter(parameter_range=(0, 10), initial_value=5)
        grid = hp.sample_parameter_grid(5)
        assert len(grid) == 5
        assert grid[0] == 0
        assert grid[-1] == 10
        # Check uniform spacing
        for i in range(len(grid) - 1):
            assert abs((grid[i + 1] - grid[i]) - 2.5) < 1e-10


class TestHyperParameterFromValue:
    """Tests for HyperParameter.from_value factory method."""

    def test_from_value_with_hyperparameter_object(self):
        original = ContinuousHyperparameter(
            parameter_range=(0, 10), initial_value=5, name="test"
        )
        result = HyperParameter.from_value(original)
        assert result is original

    def test_from_value_with_hyperparameter_unnamed(self):
        original = ContinuousHyperparameter(
            parameter_range=(0, 10), initial_value=5, name=""
        )
        result = HyperParameter.from_value(original, name="new_name")
        assert result is original
        assert result.name == "new_name"

    def test_from_value_with_bool_true(self):
        result = HyperParameter.from_value(True, name="bool_param")
        assert isinstance(result, BooleanHyperparameter)
        assert result.value is True
        assert result.is_fixed is True
        assert result.name == "bool_param"

    def test_from_value_with_bool_false(self):
        result = HyperParameter.from_value(False, name="bool_param")
        assert isinstance(result, BooleanHyperparameter)
        assert result.value is False
        assert result.is_fixed is True

    def test_from_value_with_int(self):
        result = HyperParameter.from_value(42, name="int_param")
        assert isinstance(result, DiscreteHyperparameter)
        assert result.value == 42
        assert result.is_fixed is True
        assert result.parameter_range == (42, 42)

    def test_from_value_with_float(self):
        result = HyperParameter.from_value(3.14, name="float_param")
        assert isinstance(result, ContinuousHyperparameter)
        assert result.value == 3.14
        assert result.is_fixed is True
        assert result.parameter_range == (3.14, 3.14)

    def test_from_value_with_string(self):
        result = HyperParameter.from_value("test_string", name="string_param")
        assert isinstance(result, CategoricalHyperparameter)
        assert result.value == "test_string"
        assert result.is_fixed is True

    def test_from_value_without_name_raises_assertion(self):
        with pytest.raises(AssertionError):
            HyperParameter.from_value(42)


class TestContinuousHyperparameter:
    """Tests for ContinuousHyperparameter class."""

    def test_init_with_defaults(self):
        hp = ContinuousHyperparameter(parameter_range=(0.0, 10.0), initial_value=5.0)
        assert hp.parameter_range == (0.0, 10.0)
        assert hp.current_value == 5.0
        assert hp.scale == HyperParameterScale.LINEAR
        assert hp.power == 2.0

    def test_init_without_initial_value(self):
        hp = ContinuousHyperparameter(parameter_range=(0.0, 10.0))
        assert hp.current_value == 5.0

    def test_init_with_linear_scale(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.0, 10.0), scale=HyperParameterScale.LINEAR
        )
        assert hp.scale == HyperParameterScale.LINEAR

    def test_init_with_log_scale(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.1, 100.0), scale=HyperParameterScale.LOG
        )
        assert hp.scale == HyperParameterScale.LOG

    def test_init_with_power_scale(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.0, 10.0), scale=HyperParameterScale.POWER, power=3.0
        )
        assert hp.scale == HyperParameterScale.POWER
        assert hp.power == 3.0

    def test_init_with_negative_log_range_raises_error(self):
        with pytest.raises(ValueError):
            ContinuousHyperparameter(
                parameter_range=(-1.0, 10.0), scale=HyperParameterScale.LOG
            )

    def test_init_with_zero_log_range_raises_error(self):
        with pytest.raises(ValueError):
            ContinuousHyperparameter(
                parameter_range=(0.0, 10.0), scale=HyperParameterScale.LOG
            )

    def test_init_with_invalid_range_length(self):
        with pytest.raises(AssertionError):
            ContinuousHyperparameter(parameter_range=(0, 5, 10))

    def test_sample_parameter_random_linear(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.0, 10.0), scale=HyperParameterScale.LINEAR
        )
        random.seed(42)
        for _ in range(100):
            sample = hp.sample_parameter_random()
            assert 0.0 <= sample <= 10.0

    def test_sample_parameter_random_log(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.1, 100.0), scale=HyperParameterScale.LOG
        )
        random.seed(42)
        for _ in range(100):
            sample = hp.sample_parameter_random()
            assert 0.1 <= sample <= 100.0

    def test_sample_parameter_random_power(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.0, 10.0), scale=HyperParameterScale.POWER, power=2.0
        )
        random.seed(42)
        for _ in range(100):
            sample = hp.sample_parameter_random()
            assert 0.0 <= sample <= 10.0

    def test_sample_parameter_grid_linear_single(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.0, 10.0), scale=HyperParameterScale.LINEAR
        )
        grid = hp.sample_parameter_grid(1)
        assert len(grid) == 1
        assert grid[0] == 5.0

    def test_sample_parameter_grid_linear_multiple(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.0, 10.0), scale=HyperParameterScale.LINEAR
        )
        grid = hp.sample_parameter_grid(5)
        assert len(grid) == 5
        assert grid[0] == 0.0
        assert grid[-1] == 10.0

    def test_sample_parameter_grid_log_single(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.1, 100.0), scale=HyperParameterScale.LOG
        )
        grid = hp.sample_parameter_grid(1)
        assert len(grid) == 1
        assert abs(grid[0] - math.sqrt(10)) < 1e-10

    def test_sample_parameter_grid_log_multiple(self):
        hp = ContinuousHyperparameter(
            parameter_range=(0.1, 100.0), scale=HyperParameterScale.LOG
        )
        grid = hp.sample_parameter_grid(3)
        assert len(grid) == 3
        assert abs(grid[0] - 0.1) < 1e-10
        assert abs(grid[-1] - 100.0) < 1e-10

    def test_sample_parameter_grid_power_single(self):
        hp = ContinuousHyperparameter(
            parameter_range=(1.0, 100.0), scale=HyperParameterScale.POWER, power=2.0
        )
        grid = hp.sample_parameter_grid(1)
        assert len(grid) == 1

    def test_sample_parameter_grid_power_multiple(self):
        hp = ContinuousHyperparameter(
            parameter_range=(1.0, 100.0), scale=HyperParameterScale.POWER, power=2.0
        )
        grid = hp.sample_parameter_grid(3)
        assert len(grid) == 3


class TestDiscreteHyperparameter:
    """Tests for DiscreteHyperparameter class."""

    def test_init_with_valid_integers(self):
        hp = DiscreteHyperparameter(parameter_range=(0, 10), initial_value=5)
        assert hp.parameter_range == (0, 10)
        assert hp.current_value == 5
        assert isinstance(hp.current_value, int)

    def test_init_without_initial_value(self):
        hp = DiscreteHyperparameter(parameter_range=(0, 10))
        assert hp.current_value == 5

    def test_init_with_non_integer_range_raises_error(self):
        with pytest.raises(AssertionError):
            DiscreteHyperparameter(parameter_range=(0.5, 10.5))

    def test_init_with_mixed_type_range_raises_error(self):
        with pytest.raises(AssertionError):
            DiscreteHyperparameter(parameter_range=(0, 10.5))

    def test_set_value_converts_to_int(self):
        hp = DiscreteHyperparameter(parameter_range=(0, 10))
        hp.set_value(5.7)
        assert hp.current_value == 5
        assert isinstance(hp.current_value, int)

    def test_sample_parameter_random_linear(self):
        hp = DiscreteHyperparameter(
            parameter_range=(0, 10), scale=HyperParameterScale.LINEAR
        )
        random.seed(42)
        for _ in range(100):
            sample = hp.sample_parameter_random()
            assert 0 <= sample <= 10
            assert isinstance(sample, int)

    def test_sample_parameter_random_log(self):
        hp = DiscreteHyperparameter(
            parameter_range=(1, 100), scale=HyperParameterScale.LOG
        )
        random.seed(42)
        for _ in range(100):
            sample = hp.sample_parameter_random()
            assert 1 <= sample <= 100
            assert isinstance(sample, int)

    def test_sample_parameter_grid_linear(self):
        hp = DiscreteHyperparameter(
            parameter_range=(0, 10), scale=HyperParameterScale.LINEAR
        )
        grid = hp.sample_parameter_grid(5)
        assert len(grid) == 5
        assert all(isinstance(x, int) for x in grid)
        assert grid[0] == 0
        assert grid[-1] == 10

    def test_sample_parameter_grid_log(self):
        hp = DiscreteHyperparameter(
            parameter_range=(1, 100), scale=HyperParameterScale.LOG
        )
        grid = hp.sample_parameter_grid(3)
        assert len(grid) == 3
        assert all(isinstance(x, int) for x in grid)


class TestCategoricalHyperparameter:
    """Tests for CategoricalHyperparameter class."""

    def test_init_with_tuple_categories(self):
        hp = CategoricalHyperparameter(categories=("a", "b", "c"), initial_value="b")
        assert hp.parameter_range == ("a", "b", "c")
        assert hp.current_value == "b"

    def test_init_with_list_categories(self):
        hp = CategoricalHyperparameter(categories=["a", "b", "c"], initial_value="b")
        assert hp.parameter_range == ["a", "b", "c"]
        assert hp.current_value == "b"

    def test_init_without_initial_value(self):
        hp = CategoricalHyperparameter(categories=("a", "b", "c"))
        assert hp.current_value == "a"

    def test_sample_parameter_random(self):
        hp = CategoricalHyperparameter(categories=("a", "b", "c"))
        random.seed(42)
        for _ in range(100):
            sample = hp.sample_parameter_random()
            assert sample in ("a", "b", "c")

    def test_sample_parameter_grid_with_tuple(self):
        hp = CategoricalHyperparameter(categories=("a", "b", "c"))
        grid = hp.sample_parameter_grid(2)
        assert len(grid) == 3  # Returns all categories
        assert set(grid) == {"a", "b", "c"}

    def test_sample_parameter_grid_with_list(self):
        hp = CategoricalHyperparameter(categories=["a", "b", "c"])
        grid = hp.sample_parameter_grid(2)
        assert len(grid) == 3
        assert grid == ["a", "b", "c"]

    def test_with_integer_categories(self):
        hp = CategoricalHyperparameter(categories=[1, 2, 3, 4], initial_value=2)
        assert hp.current_value == 2
        random.seed(42)
        sample = hp.sample_parameter_random()
        assert sample in [1, 2, 3, 4]

    def test_with_mixed_type_categories(self):
        hp = CategoricalHyperparameter(categories=[1, "two", 3.0, True], initial_value=1)
        assert hp.current_value == 1


class TestBooleanHyperparameter:
    """Tests for BooleanHyperparameter class."""

    def test_init_with_true(self):
        hp = BooleanHyperparameter(initial_value=True)
        assert hp.current_value is True
        assert hp.parameter_range == [True, False]

    def test_init_with_false(self):
        hp = BooleanHyperparameter(initial_value=False)
        assert hp.current_value is False
        assert hp.parameter_range == [True, False]

    def test_init_with_fixed_state(self):
        hp = BooleanHyperparameter(initial_value=True, state=HyperParameterState.FIXED)
        assert hp.is_fixed is True

    def test_init_with_optimize_state(self):
        hp = BooleanHyperparameter(
            initial_value=False, state=HyperParameterState.OPTIMIZE
        )
        assert hp.is_fixed is False

    def test_init_with_name(self):
        hp = BooleanHyperparameter(initial_value=True, name="bool_param")
        assert hp.name == "bool_param"

    def test_sample_parameter_random(self):
        hp = BooleanHyperparameter(initial_value=True)
        random.seed(42)
        samples = [hp.sample_parameter_random() for _ in range(100)]
        assert all(isinstance(s, bool) for s in samples)
        assert True in samples
        assert False in samples

    def test_sample_parameter_grid(self):
        hp = BooleanHyperparameter(initial_value=True)
        grid = hp.sample_parameter_grid(2)
        assert len(grid) == 2
        assert set(grid) == {True, False}

    def test_set_value(self):
        hp = BooleanHyperparameter(initial_value=True)
        hp.set_value(False)
        assert hp.current_value is False
        assert hp.value is False


class TestIntegration:
    """Integration tests across multiple classes."""

    def test_create_hyperparameters_for_model_tuning(self):
        """Test creating a set of hyperparameters for model tuning."""
        hp_lr = ContinuousHyperparameter(
            parameter_range=(1e-5, 1e-1),
            scale=HyperParameterScale.LOG,
            name="learning_rate",
        )
        hp_batch = DiscreteHyperparameter(
            parameter_range=(16, 256),
            scale=HyperParameterScale.POWER,
            power=2.0,
            name="batch_size",
        )
        hp_dropout = ContinuousHyperparameter(
            parameter_range=(0.0, 0.5), scale=HyperParameterScale.LINEAR, name="dropout"
        )
        hp_optimizer = CategoricalHyperparameter(
            categories=["adam", "sgd", "rmsprop"], name="optimizer"
        )
        hp_use_bn = BooleanHyperparameter(initial_value=True, name="use_batch_norm")

        hyperparameters = [hp_lr, hp_batch, hp_dropout, hp_optimizer, hp_use_bn]
        assert len(hyperparameters) == 5

        for hp in hyperparameters:
            assert hp.name != ""
            sample = hp.sample_parameter_random()
            assert sample is not None

    def test_grid_search_space(self):
        """Test creating a grid search space."""
        hp_lr = ContinuousHyperparameter(
            parameter_range=(0.001, 0.1), state=HyperParameterState.OPTIMIZE
        )
        hp_epochs = DiscreteHyperparameter(
            parameter_range=(10, 100), state=HyperParameterState.OPTIMIZE
        )

        lr_grid = hp_lr.sample_parameter_grid(3)
        epoch_grid = hp_epochs.sample_parameter_grid(3)

        assert len(lr_grid) == 3
        assert len(epoch_grid) == 3

        grid_search_space = [(lr, epochs) for lr in lr_grid for epochs in epoch_grid]
        assert len(grid_search_space) == 9

    def test_fixed_and_optimize_parameters(self):
        """Test mixing fixed and optimizable parameters."""
        hp_fixed_lr = ContinuousHyperparameter(
            parameter_range=(0.01, 0.01),
            initial_value=0.01,
            state=HyperParameterState.FIXED,
        )
        hp_variable_wd = ContinuousHyperparameter(
            parameter_range=(1e-5, 1e-2), state=HyperParameterState.OPTIMIZE
        )

        assert hp_fixed_lr.is_fixed is True
        assert hp_variable_wd.is_fixed is False

    def test_from_value_creates_fixed_parameters(self):
        """Test that from_value creates FIXED parameters."""
        hp_int = HyperParameter.from_value(42, name="answer")
        hp_float = HyperParameter.from_value(3.14, name="pi")
        hp_bool = HyperParameter.from_value(True, name="flag")

        assert hp_int.is_fixed is True
        assert hp_float.is_fixed is True
        assert hp_bool.is_fixed is True

    def test_parameter_range_types(self):
        """Test that parameter ranges can be tuples or lists."""
        hp_tuple = ContinuousHyperparameter(parameter_range=(0.0, 10.0))
        hp_list = ContinuousHyperparameter(parameter_range=[0.0, 10.0])

        assert hp_tuple.sample_parameter_grid(3) == hp_list.sample_parameter_grid(3)
