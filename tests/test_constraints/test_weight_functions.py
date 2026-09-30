import math

from qewton.constraints.weight_functions import (
    LinearRampWeight,
    StepWeight,
    CosineRampWeight,
)
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
    ContinuousHyperparameter,
)


def test_linear_ramp_weight_basic_behavior():
    weight = LinearRampWeight(start_iteration=10, full_iteration=20)

    assert weight(0) == 0.0
    assert weight(10) == 0.0
    assert math.isclose(weight(15), 0.5)
    assert weight(20) == 1.0
    assert weight(100) == 1.0


def test_linear_ramp_with_hyperparameters():
    start = DiscreteHyperparameter((0, 100), initial_value=5, name="start")
    end = DiscreteHyperparameter((0, 100), initial_value=25, name="end")
    target = ContinuousHyperparameter((0.5, 2.0), initial_value=1.5, name="target")

    weight = LinearRampWeight(
        start_iteration=start,
        full_iteration=end,
        initial_weight=0.0,
        target_weight=target,
    )

    assert math.isclose(weight(15), 0.75)
    assert math.isclose(weight(25), 1.5)


def test_step_weight_behavior():
    weight = StepWeight(activate_iteration=12, inactive_weight=0.0, active_weight=1.0)

    assert weight(0) == 0.0
    assert weight(11) == 0.0
    assert weight(12) == 1.0
    assert weight(100) == 1.0


def test_cosine_ramp_weight_behavior():
    weight = CosineRampWeight(start_iteration=0, full_iteration=10)

    assert weight(0) == 0.0
    assert math.isclose(weight(5), 0.5, rel_tol=1e-6)
    assert math.isclose(weight(10), 1.0)


def test_degenerate_interval_falls_back_to_step_behavior():
    linear = LinearRampWeight(start_iteration=10, full_iteration=10)
    cosine = CosineRampWeight(start_iteration=8, full_iteration=5)

    assert linear(9) == 0.0
    assert linear(10) == 1.0
    assert cosine(4) == 0.0
    assert cosine(8) == 1.0
