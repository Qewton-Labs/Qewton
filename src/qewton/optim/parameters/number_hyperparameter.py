from __future__ import annotations
import random
import math

from qewton.optim.parameters.helpers import (
    HyperParameterScale,
    HyperParameterState,
    HyperParameterCondition,
)
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class ContinuousHyperparameter(HyperParameter):
    """Represents a continuous HyperParameter given in the interval.

    Args:
        parameter_range (tuple | list): A tuple or list of two floats.
        initial_value (float | None, optional): An initial value.
            Defaults to None.
        state (HyperParameterState, optional): If this parameter is fixed or
            should be optimized. Defaults to HyperParameterState.OPTIMIZE.
        name (str, optional): The name of the parameter. Defaults to "".
        scale (HyperParameterScale, optional): If some special scaling should
            be applied for optimizing this parameter.
            Defaults to HyperParameterScale.LINEAR.
        power (float, optional): The power if power scaling law is used.
            Defaults to 2.0.
        active_when (None | HyperParameterCondition, optional): A condition that
            specifies when this Hyperparameter should be active. The condition
            depends on the values of other Hyperparameters. Accordingly this
            parameters is only sampled/set when the condition is fulfilled.
        default_grid (int | list, optional): The grid used for a GridSearchTuning.
            Either an int can be provided to build the grid automatically,
            or the grid itself can be passed. Defaults to 5.

    Raises:
        ValueError: If the parameters range has more than two values.
    """

    def __init__(
        self,
        parameter_range: tuple[float, float] | list[float],
        initial_value: float | None = None,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        scale: HyperParameterScale = HyperParameterScale.LINEAR,
        power: float = 2.0,  # only used for POWER scale
        active_when: None | HyperParameterCondition = None,
        default_grid: int | list = 5,
    ):
        assert len(parameter_range) == 2, "Range must be a tuple or list of length 2."
        if scale == HyperParameterScale.LOG and parameter_range[0] <= 0.0:
            raise ValueError("Logarithmic scaling in a negative range not supported!")
        if initial_value is None:
            initial_value = (parameter_range[0] + parameter_range[1]) / 2

        super().__init__(
            state=state,
            parameter_range=parameter_range,
            name=name,
            initial_value=initial_value,
            active_when=active_when,
            default_grid=default_grid,
        )
        self.scale = scale
        self.power = power

    def sample_parameter_random(self):
        lo, hi = self.parameter_range
        if self.scale == HyperParameterScale.LINEAR:
            return random.uniform(lo, hi)
        if self.scale == HyperParameterScale.LOG:
            sample_log = math.exp(random.uniform(math.log(lo), math.log(hi)))
            return sample_log
        # self.scale == HyperParameterScale.POWER:
        transformed_lo = lo ** (1 / self.power)
        transformed_hi = hi ** (1 / self.power)
        sample = random.uniform(transformed_lo, transformed_hi)
        return sample**self.power

    def sample_parameter_grid(self, n: int) -> list:
        lo, hi = self.parameter_range
        if self.scale == HyperParameterScale.LINEAR:
            return super().sample_parameter_grid(n)
        if self.scale == HyperParameterScale.LOG:
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            if n == 1:
                return [math.exp((log_lo + log_hi) / 2)]
            step = (log_hi - log_lo) / (n - 1)
            return [math.exp(log_lo + i * step) for i in range(n)]
        # self.scale == HyperParameterScale.POWER:
        transformed_lo = lo ** (1 / self.power)
        transformed_hi = hi ** (1 / self.power)
        if n == 1:
            return [((transformed_lo + transformed_hi) / 2) ** self.power]
        step = (transformed_hi - transformed_lo) / (n - 1)
        return [(transformed_lo + i * step) ** self.power for i in range(n)]

    def sample_from_unit(self, x: float):
        lo, hi = self.parameter_range
        if self.scale == HyperParameterScale.LINEAR:
            return lo + x * (hi - lo)
        if self.scale == HyperParameterScale.LOG:
            log_lo = math.log(lo)
            log_hi = math.log(hi)
            return math.exp(log_lo + x * (log_hi - log_lo))
        # self.scale == HyperParameterScale.POWER:
        transformed_lo = lo ** (1 / self.power)
        transformed_hi = hi ** (1 / self.power)
        step = transformed_hi - transformed_lo
        return (transformed_lo + x * step) ** self.power


class DiscreteHyperparameter(ContinuousHyperparameter):
    """Represents a discrete HyperParameter given in the interval.

    Args:
        parameter_range (tuple | list): A tuple or list of two ints.
        initial_value (float | None, optional): An initial value.
            Defaults to None.
        state (HyperParameterState, optional): If this parameter is fixed or
            should be optimized. Defaults to HyperParameterState.OPTIMIZE.
        name (str, optional): The name of the parameter. Defaults to "".
        scale (HyperParameterScale, optional): If some special scaling should
            be applied for optimizing this parameter.
            Defaults to HyperParameterScale.LINEAR.
        power (float, optional): The power if power scaling law is used.
            Defaults to 2.0.
        default_grid (int | list, optional): The grid used for a GridSearchTuning.
            Either an int can be provided to build the grid automatically,
            or the grid itself can be passed. Defaults to 5.

    Raises:
        ValueError: If the parameters range has more than two values.
    """

    def __init__(
        self,
        parameter_range: tuple[int, int] | list[int],
        initial_value: int | None = None,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        scale: HyperParameterScale = HyperParameterScale.LINEAR,
        power: float = 2.0,  # only used for POWER scale
        active_when: None | HyperParameterCondition = None,
        default_grid: int | list | None = 5,
    ):
        assert all(
            isinstance(x, int) for x in parameter_range
        ), "Range values must be integers."
        if default_grid is None:
            default_grid = int(parameter_range[1] - parameter_range[0] + 1)

        super().__init__(
            state=state,
            parameter_range=parameter_range,  # type: ignore
            initial_value=initial_value,
            name=name,
            scale=scale,
            power=power,
            active_when=active_when,
            default_grid=default_grid,
        )
        self.current_value = int(self.current_value)

    def set_value(self, new_value):
        self.current_value = int(new_value)

    def sample_parameter_random(self):
        if self.scale == HyperParameterScale.LINEAR:
            return random.randint(self.parameter_range[0], self.parameter_range[1])
        return int(round(super().sample_parameter_random()))

    def sample_parameter_grid(self, n: int) -> list:
        continuous_grid = super().sample_parameter_grid(n)
        return [int(value) for value in continuous_grid]

    def sample_from_unit(self, x: float):
        return int(round(super().sample_from_unit(x)))
