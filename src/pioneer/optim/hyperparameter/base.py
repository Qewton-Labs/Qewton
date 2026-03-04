from __future__ import annotations
from enum import Enum
from typing import Any
import random
import math


class HyperParameterState(Enum):
    """Denotes the state of a HyperParameter and how it is handled in the
    tuning process.

    FIXED
        The Hyperparameter does not change and is ignored while tuning.
    OPTIMIZE
        The Hyperparameter is changed in the tuning process.
    """

    FIXED = 1
    OPTIMIZE = 2


class HyperParameterScale(Enum):
    """Influence the sampling of HyperParameters.

    LINEAR
        A uniform sampling in the given parameter range.
    LOG
        A uniform logarithmic sampling in the given parameter range.
    POWER
        Sampling via a power law x**power, where power can be set in the
        HyperParameter class. power > 1 leads to a bias toward lower values,
        p < 1 leads to a bias toward higher values.
    """

    LINEAR = "linear"
    LOG = "log"
    POWER = "power"


class HyperParameter:
    """Abstract HyperParameter class that represents tunable parameters."""

    # TODO: dtype needed?
    # TODO: What about interconnected HyperParameters? E.g. we want to try out
    # different algorithms with different hyperparameters each.
    def __init__(
        self,
        parameter_range: tuple | list,
        initial_value,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
    ):
        """
        Args:
            parameter_range (tuple | list): Allowed range of values for this parameter.
            initial_value (_type_): Initial value for this parameter.
            state (HyperParameterState, optional): If this value should be fixed or be
                optimized. Defaults to HyperParameterState.OPTIMIZE.
            name (str, optional): Internal name of this parameter. Defaults to "".
        """
        self.state = state
        self.parameter_range = parameter_range
        self.current_value = initial_value
        self.name = name

    @classmethod
    def from_value(cls, x, name: str | None = None) -> HyperParameter:
        """Creates a Hyperparameter from a given value.

        Args:
            x (_type_): The object that we want to transform to a HyperParameter.
            name (str | None, optional): Internal name of this parameter.
                Defaults to None.

        Returns:
            HyperParameter: The HyperParameter object for the given value.
                By default this parameter is fixed and will not be optimized.
        """
        if isinstance(x, HyperParameter):
            if name is not None and x.name == "":
                x.name = name
            return x

        assert name is not None, "Name must be provided to create a HyperParameter."
        if isinstance(x, bool):
            return BooleanHyperparameter(
                initial_value=x, state=HyperParameterState.FIXED, name=name
            )
        if isinstance(x, int):
            return DiscreteHyperparameter(
                (x, x), initial_value=x, state=HyperParameterState.FIXED, name=name
            )
        if isinstance(x, float):
            return ContinuousHyperparameter(
                (x, x), initial_value=x, state=HyperParameterState.FIXED, name=name
            )
        # TODO: This is not save for any values?
        return CategoricalHyperparameter(
            categories=[x], state=HyperParameterState.FIXED, initial_value=x, name=name
        )

    @property
    def is_fixed(self):
        return self.state == HyperParameterState.FIXED

    @property
    def value(self):
        return self.current_value

    def set_value(self, new_value):
        self.current_value = new_value

    def sample_parameter_random(self):
        """Samples a random value from the given parameter range."""
        raise NotImplementedError

    def sample_parameter_grid(self, n: int) -> list:
        """Creates a point grid in the given parameter range.

        Args:
            n (int): The number of grid points.

        Returns:
            list: A list containing the grid of values.
        """
        lo = self.parameter_range[0]
        hi = self.parameter_range[1]
        if n == 1:
            return [(lo + hi) / 2]
        step = (hi - lo) / (n - 1)
        return [lo + i * step for i in range(n)]


class ContinuousHyperparameter(HyperParameter):
    def __init__(
        self,
        parameter_range: tuple | list,
        initial_value: float | None = None,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        scale: HyperParameterScale = HyperParameterScale.LINEAR,
        power: float = 2.0,  # only used for POWER scale
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


class DiscreteHyperparameter(ContinuousHyperparameter):
    def __init__(
        self,
        parameter_range: tuple | list,
        initial_value: int | None = None,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        scale: HyperParameterScale = HyperParameterScale.LINEAR,
        power: float = 2.0,  # only used for POWER scale
    ):
        assert all(
            isinstance(x, int) for x in parameter_range
        ), "Range values must be integers."
        super().__init__(
            state=state,
            parameter_range=parameter_range,
            initial_value=initial_value,
            name=name,
            scale=scale,
            power=power,
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


class CategoricalHyperparameter(HyperParameter):
    def __init__(
        self,
        categories: tuple | list,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        initial_value=None,
    ):
        if initial_value is None:
            initial_value = categories[0]

        super().__init__(
            state=state,
            parameter_range=categories,
            name=name,
            initial_value=initial_value,
        )
        # For tuning, some backends want objects to be serializable.
        # Hence we create here a string mapping:
        self._registry = {}
        for choice in self.parameter_range:
            if isinstance(choice, (int, float, str, bool, type(None))):
                key = choice
            else:
                # create string identifier
                key = self._make_key(choice)

            self._registry[key] = choice

    def _make_key(self, obj):
        if isinstance(obj, type):
            return obj.__name__
        return f"{obj.__class__.__name__}"

    @property
    def categories(self):
        return list(self._registry.keys())

    def set_value(self, new_value):
        if new_value in self._registry:
            self.current_value = self._registry[new_value]
        else:
            self.current_value = new_value

    def sample_parameter_random(self):
        return random.choice(self.parameter_range)

    def sample_parameter_grid(self, n: int) -> list:
        if isinstance(self.parameter_range, list):
            return self.parameter_range
        # TODO: For now just everything, should be subsampled?
        return list(self.parameter_range)


class BooleanHyperparameter(CategoricalHyperparameter):
    def __init__(
        self,
        initial_value: bool,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
    ):
        super().__init__(
            state=state,
            categories=[True, False],
            name=name,
            initial_value=initial_value,
        )


class ConditionalHyperparameter(CategoricalHyperparameter):

    def __init__(
        self,
        categories: tuple | list,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        initial_value=None,
    ):
        super().__init__(categories, state, name, initial_value)

        self.registry_param_map: dict[Any, list[HyperParameter]] = {}
        for i, key in enumerate(self._registry):
            self.registry_param_map[key] = categories[i].hyperparameter

    def sample_parameter_random(self):
        """Samples a random value from the given parameter range."""
        return None
