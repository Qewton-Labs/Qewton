from enum import Enum
import random


class HyperParameterState(Enum):
    FIXED = 1
    OPTIMIZE = 2


class HyperParameterScale(Enum):
    LINEAR = "linear"
    LOG = "log"


class HyperParameter:
    # TODO: dtype needed?
    # TODO: What about interconnected HyperParameters? E.g. we want to try out
    # different algorithms with different hyperparameters each.
    def __init__(
        self,
        state: HyperParameterState,
        parameter_range: tuple | list,
        initial_value,
        name: str = "HyperParameter",
    ):
        self.state = state
        self.parameter_range = parameter_range
        self.current_value = initial_value
        self.name = name

    @classmethod
    def from_value(cls, x, name: str | None = None):
        if isinstance(x, HyperParameter):
            if name is not None:
                x.name = name
            return x

        assert name is not None, "Name must be provided to create a HyperParameter."
        if isinstance(x, int):
            return DiscreteHyperparameter(
                HyperParameterState.FIXED, (x, x), initial_value=x, name=name
            )
        if isinstance(x, float):
            return ContinuousHyperparameter(
                HyperParameterState.FIXED, (x, x), initial_value=x, name=name
            )
        return CategoricalHyperparameter(
            HyperParameterState.FIXED, categories=[x], initial_value=x, name=name
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
        raise NotImplementedError


class ContinuousHyperparameter(HyperParameter):
    def __init__(
        self,
        state: HyperParameterState,
        parameter_range: tuple | list,
        initial_value: float | None = None,
        name: str = "ContinuousHyperparameter",
        scale: HyperParameterScale = HyperParameterScale.LINEAR,
    ):
        assert len(parameter_range) == 2, "Range must be a tuple or list of length 2."
        if initial_value is None:
            initial_value = (parameter_range[0] + parameter_range[1]) / 2

        super().__init__(
            state=state,
            parameter_range=parameter_range,
            name=name,
            initial_value=initial_value,
        )
        self.scale = scale

    def sample_parameter_random(self):
        self.set_value(random.uniform(self.parameter_range[0], self.parameter_range[1]))


class DiscreteHyperparameter(ContinuousHyperparameter):
    def __init__(
        self,
        state: HyperParameterState,
        parameter_range: tuple | list,
        initial_value: int | None = None,
        name: str = "DiscreteHyperparameter",
        scale: HyperParameterScale = HyperParameterScale.LINEAR,
    ):
        assert len(parameter_range) == 2, "Range must be a tuple or list of length 2."
        assert all(
            isinstance(x, int) for x in parameter_range
        ), "Range values must be integers."
        super().__init__(
            state=state,
            parameter_range=parameter_range,
            initial_value=initial_value,
            name=name,
            scale=scale,
        )
        self.current_value = int(self.current_value)

    def set_value(self, new_value):
        self.current_value = int(new_value)

    def sample_parameter_random(self):
        self.set_value(random.randint(self.parameter_range[0], self.parameter_range[1]))


class CategoricalHyperparameter(HyperParameter):
    def __init__(
        self,
        state: HyperParameterState,
        categories: tuple | list,
        name: str = "CategoricalHyperparameter",
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

    def sample_parameter_random(self):
        self.set_value(random.choice(self.parameter_range[0]))
