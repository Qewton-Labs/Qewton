import random

from .helpers import HyperParameterState, HyperParameterCondition
from .base import HyperParameter


class CategoricalHyperparameter(HyperParameter):
    def __init__(
        self,
        categories: tuple | list,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        initial_value=None,
        active_when: None | HyperParameterCondition = None,
    ):
        if initial_value is None:
            initial_value = categories[0]

        super().__init__(
            state=state,
            parameter_range=categories,
            name=name,
            initial_value=initial_value,
            active_when=active_when,
            default_grid=list(categories),
        )
        self.current_value = self.current_value
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
        values = list(self.parameter_range)
        reps = (n + len(values) - 1) // len(values)
        seq = values * reps
        return seq[:n]

    def sample_from_unit(self, x: float):
        return self.parameter_range[int(x * len(self.parameter_range))]


class BooleanHyperparameter(CategoricalHyperparameter):
    def __init__(
        self,
        initial_value: bool,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        active_when: None | HyperParameterCondition = None,
    ):
        super().__init__(
            state=state,
            categories=[True, False],
            name=name,
            initial_value=initial_value,
            active_when=active_when,
        )
