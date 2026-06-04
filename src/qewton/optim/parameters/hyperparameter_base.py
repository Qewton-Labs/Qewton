from __future__ import annotations
import operator

from .helpers import HyperParameterState, HyperParameterCondition


class HyperParameter:
    """Abstract HyperParameter class that represents tunable parameters."""

    def __init__(
        self,
        parameter_range: tuple | list,
        initial_value,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        active_when: None | HyperParameterCondition = None,
        default_grid: int | list = 0,
    ):
        """
        Args:
            parameter_range (tuple | list): Allowed range of values for this parameter.
            initial_value (_type_): Initial value for this parameter.
            state (HyperParameterState, optional): If this value should be fixed or be
                optimized. Defaults to HyperParameterState.OPTIMIZE.
            name (str, optional): Internal name of this parameter. Defaults to "".
            active_when (HyperParameterCondition, optional): A condition that specifies
                when this Hyperparameter should be active. The condition depends on the
                values of other Hyperparameters. Accordingly this parameters is only
                sampled/set when the condition is fulfilled. See both
                HyperParameterCondition and HyperParameterDAG.
        """
        self.state = state
        self.parameter_range = parameter_range
        self.current_value = initial_value
        self.name = name
        self.condition = active_when
        self.default_grid = default_grid

    @classmethod
    def from_value(cls, x, name: str | None = None) -> HyperParameter:
        """Creates a Hyperparameter from a given value.

        Args:
            x (_type_): The object that we want to transform to a HyperParameter.
            name (str | None, optional): Internal name of this parameter.
                Defaults to None. If the input already is an HyperParameter,
                the name will only be changed if the current name is empty string
                and the value is not None.

        Returns:
            HyperParameter: The HyperParameter object for the given value.
                By default this parameter is fixed and will not be optimized.
        """
        if isinstance(x, HyperParameter):
            if name is not None and x.name == "":
                x.name = name
            return x

        assert name is not None, "Name must be provided to create a HyperParameter."

        from .categorical_hyperparameter import (  # pylint: disable=import-outside-toplevel
            BooleanHyperparameter,
            CategoricalHyperparameter,
        )
        from .number_hyperparameter import (  # pylint: disable=import-outside-toplevel
            DiscreteHyperparameter,
            ContinuousHyperparameter,
        )

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
        return CategoricalHyperparameter(
            categories=[x], state=HyperParameterState.FIXED, initial_value=x, name=name
        )

    @property
    def is_fixed(self):
        return self.state == HyperParameterState.FIXED

    @property
    def value(self):
        return self.current_value

    @property
    def tuning_grid(self) -> list:
        if isinstance(self.default_grid, list):
            return self.default_grid
        return self.sample_parameter_grid(self.default_grid)

    def is_active(self, config=None):
        return True if self.condition is None else self.condition.evaluate(config)

    def set_value(self, new_value):
        self.current_value = new_value

    def sample_parameter_random(self):
        """Samples a random value from the given parameter range."""
        raise NotImplementedError

    def sample_from_unit(self, x: float):
        """Maps a value x from [0,1) to a value in the given parameter range."""
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

    def __hash__(self):
        return id(self)

    def _get_value(self, config):
        if config is None:
            return self.current_value
        return config[self.name]

    def _binary_condition(self, other, op):
        if isinstance(other, HyperParameter):
            return HyperParameterCondition(
                lambda config: op(
                    self._get_value(config),
                    other._get_value(config),  # pylint: disable=protected-access
                ),
                deps={self, other},
            )
        return HyperParameterCondition(
            lambda config: op(self._get_value(config), other),
            deps={self},
        )

    def __eq__(self, other):  # type: ignore[override]
        return self._binary_condition(other, operator.eq)

    def __lt__(self, other):
        return self._binary_condition(other, operator.lt)

    def __gt__(self, other):
        return self._binary_condition(other, operator.gt)

    def __le__(self, other):
        return self._binary_condition(other, operator.le)

    def __ge__(self, other):
        return self._binary_condition(other, operator.ge)

    def __ne__(self, other):  # type: ignore[override]
        return self._binary_condition(other, operator.ne)

    @property
    def dependencies(self) -> set[HyperParameter]:
        return set() if self.condition is None else self.condition.deps
