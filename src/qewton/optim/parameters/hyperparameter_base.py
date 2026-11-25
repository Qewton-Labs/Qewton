from __future__ import annotations
import operator

from qewton.optim.parameters.helpers import HyperParameterState, HyperParameterCondition
from qewton.config.saving.saving import Serializable


class HyperParameter(Serializable):
    """Abstract HyperParameter class that represents tunable parameters.

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

    def __init__(
        self,
        parameter_range: tuple | list,
        initial_value,
        state: HyperParameterState = HyperParameterState.OPTIMIZE,
        name: str = "",
        active_when: None | HyperParameterCondition = None,
        default_grid: int | list = 0,
    ):
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
    def is_fixed(self) -> bool:
        """
        Returns:
            bool: Whether this parameter is left alone while tuning: it is
                fixed, or inferred from data.
        """
        return self.state != HyperParameterState.OPTIMIZE

    @classmethod
    def size_from(cls, x, name: str) -> HyperParameter:
        """A size that fixes how a node is built: the given one (see
        :meth:`from_value`), or, for None, an open one that is inferred from
        the connected data (see :meth:`inferred`).

        Args:
            x: The size, a HyperParameter, or None.
            name: Internal name of this parameter.

        Returns:
            HyperParameter: The size as a HyperParameter.
        """
        return cls.inferred(name) if x is None else cls.from_value(x, name)

    @property
    def is_open(self) -> bool:
        """Whether this parameter waits to be inferred from the connected
        data: state INFERRED and no value yet."""
        return self.state == HyperParameterState.INFERRED and self.current_value is None

    @classmethod
    def inferred(cls, name: str) -> HyperParameter:
        """A whole-number parameter that stays open (value None) until it is
        inferred from the connected data (:meth:`infer`).

        Args:
            name: Internal name of this parameter.

        Returns:
            HyperParameter: The open parameter, in state INFERRED.
        """
        from .number_hyperparameter import (  # pylint: disable=import-outside-toplevel
            DiscreteHyperparameter,
        )

        parameter = DiscreteHyperparameter(
            (1, 1), 1, state=HyperParameterState.INFERRED, name=name
        )
        parameter.current_value = None
        return parameter

    def infer(self, value) -> bool:
        """Learn the value from the connected data: an open parameter in
        state INFERRED takes it; any other is compared with it. Called by
        the dims bound to this parameter once their size is known.

        Args:
            value: The value the data needs.

        Returns:
            bool: Whether the value changed, i.e. an open parameter was
                inferred.

        Raises:
            DataConfigMismatchError: If the parameter has another value, e.g.
                inferred at another place of a graph.
        """
        from qewton.config.errors import DataConfigMismatchError

        current = self.value
        if current == value:
            return False
        if current is None:
            if self.state != HyperParameterState.INFERRED:
                # e.g. an operation of parameters that are still open
                return False
            self.current_value = value
            return True
        raise DataConfigMismatchError(
            f"HyperParameter {self.name!r} is {current}, but the data at another "
            f"place needs {value}."
        )

    def reopen(self) -> None:
        """Forget a value inferred from the connected data, so that it is
        inferred anew (:meth:`infer`); a parameter not in state INFERRED is
        left as it is. Called when the connections it was inferred from may
        be gone, see :meth:`~qewton.graphs.graphs.Graph.remove`."""
        if self.state == HyperParameterState.INFERRED:
            self.current_value = None

    @property
    def value(self):
        """
        Returns:
            _type_: The current value of this HyperParameter.
        """
        return self.current_value

    @property
    def tuning_grid(self) -> list:
        """Returns a grid of values for tuning this parameter."""
        if isinstance(self.default_grid, list):
            return self.default_grid
        return self.sample_parameter_grid(self.default_grid)

    def is_active(self, config: dict = {}) -> bool:
        """Checks whether this HyperParameter is active under the
        given configuration. This is the case if there is some connection to
        HyperParameters. The concrete values of the Parameters are provided in
        the config.
        """
        if config is None:
            config = {}
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

    # Arithmetic: an operation on HyperParameters is a HyperParameter whose
    # value follows theirs. A number operand is a constant of the operation,
    # not a parameter of its own.

    def _arithmetic(self, other, name: str, reflected: bool = False) -> HyperParameterOperation:
        if isinstance(other, HyperParameter):
            first, second = (other, self) if reflected else (self, other)
            return HyperParameterOperation(Arithmetic(name), dependencies=[first, second])
        return HyperParameterOperation(
            Arithmetic(name, constant=other, reflected=reflected), dependencies=[self]
        )

    def __add__(self, other):
        return self._arithmetic(other, "add")

    def __radd__(self, other):
        return self._arithmetic(other, "add", reflected=True)

    def __sub__(self, other):
        return self._arithmetic(other, "sub")

    def __rsub__(self, other):
        return self._arithmetic(other, "sub", reflected=True)

    def __mul__(self, other):
        return self._arithmetic(other, "mul")

    def __rmul__(self, other):
        return self._arithmetic(other, "mul", reflected=True)

    def __truediv__(self, other):
        return self._arithmetic(other, "truediv")

    def __rtruediv__(self, other):
        return self._arithmetic(other, "truediv", reflected=True)

    @property
    def dependencies(self) -> set[HyperParameter]:
        return set() if self.condition is None else self.condition.deps


class Arithmetic(Serializable):
    """An arithmetic operation (`operator.add`, `sub`, `mul`, `truediv`) of
    the values of HyperParameters, with a constant number as one operand if
    one is given. Saved by its name.

    Args:
        name (str): The name of the operation in the `operator` module.
        constant (float | int | None, optional): A constant operand.
        reflected (bool, optional): Whether the constant is the first operand.
    """

    def __init__(self, name: str, constant=None, reflected: bool = False):
        self.name = name
        self.constant = constant
        self.reflected = reflected

    def __call__(self, *values):
        operation = getattr(operator, self.name)
        if self.constant is None:
            return operation(*values)
        (value,) = values
        return operation(self.constant, value) if self.reflected else operation(value, self.constant)


class HyperParameterOperation(HyperParameter):
    """A HyperParameter that is defined as an operation on other HyperParameters.

    Args:
        operation (callable): A function that takes the values of the dependencies
            and returns the value of this HyperParameter.
        dependencies (list[HyperParameter]): The HyperParameters that this
            HyperParameter depends on. The HyperParameters need to be provided
            in the order that the operation expects them!
    """

    def __init__(self, operation, dependencies: list[HyperParameter], name: str = ""):
        self.operation = operation
        self.hp_dependencies = dependencies
        super().__init__(
            parameter_range=(None, None),
            initial_value=None,
            state=HyperParameterState.FIXED,
            name=name,
        )

    @property
    def dependencies(self) -> set[HyperParameter]:
        return set(self.hp_dependencies)

    def set_value(self, new_value):
        raise RuntimeError(
            "Cannot set value of HyperParameterOperation. The value is determined by the operation on the dependencies."
        )

    @property
    def is_open(self) -> bool:
        """Whether one of the dependencies waits to be inferred."""
        return any(hp.is_open for hp in self.hp_dependencies)

    @property
    def value(self):
        """The operation on the values of the dependencies; None while one
        of them is open."""
        values = [hp.value for hp in self.hp_dependencies]
        return None if None in values else self.operation(*values)

    def sample_parameter_random(self):
        values = [hp.sample_parameter_random() for hp in self.hp_dependencies]
        return self.operation(*values)

    def sample_from_config(self, config):
        values = [config[hp.name] for hp in self.hp_dependencies]
        return self.operation(*values)

    def sample_from_unit(self, x: float):
        values = [hp.sample_from_unit(x) for hp in self.hp_dependencies]
        return self.operation(*values)

    def sample_parameter_grid(self, n: int) -> list:
        value_grid = [hp.sample_parameter_grid(n) for hp in self.hp_dependencies]
        return self._eval_grid(value_grid, n)

    @property
    def tuning_grid(self) -> list:
        raise RuntimeError(
            "Cannot get tuning grid of HyperParameterOperation. The grid is determined by the operation on the dependencies."
        )

    def _eval_grid(self, value_grid, n):
        grid = []
        for i in range(n):
            values = [value_grid[j][i] for j in range(len(self.hp_dependencies))]
            grid.append(self.operation(*values))
        return grid
