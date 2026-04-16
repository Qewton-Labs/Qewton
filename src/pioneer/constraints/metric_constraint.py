from .base import Constraint
from ..config.configuration_base import DataConfiguration
from ..optim.parameters.categorical_hyperparameter import (
    HyperParameter,
    BooleanHyperparameter,
)
from ..optim.parameters.number_hyperparameter import ContinuousHyperparameter
from ..graphs.nodes import InputPort
from ..algorithms.implementation import DEFAULT_DL_IMPLEMENTATION
from ..algorithms.building_blocks.math import Subtract, Square, Mean, Divide


class MetricConstraint(Constraint):
    def __init__(
        self,
        input_config: DataConfiguration,
        name="MetricConstraint",
        relative: bool | BooleanHyperparameter = False,
        weight: float | ContinuousHyperparameter = 1,
        backend=DEFAULT_DL_IMPLEMENTATION,
        epsilon=1e-8,
    ):
        super().__init__(name, weight)
        self.input_config = input_config
        self.relative = HyperParameter.from_value(relative, "Relative Constraint")
        self.epsilon = epsilon  # for computation of the relative loss.

        self.input_1 = InputPort(self.input_config, self, name="input1")
        self.input_2 = InputPort(self.input_config, self, name="input2")
        self._input_ports = [self.input_1, self.input_2]

        self.backend = backend


class MSEConstraint(MetricConstraint):
    # TODO: Add different kind of norms

    def __init__(
        self,
        input_config: DataConfiguration,
        name="MSEConstraint",
        relative: bool | BooleanHyperparameter = False,
        weight: float | ContinuousHyperparameter = 1,
        backend=DEFAULT_DL_IMPLEMENTATION,
        epsilon=1e-8,
    ):
        super().__init__(
            input_config, name, relative, weight, backend=backend, epsilon=epsilon
        )
        self.subtract_operation = Subtract(backend=backend)
        self.square_operation = Square(backend=backend)
        self.mean_operation = Mean(axis=None, backend=backend)
        self.divide_operation = Divide(backend=backend)

    def compute_loss(self):
        x = self.input_1.value
        y = self.input_2.value
        residual = self.subtract_operation(input1=x, input2=y)
        residual = self.square_operation(input=residual)
        if self.relative.value:
            # TODO: Improve this relative error, do be batch wise and not
            # element wise (needs dataconfig)
            data_norm = self.square_operation(input=y)
            residual = self.divide_operation(input1=residual, input2=data_norm)
        self.loss = self.mean_operation(input=residual)
