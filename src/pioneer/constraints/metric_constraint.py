from ..config.configuration_base import DataConfiguration
from ..optim.parameters.categorical_hyperparameter import (
    HyperParameter,
    BooleanHyperparameter,
)
from ..optim.parameters.number_hyperparameter import ContinuousHyperparameter
from ..nodes.base import InputPort
from .base import Constraint


class MetricConstraint(Constraint):
    def __init__(
        self,
        input_config: DataConfiguration,
        name="MetricConstraint",
        relative: bool | BooleanHyperparameter = False,
        weight: float | ContinuousHyperparameter = 1,
        epsilon=1e-8,
    ):
        super().__init__(name, weight)
        self.input_config = input_config
        self.relative = HyperParameter.from_value(relative, "Relative Constraint")
        self.epsilon = epsilon  # for computation of the relative loss.

        self.input_port_1 = InputPort(self.input_config, self, name="Input 1")
        self.input_port_2 = InputPort(self.input_config, self, name="Input 2")
        self._input_ports = [self.input_port_1, self.input_port_2]


class MSEConstraint(MetricConstraint):
    # TODO: Add different kind of norms

    def __init__(
        self,
        input_config: DataConfiguration,
        name="MSEConstraint",
        relative: bool | BooleanHyperparameter = False,
        weight: float | ContinuousHyperparameter = 1,
        epsilon=1e-8,
    ):
        super().__init__(input_config, name, relative, weight, epsilon=epsilon)

    def compute_loss(self):
        x = self.input_port_1.value
        y = self.input_port_2.value
        residual = x - y
        if hasattr(residual, "mean"):
            if self.relative.value:
                # TODO: Assumes first axis is batch axis. Would like
                # to get this from the dataconfiguration but problematic,
                # since most models just have [..., feature_axis]?
                axes = tuple(range(1, len(self.input_config)))
                residual = (residual**2).mean(axes)
                data_norm = (y**2).mean(axes)
                self.loss = (residual / (data_norm + self.epsilon)).mean()
            else:
                self.loss = (residual**2).mean()
        elif "tensorflow" in str(type(residual)):
            # TODO: Not so nice, better to move this into a child class
            import tensorflow as tf

            self.loss = tf.reduce_mean(tf.square(residual))
        else:
            raise ValueError("The MSE can not be computed for this input")


class ResidualConstraint(Constraint):
    # TODO: Just some rough first version to see if PINNs would work
    def __init__(
        self,
        input_config_1: DataConfiguration,
        input_config_2: DataConfiguration,
        residual_fn,
        name="ResidualConstraint",
        weight: float | ContinuousHyperparameter = 1,
    ):
        super().__init__(
            name=name,
            weight=weight,
        )

        self.input_port_1 = InputPort(input_config_1, self, name="Input 1")
        self.input_port_2 = InputPort(input_config_2, self, name="Input 2")
        self._input_ports = [self.input_port_1, self.input_port_2]
        self.residual_fn = residual_fn

    def compute_loss(self):
        x = self.input_port_1.value
        y = self.input_port_2.value
        residual = self.residual_fn(x, y)
        self.loss = (residual**2).mean()
