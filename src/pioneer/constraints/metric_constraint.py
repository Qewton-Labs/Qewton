from enum import Enum
from typing import Any

from ..config.configuration_base import DataConfiguration
from ..optim.hyperparameter.base import (
    HyperParameter,
    ContinuousHyperparameter,
    BooleanHyperparameter,
)
from ..nodes.base import Port
from .base import Constraint


class MetricConstraint(Constraint):
    class InputKeys(str, Enum):  # type: ignore[override]
        INPUT1 = "input_1"
        INPUT2 = "input_2"

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

    @property
    def input_ports(self) -> dict[str, Port]:
        return {
            self.InputKeys.INPUT1: Port(self.input_config, self, "port_1", True),
            self.InputKeys.INPUT2: Port(self.input_config, self, "port_2", True),
        }


class MSEConstraint(MetricConstraint):
    # TODO: Add relative constraints and also different kind of norms, etc.

    def __init__(
        self,
        input_config: DataConfiguration,
        name="MSEConstraint",
        relative: bool | BooleanHyperparameter = False,
        weight: float | ContinuousHyperparameter = 1,
        epsilon=1e-8,
    ):
        super().__init__(input_config, name, relative, weight, epsilon=epsilon)

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Inputs can not be None")
        x = inputs[self.InputKeys.INPUT1]
        y = inputs[self.InputKeys.INPUT2]
        return self._compute_mean(x, y)

    def _compute_mean(self, x, y):
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
        return {}


class ResidualConstraint(MSEConstraint):
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
            input_config=input_config_1,
            name=name,
            weight=weight,
        )
        self.input_config_2 = input_config_2
        self.residual_fn = residual_fn

    @property
    def input_ports(self) -> dict[str, Port]:
        return {
            self.InputKeys.INPUT1: Port(self.input_config, self, "port_1", True),
            self.InputKeys.INPUT2: Port(self.input_config_2, self, "port_2", True),
        }

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Inputs can not be None")
        x = inputs[self.InputKeys.INPUT1]
        y = inputs[self.InputKeys.INPUT2]
        residual = self.residual_fn(x, y)
        self.loss = (residual**2).mean()
        return {}
