from enum import Enum
from typing import Any

from ..config.configuration_base import DataConfiguration
from ..optim.hyperparameter.base import ContinuousHyperparameter
from ..nodes.base import Port
from .base import Constraint


class MSEConstraint(Constraint):

    class InputKeys(str, Enum):  # type: ignore[override]
        INPUT1 = "input_1"
        INPUT2 = "input_2"

    def __init__(
        self,
        input_config: DataConfiguration,
        name="MSEConstraint",
        weight: float | ContinuousHyperparameter = 1,
    ):
        super().__init__(name, weight)
        self.input_config = input_config
        self.loss = 0.0

    @property
    def input_ports(self) -> dict[str, Port]:
        return {
            self.InputKeys.INPUT1: Port(self.input_config, self, "port_1", True),
            self.InputKeys.INPUT2: Port(self.input_config, self, "port_2", True),
        }

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Inputs can not be None")
        x = inputs[self.InputKeys.INPUT1]
        y = inputs[self.InputKeys.INPUT2]
        residual = x - y
        return self._compute_mean(residual)

    def _compute_mean(self, residual):
        if hasattr(residual, "mean"):
            self.loss = (residual**2).mean()
            return {}
        if "tensorflow" in str(type(residual)):
            # TODO: Not so nice, better to move this into a child class
            import tensorflow as tf

            self.loss = tf.reduce_mean(tf.square(residual))
            return {}
        raise ValueError("The MSE can not be computed for this input")


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
        return self._compute_mean(residual)
