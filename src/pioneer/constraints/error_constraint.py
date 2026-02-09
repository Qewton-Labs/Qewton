from enum import Enum
from typing import Any

from ..config.configuration_base import DataConfiguration
from ..optim.base import EvaluationMode
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
        evaluation_mode: EvaluationMode,
        name="MSEConstraint",
        weight: float | ContinuousHyperparameter = 1,
    ):
        super().__init__(evaluation_mode, name, weight)
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
        diff = x - y
        if hasattr(diff, "mean"):
            self.loss = (diff ** 2).mean()
            return {}
        if "tensorflow" in str(type(diff)):
            import tensorflow as tf # TODO: Not so nice, better to move this into a child class
            self.loss = tf.reduce_mean(tf.square(diff))
            return {}
        raise ValueError("The MSE can not be computed for this input")
