from enum import Enum

from ..config.backend import TensorType, Backend

from ..optim.base import EvaluationPhase
from ..optim.parameters.hyperparameter_base import HyperParameter
from ..graphs.nodes import Node, Port


class ConstraintObjective(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintType(Enum):
    LOSS = "loss"
    METRIC = "metric"


class Constraint:
    """
    TODO: These could be become graph nodes as well, but this may be not
    so nice for end user if they want to just implement there own constraint.

    problem constraints, e.g. data, PDE, symmetries etc...
    """

    def __init__(
        self,
        weight: float | HyperParameter = 1.0,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        constraint_type: ConstraintType = ConstraintType.LOSS,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
    ):
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
        self.objective: ConstraintObjective = objective
        self.constraint_type: ConstraintType = constraint_type
        self.loss_port: Port
        self.evaluated_in_mode = evaluated_in_mode

    def get_loss(self, add_weight: bool = True):
        return self.loss_port.value * (self.weight.value if add_weight else 1)

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.weight]

    def set_evaluation_mode(self, new_mode: EvaluationPhase):
        self.evaluated_in_mode = new_mode
