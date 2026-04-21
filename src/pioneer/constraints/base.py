from enum import Enum

from ..optim.base import EvaluationPhase
from ..optim.parameters.hyperparameter_base import HyperParameter
from ..graphs.nodes import Node


class ConstraintObjective(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintType(Enum):
    LOSS = "loss"
    METRIC = "metric"


class Constraint(Node):
    """
    TODO: These could be become graph nodes as well, but this may be not
    so nice for end user if they want to just implement there own constraint.

    problem constraints, e.g. data, PDE, symmetries etc...
    """

    def __init__(
        self,
        name="Constraint",
        weight: float | HyperParameter = 1.0,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        constraint_type: ConstraintType = ConstraintType.LOSS,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
    ):
        super().__init__(name=name)
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
        self.objective: ConstraintObjective = objective
        self.constraint_type: ConstraintType = constraint_type
        self.loss = 0.0
        self.evaluated_in_mode = evaluated_in_mode
        self._output_ports = []

    def run(self) -> None:
        if self.evaluated_in_mode not in (self.mode, EvaluationPhase.ALWAYS):
            return
        self.check_constraint()

    def check_constraint(self):
        pass

    def get_loss(self, add_weight: bool = True):
        return self.loss * (self.weight.value if add_weight else 1)

    def reset(self):
        self.loss = 0.0

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.weight]

    def set_evaluation_mode(self, new_mode: EvaluationPhase):
        self.evaluated_in_mode = new_mode
