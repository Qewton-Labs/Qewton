from enum import Enum

from ..optim.base import EvaluationPhase
from ..optim.hyperparameter.base import HyperParameter
from ..nodes.base import Node


class ConstraintObjective(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Constraint(Node):
    """
    problem constraints, e.g. data, PDE, symmetries etc...
    """

    def __init__(
        self,
        name="Constraint",
        weight: float | HyperParameter = 1.0,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
    ):
        super().__init__(name=name)
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
        self.objective: ConstraintObjective = objective
        self.loss = 0.0
        self.evaluated_in_mode = evaluated_in_mode
        self._output_ports = []

    def run(self) -> None:
        if self.evaluated_in_mode not in (self.mode, EvaluationPhase.ALWAYS):
            return
        self.compute_loss()

    def compute_loss(self):
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
