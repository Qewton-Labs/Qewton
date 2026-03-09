from typing import Literal

from ..optim.base import EvaluationMode
from ..optim.hyperparameter.base import HyperParameter
from ..optim.hyperparameter.number_hyperparameter import ContinuousHyperparameter
from ..nodes.base import Node, Port

# class Constraint:
#     def is_fulfilled_by(self, algorithm, data=None):
#         raise NotImplementedError("Method not implemented in base class.")


# class ResourceConstraint(Constraint):
#     """
#     technical constraints, e.g. memory, compute time etc...
#     """

#     def __init__(self):
#         pass


class Constraint(Node):
    """
    problem constraints, e.g. data, PDE, symmetries etc...

    TODO: Can this be the main Constraint-class
    (what inputs do the ResourceConstraint need, should they live outside the graph?)
    """

    def __init__(
        self,
        name="Constraint",
        weight: float | ContinuousHyperparameter = 1.0,
        objective: Literal["minimize", "maximize"] = "minimize",
    ):
        if objective not in ("minimize", "maximize"):
            raise ValueError(
                f"objective must be 'minimize' or 'maximize', got '{objective}'"
            )
        super().__init__(name=name)
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
        self.objective: str = objective
        self.loss = 0.0
        self.mode = EvaluationMode.ALWAYS

    @property
    def output_ports(self) -> dict[str, Port]:
        return {}

    def get_loss(self, add_weight: bool = True):
        return self.loss * (self.weight.value if add_weight else 1)

    def reset(self):
        self.loss = 0.0

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.weight]

    def set_mode(self, new_mode: EvaluationMode):
        self.mode = new_mode
