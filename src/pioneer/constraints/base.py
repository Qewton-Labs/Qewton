from enum import Enum

from ..config.backend import TensorType, Backend, DEFAULT_DL_BACKEND

from ..optim.base import EvaluationPhase
from ..optim.parameters.hyperparameter_base import HyperParameter
from ..graphs.nodes import Node, OutputPort


class ConstraintObjective(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ConstraintType(Enum):
    LOSS = "loss"
    MONITOR = "monitor"


class Constraint(Node):
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
        name="Constraint",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
        self.objective: ConstraintObjective = objective
        self.constraint_type: ConstraintType = constraint_type
        self.evaluated_in_mode = evaluated_in_mode
        super().__init__(name=name, backend=backend, **kwargs)

    @property
    def loss_port(self) -> OutputPort:
        return self.output_ports[0]

    def get_loss(self, add_weight: bool = True):
        if self.loss_port.value is None:
            raise ValueError(
                "Loss value is not computed yet. Make sure to run the forward pass of the\
                    graph before getting the loss."
            )
        return self.loss_port.value * (self.weight.value if add_weight else 1)

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.weight] + super().hyperparameters

    def set_evaluation_mode(self, new_mode: EvaluationPhase):
        self.evaluated_in_mode = new_mode
