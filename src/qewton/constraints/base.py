from enum import Enum

from qewton.config.backend import TensorType, Backend, DEFAULT_DL_BACKEND

from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.nodes import Node, OutputPort


class ConstraintObjective(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Constraint(Node):
    """A constraint represents a condition/loss/metric that should be fulfilled or
    monitored. A constraint can be optimized during training or just be tracked
    for validation/testing. Constraints can be used to implement problem specific
    regularization, e.g. physics informed losses, symmetries, data constraints etc.

    A constraint usually has one output port that represents the value of
    the constraint, which can be used to compute the loss for optimization
    or just for monitoring. The constraint can have any number of input ports,
    which can be used to feed in the necessary data for computing the constraint.

    Args:
        weight (float | HyperParameter, optional): An additional weight for
            this constraint. Defaults to 1.0.
        objective (ConstraintObjective, optional): Wether to minimize of maximize
            this constraint. Defaults to ConstraintObjective.MINIMIZE.
        evaluated_in_mode (EvaluationPhase, optional): When this constraint should
            be evaluated. Defaults to EvaluationPhase.ALWAYS.
        name (str, optional): A name for this constraint. Defaults to "Constraint".
        backend (type[Backend[TensorType]], optional): What backend to use.
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        weight: float | HyperParameter = 1.0,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        name="Constraint",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
        self.objective: ConstraintObjective = objective

        self.evaluated_in_mode = evaluated_in_mode
        super().__init__(name=name, backend=backend, **kwargs)

    @property
    def loss_port(self) -> OutputPort:
        """The output port in which the current loss value will be written.

        Returns:
            OutputPort: The output port in which the current loss value
                will be written.
        """
        return self.output_ports[0]

    def get_loss(self, add_weight: bool = True):
        """Return the current loss value of this constraint,
        multiplied by the weight if add_weight is True.

        Args:
            add_weight (bool, optional): If to add the weight. Defaults to True.

        Raises:
            ValueError: If the loss value is not computed yet.

        Returns:
            Number: The current loss value of this constraint.
        """
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
        """Set the evaluation mode of this constraint. This determines in which
        phase this constraint will be evaluated.

        Args:
            new_mode (EvaluationPhase): The new evaluation mode for this constraint.
        """
        self.evaluated_in_mode = new_mode

    def run(self) -> None:
        if self.evaluated_in_mode not in (self.mode, EvaluationPhase.ALWAYS):
            self.loss_port.set_value(None)
        return super().run()
