from ..optim.base import EvaluationMode
from ..optim.hyperparameter.base import HyperParameter, ContinuousHyperparameter
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

    TODO: Maybe the loss should be moved to a special runtime constraint
    such that it gets not problematic in parallel processes?
    """

    def __init__(
        self,
        name="Constraint",
        weight: float | ContinuousHyperparameter = 1.0,
    ):
        super().__init__(name=name)
        self.weight: HyperParameter = HyperParameter.from_value(weight, "Weight")
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
