from pioneer.config.backend import Backend, TensorType
from pioneer.constraints.base import ConstraintObjective, ConstraintType
from pioneer.optim.base import EvaluationPhase
from pioneer.optim.parameters.hyperparameter_base import HyperParameter

from .base import Constraint


class PINNConstraint(Constraint):

    def __init__(
        self,
        name="Constraint",
        weight: float | HyperParameter = 1,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        constraint_type: ConstraintType = ConstraintType.LOSS,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        backend: type[Backend[TensorType]] | None = None,
    ):
        super().__init__(
            name, weight, objective, constraint_type, evaluated_in_mode, backend
        )
