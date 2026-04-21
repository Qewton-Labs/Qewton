from .base_callback import Callback
from ..training_controllers import TrainerState
from ...base import EvaluationPhase
from ....graphs.graphs import Graph
from ....constraints.base import Constraint, ConstraintType


class GraphEvalCallback(Callback):

    def __init__(
        self,
        graphs: set[Graph],
        evaluation_phase: EvaluationPhase,
        constraints: list[Constraint],
        evaluation_interval=1,
        priority=0,
    ) -> None:
        super().__init__(priority)
        self.graphs = graphs
        self.evaluation_phase = evaluation_phase
        self.evaluation_interval = evaluation_interval
        self.constraints = constraints

    def training_step(self, phase_idx: int, state: TrainerState):
        if phase_idx % self.evaluation_interval == 0:
            for graph in self.graphs:
                graph.run(self.evaluation_phase)

            for constraint in self.constraints:
                value = constraint.get_loss()

                if constraint.constraint_type == ConstraintType.LOSS:
                    state.losses[self.evaluation_phase][constraint.name] = value
                else:  # ConstraintType.METRIC
                    state.metrics[self.evaluation_phase][constraint.name] = value
