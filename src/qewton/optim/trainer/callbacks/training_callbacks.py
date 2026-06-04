from typing import Any, Callable

from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.training_controllers import TrainerState
from qewton.optim.base import EvaluationPhase
from qewton.graphs.graphs import Graph
from qewton.constraints.base import Constraint


class GraphEvalCallback(Callback):

    def __init__(
        self,
        graphs: set[Graph],
        evaluation_phase: EvaluationPhase,
        constraints: list[Constraint] | set[Constraint],
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
                state.losses[self.evaluation_phase][constraint.name] = value


class FunctionEvalCallback(Callback):

    def __init__(
        self,
        functions: list[Callable[[int, TrainerState], Any]],
        evaluation_phase: EvaluationPhase,
        evaluation_interval=1,
        priority=0,
    ) -> None:
        super().__init__(priority)
        self.functions = functions
        self.evaluation_phase = evaluation_phase
        self.evaluation_interval = evaluation_interval

    def training_step(self, phase_idx: int, state: TrainerState):
        if phase_idx % self.evaluation_interval == 0:
            for fn in self.functions:
                loss = fn(phase_idx, state)
                state.losses[self.evaluation_phase][fn.__name__] = loss
