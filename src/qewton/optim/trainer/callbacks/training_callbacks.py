from typing import Any, Callable

from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.training_controllers import TrainerState, OptimizationPhase
from qewton.optim.base import EvaluationPhase
from qewton.graphs.graphs import Graph
from qewton.constraints.base import Constraint
from qewton.data.dataloaders.base import DataNode


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


class CacheDataCallback(Callback):
    """Calls .cache on all given DataNodes, such that always the same
    data is provided. E.g., can be useful for fixing when switching to LBFGS.

    Args:
    data_nodes (list[DataNode]): The nodes that provide data and should be cached.
    phase_to_start_cache (OptimizationPhase): The phase were the caching should start.
    cache_batches (int, optional): If multiple batches of data should be cached.
        Defaults to 1.
    priority (int, optional): The priority of this Callback. Defaults to 0.
    """

    def __init__(
        self,
        data_nodes: list[DataNode],
        phase_to_start_cache: OptimizationPhase,
        cache_batches: int = 1,
        priority=0,
    ) -> None:
        self.data_nodes = data_nodes
        self.phase_to_start_cache = phase_to_start_cache
        self.cache_batches = cache_batches
        super().__init__(priority)

    def on_optimization_phases_start(self, state: TrainerState):
        if state.current_optimization_phase == self.phase_to_start_cache:
            for d_node in self.data_nodes:
                d_node.cache(self.cache_batches)
