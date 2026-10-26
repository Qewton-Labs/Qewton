"""Callbacks that evaluate graphs and functions or cache data during
training."""

from typing import Any, Callable

from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.training_controllers import TrainerState, OptimizationPhase
from qewton.optim.base import EvaluationPhase
from qewton.graphs.graphs import Graph
from qewton.constraints.base import Constraint
from qewton.data.dataloaders.base import DataNode


def _record_losses(state: TrainerState, phase: EvaluationPhase, entries) -> None:
    # Writes (key, owner, loss) entries into state.losses[phase]. Raises
    # ValueError if two different owners share a key.
    owners: dict[str, Any] = {}
    for key, owner, loss in entries:
        if key in owners and owners[key] is not owner:
            raise ValueError(
                f"Two different losses are named {key!r}, so one would "
                "overwrite the other. Give them distinct names."
            )
        owners[key] = owner
        state.losses[phase][key] = loss


class GraphEvalCallback(Callback):
    """Callback that runs graphs and records the losses of constraints.

    Every `evaluation_interval` steps, all graphs are run in
    `evaluation_phase` and the loss of each constraint is stored in
    ``state.losses[evaluation_phase]`` under the constraint name. In the
    training phase the losses include the constraint weights.

    Args:
        graphs: The graphs to run.
        evaluation_phase: The phase the graphs are run in and the losses are
            recorded for.
        constraints: The constraints whose losses are recorded.
        evaluation_interval (int, optional): The number of steps between
            evaluations. Defaults to 1.
        priority (int, optional): The priority of this callback. Defaults to 0.

    Attributes:
        graphs (set of Graph): The graphs to run.
        evaluation_phase (EvaluationPhase): The phase of the evaluation.
        evaluation_interval (int): The number of steps between evaluations.
        constraints (list or set of Constraint): The constraints whose losses
            are recorded.
    """

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
        """Run the graphs and record the losses, if the step is due.

        Args:
            phase_idx: The step index within the current optimization phase.
            state: The trainer state the losses are written to.

        Raises:
            ValueError: If two different constraints have the same name.
        """
        if phase_idx % self.evaluation_interval == 0:
            for graph in self.graphs:
                graph.run(self.evaluation_phase)

            _record_losses(
                state,
                self.evaluation_phase,
                (
                    (
                        c.name,
                        c,
                        c.get_loss(
                            self.evaluation_phase == EvaluationPhase.TRAIN,
                            training_iteration=state.iteration,
                        ),
                    )
                    for c in self.constraints
                ),
            )


class FunctionEvalCallback(Callback):
    """Callback that records the values of functions as losses.

    Every `evaluation_interval` steps, each function is called with the
    step index and the trainer state, and its result is stored in
    ``state.losses[evaluation_phase]`` under the function's name.

    Args:
        functions: The functions to evaluate.
        evaluation_phase: The phase the values are recorded for.
        evaluation_interval (int, optional): The number of steps between
            evaluations. Defaults to 1.
        priority (int, optional): The priority of this callback. Defaults to 0.

    Attributes:
        functions (list of callable): The functions to evaluate.
        evaluation_phase (EvaluationPhase): The phase the values are recorded
            for.
        evaluation_interval (int): The number of steps between evaluations.
    """

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
        """Evaluate the functions and record their values, if the step is
        due.

        Args:
            phase_idx: The step index within the current optimization phase.
            state: The trainer state the values are written to.

        Raises:
            ValueError: If two different functions have the same name.
        """
        if phase_idx % self.evaluation_interval == 0:
            _record_losses(
                state,
                self.evaluation_phase,
                ((fn.__name__, fn, fn(phase_idx, state)) for fn in self.functions),
            )


class CacheDataCallback(Callback):
    """Callback that caches the data of data nodes from a given phase on.

    When `phase_to_start_cache` starts, ``cache`` is called on every data
    node, so they provide the same data from then on, e.g. for an LBFGS
    phase.

    Args:
        data_nodes: The data nodes to cache.
        phase_to_start_cache: The optimization phase at whose start the data is
            cached.
        cache_batches: The number of batches to cache.
        priority (int, optional): The priority of this callback. Defaults to 0.

    Attributes:
        data_nodes (list of DataNode): The data nodes to cache.
        phase_to_start_cache (OptimizationPhase): The phase at whose start the
            data is cached.
        cache_batches (int): The number of batches to cache.
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
        """Cache the data if `phase_to_start_cache` is starting.

        Args:
            state: The trainer state.
        """
        if state.current_optimization_phase == self.phase_to_start_cache:
            for d_node in self.data_nodes:
                d_node.cache(self.cache_batches)
