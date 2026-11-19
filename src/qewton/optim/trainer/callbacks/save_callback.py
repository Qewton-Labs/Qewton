import json
import math
import os
from typing import Any

from qewton.config.saving.callables import save
from qewton.constraints.base import Constraint, ConstraintObjective
from qewton.graphs.nodes import Node
from qewton.optim.base import EvaluationPhase
from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.training_controllers import TrainerState


class SaveBestModelCallback(Callback):
    """Saves nodes with :func:`qewton.save` whenever the tracked loss improves.

    Every `check_interval` steps of an optimization phase, the losses of
    `constraints` are summed up (a constraint with objective MAXIMIZE adds
    its negative value). If this sum is smaller than the best one seen so
    far, every node in `nodes` is saved to
    ``<save_path>/<folder_name>/<node name>``, replacing the previous
    checkpoint, together with a ``checkpoint_info.json`` holding the
    iteration and the tracked value.

    Args:
        nodes: The nodes to save, e.g. the models of a graph. Their names
            must be unique, since each node is saved under its name.
        constraints: The constraints whose losses are tracked.
        check_interval (int, optional): The number of steps between checks.
            Defaults to 100.
        evaluation_phase (EvaluationPhase, optional): The phase whose losses
            are read. If None, the validation loss of a constraint is used
            if the trainer records one, else its training loss. Defaults to
            None.
        folder_name (str, optional): The folder inside the trainer's save
            path that the nodes are saved to. Defaults to "best_model".
        priority (int, optional): Priority of this callback. Defaults to 0.
    """

    def __init__(
        self,
        nodes: Node | list[Node],
        constraints: Constraint | list[Constraint],
        check_interval: int = 100,
        evaluation_phase: EvaluationPhase | None = None,
        folder_name: str = "best_model",
        priority=0,
    ) -> None:
        super().__init__(priority)
        self.nodes = nodes if isinstance(nodes, (list, tuple)) else [nodes]
        self.constraints = (
            constraints if isinstance(constraints, (list, tuple)) else [constraints]
        )
        if len(self.nodes) == 0:
            raise ValueError("SaveBestModelCallback needs at least one node to save.")
        if len(self.constraints) == 0:
            raise ValueError("SaveBestModelCallback needs at least one constraint.")
        if check_interval < 1:
            raise ValueError(f"check_interval must be positive, got {check_interval}.")
        self.check_interval = check_interval
        self.evaluation_phase = evaluation_phase
        self.folder_name = folder_name

        self.best_value = math.inf
        self.best_iteration = -1
        self._phase_step = 0

    @property
    def saves_data(self) -> bool:
        return True

    def on_training_start(self, state: TrainerState):
        names = [node.name for node in self.nodes]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"The nodes to save must have unique names, but {sorted(duplicates)} "
                "appear more than once."
            )
        self.best_value = math.inf
        self.best_iteration = -1

    def on_optimization_phases_start(self, state: TrainerState):
        self._phase_step = 0

    def on_train_step_end(self, state: TrainerState):
        step = self._phase_step
        self._phase_step += 1
        if step % self.check_interval != 0:
            return

        value = self._tracked_value(state)
        if value is None or not value < self.best_value:  # also skips NaN
            return

        self.best_value = value
        self.best_iteration = state.iteration
        self._save(state)

    def _tracked_value(self, state: TrainerState) -> float | None:
        # Returns None if a tracked loss was not computed yet.
        total = 0.0
        for constraint in self.constraints:
            loss = self._lookup_loss(state, constraint)
            if loss is None:
                return None
            value = _to_float(loss)
            if constraint.objective == ConstraintObjective.MAXIMIZE:
                value = -value
            total += value
        return total

    def _lookup_loss(self, state: TrainerState, constraint: Constraint) -> Any:
        # The name is read here and not in __init__, since the trainer may
        # rename constraints to make their names unique.
        if self.evaluation_phase is not None:
            phases = [self.evaluation_phase]
        else:
            phases = [EvaluationPhase.VALIDATION, EvaluationPhase.TRAIN]
        for phase in phases:
            if constraint.name in state.losses[phase]:
                return state.losses[phase][constraint.name]
        raise KeyError(
            f"No loss of constraint {constraint.name!r} is recorded in "
            f"{[p.name for p in phases]}. Is it part of the trainer's graphs?"
        )

    def _save(self, state: TrainerState):
        folder = os.path.join(state.save_path, self.folder_name)
        os.makedirs(folder, exist_ok=True)
        for node in self.nodes:
            save(node, os.path.join(folder, node.name), replace=True)
        info = {
            "iteration": self.best_iteration,
            "value": self.best_value,
            "constraints": [c.name for c in self.constraints],
        }
        with open(
            os.path.join(folder, "checkpoint_info.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(info, f, indent=4)


def _to_float(loss: Any) -> float:
    if hasattr(loss, "detach"):
        loss = loss.detach().cpu()
    if hasattr(loss, "item"):
        return float(loss.item())
    return float(loss)
