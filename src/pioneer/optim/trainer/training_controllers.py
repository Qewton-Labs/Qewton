from dataclasses import dataclass, field
from typing import Any

from ..parameters.hyperparameter_base import HyperParameter


class TrainingPhase:

    def __init__(
        self,
        optimizer: Any,
        lr: float | HyperParameter,
        max_iterations: int | HyperParameter,
        optimizer_args: dict | None = None,
        lr_scheduler: Any = None,
        lr_scheduler_args: dict | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.lr = HyperParameter.from_value(lr, "Learning Rate")
        self.max_iterations = HyperParameter.from_value(max_iterations, "Max Iterations")
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = (
            lr_scheduler_args if lr_scheduler_args is not None else {}
        )

    def get_hyperparameter(self) -> set[HyperParameter]:
        hp_set = set[HyperParameter]()
        self._scan_for_hyperparameter(vars(self).values(), hp_set)
        return hp_set

    def _scan_for_hyperparameter(self, value_collection, hp_set: set[HyperParameter]):
        for value in value_collection:
            if isinstance(value, HyperParameter):
                hp_set.add(value)
            elif isinstance(value, (list, tuple, dict)):
                self._scan_for_hyperparameter(value, hp_set)


@dataclass
class TrainerState:
    current_stage: TrainingPhase
    iteration: int = 0
    total_train_loss: float | None = None
    train_losses: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_losses: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    stop_stage: bool = False
    _stop_training: bool = False

    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, stop):
        self._stop_training = stop
        if stop:
            self.stop_stage = stop
