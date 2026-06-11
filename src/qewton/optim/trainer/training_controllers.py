from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import time
import os

from qewton.optim.trainer.optimizers.optimizers import Optimizer
from qewton.optim.base import EvaluationPhase

from qewton.optim.parameters.hyperparameter_base import HyperParameter


@dataclass
class LogEntry:
    iteration: int
    timestamp: float
    losses: dict[EvaluationPhase, dict[str, float | None]]


class TrainerState:

    def __init__(
        self, save_path, enable_logging: bool = True, log_interval: int = 100
    ) -> None:
        self.current_optimization_phase: OptimizationPhase
        self.iteration: int = 0

        self.total_train_loss: float = 0.0
        self.losses: dict[EvaluationPhase, dict[str, float | None]] = {}
        for eval_phase in EvaluationPhase:
            self.losses[eval_phase] = {}

        self.history: list[LogEntry] = []
        self.enable_logging = enable_logging
        self.log_interval = log_interval

        self.stop_stage: bool = False
        self._stop_training: bool = False

        self.save_path = save_path

        self.total_train_time = 0.0
        self.start_time = 0.0
        self.termination_reason = ""

    def start_training_timer(self):
        self.start_time = time.time()

    def stop_training_timer(self, reason: str = ""):
        self.total_train_time += time.time() - self.start_time
        if self.termination_reason == "":
            self.termination_reason = reason
        self.stop_training = True

    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, stop):
        self._stop_training = stop
        if stop:
            self.stop_stage = stop

    def detach_data(self, data_dict) -> dict[EvaluationPhase, dict[str, float | None]]:
        # TODO: make this backend independent!
        new_dict = {}
        for phase_name, phase_loss in data_dict.items():
            new_dict[phase_name] = {}
            for key, loss in phase_loss.items():
                new_dict[phase_name][key] = loss
                if hasattr(loss, "detach"):
                    new_dict[phase_name][key] = loss.detach().cpu().item()  # type: ignore
        return new_dict

    def clear_data_dict(self):
        for phase in EvaluationPhase:
            for k in self.losses[phase]:
                self.losses[phase][k] = None

    def check_file_path(self, callbacks):
        if not any(cb.saves_data for cb in callbacks):
            self.save_path = ""
            return

        file_path = self.save_path
        counter = 0
        while os.path.exists(file_path):
            counter += 1
            file_path = f"{self.save_path}_{counter}"

        os.makedirs(file_path, exist_ok=True)
        self.save_path = file_path

    def log_step(self):
        if not self.enable_logging:
            return
        if self.iteration % self.log_interval != 0:
            return

        loss_dict = self.detach_data(self.losses)

        self.history.append(
            LogEntry(iteration=self.iteration, timestamp=time.time(), losses=loss_dict)
        )


class OptimizationPhase:

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float | HyperParameter,
        max_iterations: int | HyperParameter,
        optimizer_args: dict | None = None,
        lr_scheduler: Any = None,
        lr_scheduler_args: dict | None = None,
    ) -> None:
        self.optimizer: Optimizer = optimizer
        self.optimizer_obj: Any
        self.lr = HyperParameter.from_value(lr, "Learning Rate")
        self.max_iterations = HyperParameter.from_value(max_iterations, "Max Iterations")
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = (
            lr_scheduler_args if lr_scheduler_args is not None else {}
        )

        # Find correct function for the optimizer type
        self.setup_fn: Callable = optimizer.backend.optim.setup_optimizer
        self.step_fn: Callable = optimizer.backend.optim.do_optimization_step
        self.cleanup_fn: Callable = optimizer.backend.optim._cleanup

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

    def setup_optimizer(self, trainer):
        # TODO: Add learning rate scheduler
        self.optimizer_obj = self.setup_fn(self, trainer)

    def do_optimization_step(
        self, eval_function: Callable, step_idx: int, train_state: TrainerState
    ):
        self.step_fn(self, eval_function, step_idx, train_state)

    def cleanup(self):
        self.cleanup_fn()
