from __future__ import annotations
from typing import Any, Callable
import time

from .optimizers.optim_setups.pytorch_optims import (
    _pytorch_setup_optimizer,
    _pytorch_do_optimization_step,
)
from .optimizers.optimizers import Optimizer
from ..base import EvaluationPhase
from ...algorithms.implementation import (
    TorchImplementation,
    TensorflowImplementation,
)
from ..parameters.hyperparameter_base import HyperParameter


class TrainerState:

    def __init__(self, save_path) -> None:
        self.current_optimization_phase: OptimizationPhase
        self.iteration: int = 0

        self.total_train_loss: float = 0.0
        self.losses: dict[EvaluationPhase, dict[str, float]] = {}
        self.metrics: dict[EvaluationPhase, dict[str, float]] = {}
        for eval_phase in EvaluationPhase:
            self.losses[eval_phase] = {}
            self.metrics[eval_phase] = {}

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

    def detach_data(self):
        # TODO: make this backend independent!
        for phase_loss in self.losses.values():
            for key, loss in phase_loss.items():
                if hasattr(loss, "detach"):
                    phase_loss[key] = loss.detach().cpu().item()  # type: ignore


class OptimizationPhase:

    optimizer_setup_fn = {
        TorchImplementation: _pytorch_setup_optimizer,
        TensorflowImplementation: None,  # TODO
    }
    optimizer_step_fn = {
        TorchImplementation: _pytorch_do_optimization_step,
        TensorflowImplementation: None,  # TODO
    }

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float | HyperParameter,
        max_iterations: int | HyperParameter,
        optimizer_args: dict | None = None,
        lr_scheduler: Any = None,
        lr_scheduler_args: dict | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.optimizer_obj: Any
        self.lr = HyperParameter.from_value(lr, "Learning Rate")
        self.max_iterations = HyperParameter.from_value(max_iterations, "Max Iterations")
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = (
            lr_scheduler_args if lr_scheduler_args is not None else {}
        )

        # Find correct function for the optimizer type
        self.setup_fn: Callable
        self.step_fn: Callable
        if optimizer.backend == TorchImplementation:
            self.setup_fn = self.optimizer_setup_fn[TorchImplementation]
            self.step_fn = self.optimizer_step_fn[TorchImplementation]
        elif optimizer.backend == TensorflowImplementation:
            self.setup_fn = self.optimizer_setup_fn[TensorflowImplementation]
            self.step_fn = self.optimizer_step_fn[TensorflowImplementation]
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer.backend}")

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
