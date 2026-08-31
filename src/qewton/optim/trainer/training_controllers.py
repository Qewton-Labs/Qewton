from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import time
import os

from qewton.optim.trainer.optimizers.optimizers import Optimizer
from qewton.optim.trainer.optimizers.schedulers import LR_Scheduler
from qewton.optim.base import EvaluationPhase

from qewton.optim.parameters.hyperparameter_base import HyperParameter


@dataclass
class LogEntry:
    """A record of training metrics captured at a logging interval.

    Attributes:
        iteration (int): Global iteration index when the log entry was created.
        timestamp (float): Wall-clock time of the log entry.
        losses (dict[EvaluationPhase, dict[str, float | None]]): Snapshot of
            loss values for each evaluation phase.
    """

    iteration: int
    timestamp: float
    losses: dict[EvaluationPhase, dict[str, float | None]]


class TrainerState:
    """Shared state container for training progress and logging.

    TrainerState is passed to callbacks and optimization phases so they can
    inspect and update the current training status.

    Args:
        save_path (str): Directory path used for saving callback output.
        enable_logging (bool, optional): Whether to record log history.
            Defaults to True.
        log_interval (int, optional): Number of iterations between log entries.
            Defaults to 100.
    """

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
        """Start the training duration timer."""
        self.start_time = time.time()

    def stop_training_timer(self, reason: str = ""):
        """Stop the timer and optionally record a termination reason.

        Args:
            reason (str): Reason for stopping training.
        """
        self.total_train_time += time.time() - self.start_time
        if self.termination_reason == "":
            self.termination_reason = reason
        self.stop_training = True

    @property
    def stop_training(self):
        """bool: Whether training should stop completely."""
        return self._stop_training

    @stop_training.setter
    def stop_training(self, stop):
        """Set the global training stop flag.

        Args:
            stop (bool): If True, training will stop and the current stage
                will also be marked to stop.
        """
        self._stop_training = stop
        if stop:
            self.stop_stage = stop

    def detach_data(self, data_dict) -> dict[EvaluationPhase, dict[str, float | None]]:
        """Convert backend tensor losses to plain Python scalars.

        Args:
            data_dict: Raw loss dictionary potentially containing tensor objects.

        Returns:
            dict[EvaluationPhase, dict[str, float | None]]: Detached loss values.
        """
        new_dict = {}
        for phase_name, phase_loss in data_dict.items():
            new_dict[phase_name] = {}
            for key, loss in phase_loss.items():
                new_dict[phase_name][key] = loss
                if hasattr(loss, "detach"):
                    new_dict[phase_name][key] = loss.detach().cpu().item()  # type: ignore
        return new_dict

    def clear_data_dict(self):
        """Reset stored loss values for all evaluation phases."""
        for phase in EvaluationPhase:
            for k in self.losses[phase]:
                self.losses[phase][k] = None

    def check_file_path(self, callbacks):
        """Validate the save path and create a unique directory when needed.

        Args:
            callbacks (list[Callback]): Registered callbacks that may save data.
        """
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
        """Append the current loss snapshot to history if logging is enabled."""
        if not self.enable_logging:
            return
        if self.iteration % self.log_interval != 0:
            return

        loss_dict = self.detach_data(self.losses)

        self.history.append(
            LogEntry(iteration=self.iteration, timestamp=time.time(), losses=loss_dict)
        )


class OptimizationPhase:
    """Encapsulates one phase of optimization, including optimizer and
    learning rate scheduling.

    OptimizationPhase manages optimizer setup, per-step execution, cleanup, and
    hyperparameter extraction for a single training stage.

    Args:
        optimizer (Optimizer): Optimizer wrapper defining the backend and params.
        lr (float | HyperParameter): Learning rate value or hyperparameter.
        max_iterations (int | HyperParameter): Iteration count or hyperparameter.
        optimizer_args (dict | None, optional): Additional optimizer kwargs.
            Defaults to None.
        lr_scheduler (Any, optional): Optional learning rate scheduler class or
            callable. Defaults to None.
        lr_scheduler_args (dict | None, optional): Arguments for the scheduler.
            Defaults to None.

    Raises:
        ValueError: If the optimizer backend is unsupported.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float | HyperParameter,
        max_iterations: int | HyperParameter,
        optimizer_args: dict | None = None,
        lr_scheduler: LR_Scheduler | None = None,
    ) -> None:
        self.optimizer: Optimizer = optimizer
        self.optimizer_obj: Any
        self.lr = HyperParameter.from_value(lr, "Learning Rate")
        self.max_iterations = HyperParameter.from_value(max_iterations, "Max Iterations")
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler = lr_scheduler

        # Find correct function for the optimizer type
        self.setup_fn: Callable = optimizer.backend.optim.setup_optimizer
        self.step_fn: Callable = optimizer.backend.optim.do_optimization_step
        self.cleanup_fn: Callable = optimizer.backend.optim._cleanup

    def get_hyperparameter(self) -> set[HyperParameter]:
        """Return the set of HyperParameter instances referenced by this phase.

        Returns:
            set[HyperParameter]: Hyperparameters used by this optimization phase.
        """
        hp_set = set[HyperParameter]()
        self._scan_for_hyperparameter(vars(self).values(), hp_set)
        if self.lr_scheduler is not None:
            hp_set.update(self.lr_scheduler.hyper_parameters())
        return hp_set

    def _scan_for_hyperparameter(self, value_collection, hp_set: set[HyperParameter]):
        """Recursively scan a collection for HyperParameter instances.

        Args:
            value_collection: Iterable of values to inspect.
            hp_set (set[HyperParameter]): Set to populate with discovered params.
        """
        for value in value_collection:
            if isinstance(value, HyperParameter):
                hp_set.add(value)
            elif isinstance(value, (list, tuple, set)):
                self._scan_for_hyperparameter(value, hp_set)
            elif isinstance(value, dict):
                self._scan_for_hyperparameter(value.values(), hp_set)

    def setup_optimizer(self, trainer):
        """Build and initialize the backend optimizer object for this phase.

        Args:
            trainer (Trainer): Trainer instance used to configure the optimizer.
        """
        self.optimizer_obj = self.setup_fn(self, trainer)
        if self.lr_scheduler is not None:
            self.lr_scheduler_obj = self.lr_scheduler.build_scheduler(self.optimizer_obj)

    def do_optimization_step(
        self, eval_function: Callable, step_idx: int, train_state: TrainerState
    ):
        """Execute one optimization step using the backend-specific step function.

        Args:
            eval_function (Callable): Function invoked to compute losses.
            step_idx (int): Index of the current step in this phase.
            train_state (TrainerState): Shared training state.
        """
        self.step_fn(self, eval_function, step_idx, train_state)
        # TODO: Maybe only lr step after one epoch?
        if self.lr_scheduler is not None:
            self.lr_scheduler_obj.step()

    def cleanup(self):
        """Perform backend-specific cleanup after the optimization phase ends."""
        self.cleanup_fn()
