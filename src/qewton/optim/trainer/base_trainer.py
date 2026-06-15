"""
Base trainer implementation for optimization workflows.

This module defines the Trainer class, which controls the execution of one
or multiple optimization phases, manages callbacks, handles hyperparameters,
and tracks training state.

The main training is controlled via callbacks.
"""

from typing import Any

from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.callbacks.progressbar_callback import ProgressBarCallback
from qewton.optim.trainer.training_controllers import OptimizationPhase, TrainerState
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase
from qewton.optim.parameters.hyperparameter_base import HyperParameter


###############################
# TODO: Add parallelization?
#
# TODO: Add stop and restart training
#
# TODO: Maybe split callbacks cleanly into training/validation/etc.
###############################
class Trainer:
    """Coordinator for optimization training runs.

    A Trainer manages a sequence of optimization phases, configures callbacks,
    applies hyperparameters, and advances the training loop. The training
    lifecycle is exposed through hooks that invoke registered callbacks at
    key events.

    Args:
        optimization_phases (OptimizationPhase | list[OptimizationPhase]):
            One or more optimization phases defining the training workflow.
        callbacks (Callback | list[Callback]): One or more callback objects
            used to receive training events.
        hyperparameters (set[HyperParameter]): A set of global hyperparameters
            used during training.
        device (str): Target device string for training, e.g. "cpu" or "cuda".
        save_path (str): Directory or file path where training results are stored.
        progress_bar (ProgressBarCallback | None, optional): Optional progress bar
            callback instance. If omitted, a default ProgressBarCallback is added.
        enable_logging (bool): Whether training logs should be recorded.
        log_interval (int): Interval in iterations for logging output.
    """

    def __init__(
        self,
        optimization_phases: OptimizationPhase | list[OptimizationPhase],
        callbacks: Callback | list[Callback],
        hyperparameters: set[HyperParameter],
        device="cpu",
        save_path: str = "train_results",
        progress_bar: ProgressBarCallback | None = None,
        enable_logging=True,
        log_interval=100,
    ) -> None:
        if isinstance(optimization_phases, OptimizationPhase):
            optimization_phases = [optimization_phases]
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        if progress_bar is None:
            progress_bar = ProgressBarCallback()
        self.optimization_phases = optimization_phases
        self.trainable_parameters: _TrainableParameterBase
        callbacks.append(progress_bar)  # add progress bar by default
        self.callbacks = sorted(callbacks, key=lambda cb: cb.priority)

        self.hyperparameters = hyperparameters
        for stage in self.optimization_phases:
            self.hyperparameters |= stage.get_hyperparameter()

        self.device = device
        self.train_state = TrainerState(
            save_path, enable_logging=enable_logging, log_interval=log_interval
        )

    def set_trainable_parameters(self, parameters: _TrainableParameterBase):
        """Attach the trainable parameter container used by optimization phases.

        Args:
            parameters (_TrainableParameterBase): Container holding the
                parameters to update during training.
        """
        self.trainable_parameters = parameters

    def run(self, show_progress=True):
        """Execute the full training loop across all configured phases.

        Each optimization phase is initialized, stepped through for its
        configured number of iterations, and cleaned up. Registered callbacks
        are invoked at lifecycle events.

        Args:
            show_progress (bool): If False, remove progress bar callbacks before
                starting training.
        """
        if not show_progress:
            for cb in self.callbacks:
                if isinstance(cb, ProgressBarCallback):
                    self.callbacks.remove(cb)

        self.on_training_start()

        # Go through all training stages:
        for phase in self.optimization_phases:
            self.train_state.current_optimization_phase = phase
            self.train_state.stop_stage = False
            phase.setup_optimizer(self)  # build the optimizer

            self.on_optimization_phases_start()

            # Run iterations inside the stage
            for local_idx in range(phase.max_iterations.current_value):
                phase.do_optimization_step(
                    self.training_step, local_idx, self.train_state
                )
                self.on_training_step_end()

                # Check if training stage should be stopped
                if self.train_state.stop_stage:
                    break

            self.on_optimization_phases_end()

            # Check if training should be stopped completely
            if self.train_state.stop_training:
                break

        self.on_training_end()

    def on_training_start(self):
        """Notify callbacks that training is starting and prepare the training state."""
        self.train_state.check_file_path(self.callbacks)
        self.train_state.start_training_timer()
        for cb in self.callbacks:
            cb.on_training_start(self.train_state)

    def on_optimization_phases_start(self):
        """Notify callbacks that a new optimization phase is beginning."""
        for cb in self.callbacks:
            cb.on_optimization_phases_start(self.train_state)

    def training_step(self, idx: int):
        """Execute a single training iteration and notify callbacks.

        Args:
            idx (int): Local index of the current iteration within the current phase.
        """
        self.train_state.clear_data_dict()

        for cb in self.callbacks:
            cb.training_step(idx, self.train_state)

    def on_training_step_end(self):
        """Finalize the current training iteration and update training state."""
        self.train_state.log_step()

        self.train_state.iteration += 1  # global iteration counter
        for cb in self.callbacks:
            cb.on_train_step_end(self.train_state)

    def on_optimization_phases_end(self):
        """Notify callbacks that the current optimization phase has finished."""
        for cb in self.callbacks:
            cb.on_optimization_phases_end(self.train_state)

    def on_training_end(self):
        """Notify callbacks that training has completed and stop timers."""
        for cb in self.callbacks:
            cb.on_training_end(self.train_state)

        if not self.train_state.stop_training:
            self.train_state.stop_training_timer("Training finished")

    def set_hyperparameter(self, param_dict: dict[str, Any]):
        """Update hyperparameter values from a dictionary.

        Args:
            param_dict (dict[str, Any]): Mapping of hyperparameter names
                to new values.
        """
        for param in self.hyperparameters:
            if param.name in param_dict:
                param.set_value(param_dict[param.name])

    def set_device(self, device):
        """Set the target device for training."""
        self.device = device

    def check_tuning_constraints_exist(self, constraints):
        """Stub for checking tuning constraints compatibility."""
        return False

    def populate_state_dict(self):
        """Collect relevant loss and metric names into the state dict.

        This method is intended to prepare the training state with the expected
        output names before training begins.
        """

    def cleanup(self):
        """Perform cleanup tasks after the last optimization phase completes."""
        self.optimization_phases[
            -1
        ].cleanup()  # clean up after the last optimization phase
