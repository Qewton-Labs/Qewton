from typing import Any

from .callbacks.base_callback import Callback
from .callbacks.progressbar_callback import ProgressBarCallback
from .training_controllers import OptimizationPhase, TrainerState
from ..parameters.trainable_parameters import _TrainableParameterBase
from ..parameters.hyperparameter_base import HyperParameter


###############################
# TODO: Add parallelization?
#
# TODO: Add stop and restart training
#
# TODO: Maybe split callbacks cleanly into training/validation/etc.
###############################
class Trainer:

    def __init__(
        self,
        optimization_phases: OptimizationPhase | list[OptimizationPhase],
        callbacks: Callback | list[Callback],
        hyperparameters: set[HyperParameter],
        device="cpu",
        save_path: str = "train_results",
        progress_bar: ProgressBarCallback = ProgressBarCallback(),
    ) -> None:
        if isinstance(optimization_phases, OptimizationPhase):
            optimization_phases = [optimization_phases]
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]

        self.optimization_phases = optimization_phases
        self.trainable_parameters: _TrainableParameterBase
        callbacks.append(progress_bar)  # add progress bar by default
        self.callbacks = sorted(callbacks, key=lambda cb: cb.priority)

        self.hyperparameters = hyperparameters
        for stage in self.optimization_phases:
            self.hyperparameters |= stage.get_hyperparameter()

        self.device = device
        self.train_state = TrainerState(save_path)

    def set_trainable_parameters(self, parameters: _TrainableParameterBase):
        self.trainable_parameters = parameters

    def run(self, show_progress=True):
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
                # self.training_step(local_idx)
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
        self.train_state.start_training_timer()
        for cb in self.callbacks:
            cb.on_training_start(self.train_state)

    def on_optimization_phases_start(self):
        for cb in self.callbacks:
            cb.on_optimization_phases_start(self.train_state)

    def training_step(self, idx: int):
        for cb in self.callbacks:
            cb.training_step(idx, self.train_state)

    def on_training_step_end(self):
        self.train_state.iteration += 1  # global iteration counter
        for cb in self.callbacks:
            cb.on_train_step_end(self.train_state)

    def on_optimization_phases_end(self):
        for cb in self.callbacks:
            cb.on_optimization_phases_end(self.train_state)

    def on_training_end(self):
        for cb in self.callbacks:
            cb.on_training_end(self.train_state)

        if not self.train_state.stop_training:
            self.train_state.stop_training_timer()

    def set_hyperparameter(self, param_dict: dict[str, Any]):
        for param in self.hyperparameters:
            if param.name in param_dict:
                param.set_value(param_dict[param.name])

    def set_device(self, device):
        self.device = device

    def evaluate_tuning_constraints(self):
        pass

    def populate_state_dict(self):
        """Collect all relevant loss and metric names into the state dict, to
        know at the start of training which values are to be expected."""
