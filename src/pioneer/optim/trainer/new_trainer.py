from .training_controllers import TrainingPhase, TrainerState
from .trainable_parameters import _TrainableParameterBase
from ..hyperparameter.base import HyperParameter


class TrainerBase:

    def __init__(
        self,
        training_phases: TrainingPhase | list[TrainingPhase],
        trainable_parameters: _TrainableParameterBase,
        hyperparameters: set[HyperParameter],
        device="cpu",
        save_path: str = "train_results",
        # TODO: Callbacks/Logging
    ) -> None:
        if isinstance(training_phases, TrainingPhase):
            training_phases = [training_phases]
        self.training_phases = training_phases
        self.trainable_parameters = trainable_parameters

        self.hyperparameters = hyperparameters
        for stage in self.training_phases:
            self.hyperparameters.union(stage.get_hyperparameter())

        self.device = device
        self.train_state = TrainerState(self.training_phases[0])
        self.save_path = save_path

    def run(self):
        self.on_training_start()

        # Go through all training stages:
        for stage in self.training_phases:
            self.train_state.current_stage = stage
            self.train_state.stop_stage = False

            self.on_training_stage_start()

            # Run iterations inside the stage
            for local_idx in range(stage.max_iterations.current_value):
                self.training_step(local_idx)

                self.train_state.iteration += 1  # global iteration counter

                # Check if training stage should be stopped
                if self.train_state.stop_stage:
                    break

            self.on_training_stage_end()

            # Check if training should be stopped completely
            if self.train_state.stop_training:
                break

        self.on_training_end()

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_training_stage_start(self):
        pass

    def training_step(self, idx: int):
        pass

    def on_training_stage_end(self):
        pass

    def run_validation(self):
        pass


class CallbackBase:
    """
    Base class for callbacks. Override the hooks you need.
    """

    def on_train_start(self, trainer: TrainerBase, state: TrainerState):
        pass

    def on_stage_start(self, trainer: TrainerBase, state: TrainerState):
        pass

    def on_iteration_end(self, trainer: TrainerBase, state: TrainerState):
        pass

    def on_stage_end(self, trainer: TrainerBase, state: TrainerState):
        pass

    def on_train_end(self, trainer: TrainerBase, state: TrainerState):
        pass
