from qewton.optim.trainer.training_controllers import TrainerState


class Callback:
    """
    Base class for callbacks. Override the hooks you need.
    """

    def __init__(self, priority=0) -> None:
        self.priority = priority

    @property
    def saves_data(self) -> bool:
        return False

    def on_training_start(self, state: TrainerState):
        pass

    def on_optimization_phases_start(self, state: TrainerState):
        pass

    def training_step(self, phase_idx: int, state: TrainerState):
        pass

    def on_train_step_end(self, state: TrainerState):
        pass

    def on_optimization_phases_end(self, state: TrainerState):
        pass

    def on_training_end(self, state: TrainerState):
        pass
