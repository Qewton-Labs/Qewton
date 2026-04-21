from .base_callback import Callback
from ..training_controllers import TrainerState


class LogCallback(Callback):
    # TODO: Decide on good log-format
    def __init__(self, log_interval=100, priority=-100):
        super().__init__(priority)
        self.log_interval = log_interval

    def on_training_start(self, state: TrainerState):
        pass

    def on_train_step_end(self, state: TrainerState):
        if state.iteration % self.log_interval == 0:
            pass
