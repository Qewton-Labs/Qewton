from .state import TuningState
from ...trainer.callbacks.base_callback import Callback


class TuningCallback(Callback):

    def __init__(self, priority=0) -> None:
        super().__init__(priority)
        self.tune_state: TuningState | None = None

    def set_tune_state(self, tune_state: TuningState):
        self.tune_state = tune_state
