from tqdm import tqdm

from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.training_controllers import TrainerState


class ProgressBarCallback(Callback):
    """Creates a progress bar for showing the training progress."""

    def __init__(self, priority=-100) -> None:
        super().__init__(priority)
        self.progress_bar: tqdm
        self.phase_counter = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("progress_bar", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.progress_bar: tqdm

    def on_optimization_phases_start(self, state: TrainerState):
        self.phase_counter += 1
        self.progress_bar = tqdm(
            total=state.current_optimization_phase.max_iterations.value,
            desc="Optimization Phase " + str(self.phase_counter),
        )

    def on_train_step_end(self, state: TrainerState):
        self.progress_bar.update(1)

        self.progress_bar.set_postfix({"loss": state.total_train_loss})

    def on_optimization_phases_end(self, state: TrainerState):
        self.progress_bar.close()
