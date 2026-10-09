from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.training_controllers import TrainerState


class NaNStoppingCallback(Callback):
    """
    Callback that stops training if the loss becomes NaN.
    """

    def on_train_step_end(self, state: TrainerState):
        for phase_losses in state.losses.values():
            for key, loss_value in phase_losses.items():
                # Check if the loss value is NaN
                if loss_value is not None and (loss_value != loss_value):  # NaN check
                    state.stop_training_timer(
                        f"Training stopped due to NaN loss value in constraint '{key}'"
                    )
