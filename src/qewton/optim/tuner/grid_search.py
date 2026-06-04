from typing import Any

from .base import Tuner
from .tuning_callbacks.state import TuningState
from .tuning_callbacks.tuning_callback import TuningCallback
from ...optim.trainer.base_trainer import Trainer


class GridSearchTuner(Tuner):

    def __init__(
        self,
        trainer_factory: Trainer,
        tuning_objectives: list,
        trial_number: int = 1,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        track_tune_state: bool | TuningState = True,
        tuning_callbacks: list[TuningCallback] | None = None,
        save_path: str = "tuner_results",
        use_multiprocessing: bool = True,
    ) -> None:
        super().__init__(
            trainer_factory,
            tuning_objectives,
            trial_number,
            devices=devices,
            trials_per_device=trials_per_device,
            tuning_callbacks=tuning_callbacks,
            track_tune_state=track_tune_state,
            save_path=save_path,
            use_multiprocessing=use_multiprocessing,
        )
        self.grid_params = self.hp_dag.create_grid_samples(self.trial_number)
        if len(self.grid_params) < self.trial_number:
            print(f"Hyperparameters yielded only a grid of {len(self.grid_params)} \
                combinations. The number of trials is reduced accordingly.")
            self.trial_number = len(self.grid_params)

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        return self.grid_params
