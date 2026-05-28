from typing import Any, Callable

from .base import Tuner
from ...optim.trainer.base_trainer import Trainer


class GridSearchTuner(Tuner):

    def __init__(
        self,
        trainer_factory: Trainer,
        trial_number: int = 1,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        save_path: str = "tuner_results",
    ) -> None:
        super().__init__(
            trainer_factory,
            trial_number,
            devices=devices,
            trials_per_device=trials_per_device,
            save_path=save_path,
        )
        self.grid_params = self.hp_dag.create_grid_samples(self.trial_number)
        if len(self.grid_params) < self.trial_number:
            print(
                f"Hyperparameters yielded only a grid of {len(self.grid_params)} \
                combinations. The number of trials is reduced accordingly."
            )
            self.trial_number = len(self.grid_params)

    # def _get_trial_parameters(
    #     self, current_trial: int
    # ) -> list[dict[str, dict[str, Any]]]:
    #     return self.grid_params[current_trial : current_trial + self.process_number]

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        return self.grid_params
