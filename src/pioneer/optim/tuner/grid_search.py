from typing import Any
from itertools import product
import math

from pioneer.constraints.base import Constraint
from pioneer.optim.trainer.base import Trainer


from .base import Tuner
from ..hyperparameter.base import CategoricalHyperparameter


class GridSearchTuner(Tuner):

    def __init__(
        self,
        trainer: Trainer,
        tuning_constraints: list[Constraint],
        trial_number: int = 1,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        save_path: str = "tuner_results",
    ) -> None:
        super().__init__(
            trainer,
            tuning_constraints,
            trial_number,
            devices=devices,
            trials_per_device=trials_per_device,
            save_path=save_path,
        )
        self.grid_params = self._build_parameter_grid()

    def _get_trial_parameters(
        self, current_trials: int
    ) -> list[dict[str, dict[str, Any]]]:
        current_params = []
        for k in range(self.process_number):
            current_params.append(
                self._sample_parameters_grid(current_trials + k)  # type: ignore
            )
        return current_params

    def _build_parameter_grid(self):
        # First find out how many intervals and categorical parameters there are
        n_intervals = 0
        n_categorical = 0
        # TODO: How do we handle the size of the intervals? Should we even
        # keep this in mind or just sample independent of it (as currently is)?
        for param_list in self.tunable_parameters.values():
            for param in param_list:
                if isinstance(param, CategoricalHyperparameter):
                    n_categorical += len(param.parameter_range)
                else:
                    n_intervals += 1
        # Divide total trials over all parameters (wanting to use all categorical ones)
        n_per_dim = int(
            math.ceil((self.trial_number / n_categorical) ** (1 / n_intervals))
        )

        # Create the point grid
        grid_axis = []
        for param_list in self.tunable_parameters.values():
            for param in param_list:
                grid_axis.append(param.sample_parameter_grid(n_per_dim))

        grid = list(product(*grid_axis))

        # Resample the grid if the above division yielded to many points.
        # This of course will lead to some "holes" in the grid.
        size = len(grid)
        if size <= self.trial_number:
            return grid
        return [grid[i * size // self.trial_number] for i in range(self.trial_number)]

    def _sample_parameters_grid(self, index: int) -> dict[str, dict[str, Any]]:
        sampled_parameters: dict[str, dict[str, Any]] = {}
        param_counter: int = 0
        index = min(index, self.trial_number - 1)  # last ones are doubled.
        for key, param_list in self.tunable_parameters.items():
            sampled_parameters[key] = {}
            for param in param_list:
                sampled_parameters[key][param.name] = self.grid_params[index][
                    param_counter
                ]
                param_counter += 1
        return sampled_parameters
