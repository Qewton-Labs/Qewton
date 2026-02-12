from typing import Any

from .base import Tuner


class RandomSearchTuner(Tuner):

    def _get_trial_parameters(
        self, current_trials: int
    ) -> list[dict[str, dict[str, Any]]]:
        current_params = []
        for _k in range(self.process_number):
            current_params.append(self._sample_parameters_random())
        return current_params

    def _sample_parameters_random(self) -> dict[str, dict[str, Any]]:
        sampled_parameters: dict[str, dict[str, Any]] = {}
        for key, param_list in self.tunable_parameters.items():
            sampled_parameters[key] = {}
            for param in param_list:
                sampled_parameters[key][param.name] = param.sample_parameter_random()
        return sampled_parameters
