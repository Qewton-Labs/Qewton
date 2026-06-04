from typing import Any

from qewton.optim.tuner.base import Tuner


class RandomSearchTuner(Tuner):

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        return self.hp_dag.create_random_samples(self.trial_number)
