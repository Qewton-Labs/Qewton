import csv
import os
import math
from multiprocessing import Pool
from copy import deepcopy
from typing import Any
from enum import Enum
from itertools import product


from ..trainer.base import Trainer
from ...constraints.base import Constraint
from ..hyperparameter.base import HyperParameter, CategoricalHyperparameter


class TuningStrategy(Enum):
    RANDOM = "random"
    GRID = "grid"


# Helper methods for parallel execution
# TODO: Maybe move into class with @static (could break when two tuners are run?)
class _WorkerState:
    trainer = None


def _init_worker(trainer):
    _WorkerState.trainer = trainer


def worker_eval(params):
    local_trainer: Trainer = deepcopy(_WorkerState.trainer)  # type: ignore
    local_trainer.set_hyperparameter(params)
    local_trainer.run(show_progress=False)
    results = local_trainer.get_tuning_results()
    return [{"params": params}, results]


# TODO: Implement stuff like early stopping? And also stop when some given
# constrain is reached
#
# TODO: We want to save the best performing results?
#
# TODO: Add constraints for memory usage and speed
#
# TODO: Add different tuning strategies (also make the currently implemented one more
#       general, e.g. conditional parameters are currently not handled)
#
# TODO: Move data to different devices (i think best if this happens in the trainer!)
#
# TODO: Enable to restart tuning from a given point
#
# TODO: Is passing the data via dictionaries the best way?
class Tuner:

    def __init__(
        self,
        trainer: Trainer,
        tuning_constraints: list[Constraint],
        trial_number: int = 10,
        process_number: int = 1,
        tuning_strategy: TuningStrategy = TuningStrategy.RANDOM,
        save_path: str = "tuner_results",
    ) -> None:
        self.trainer_object = trainer
        self.process_number = process_number
        self.trial_number = trial_number
        self.tuning_strategy = tuning_strategy
        self.tuning_constraints = tuning_constraints
        self.trainer_object.set_tuning_constraints(self.tuning_constraints)

        # Build saving path
        self.save_path = save_path
        # TODO: does this work for all systems? Windows and Mac?
        csv_path = self.save_path + "/" + self.trainer_object.save_path
        counter = 0
        csv_path_extension = csv_path + str(counter) + ".csv"
        while os.path.exists(csv_path_extension):
            counter += 1
            csv_path_extension = csv_path + str(counter) + ".csv"
        self.csv_path = csv_path_extension
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        # Find what parameters can be tuned
        self.tunable_parameters: dict[str, list[HyperParameter]] = {}
        self._get_tuneable_parameters()

    def _get_tuneable_parameters(self):
        hyperparameter_dict = self.trainer_object.get_hyperparameter()
        for node_name, param_list in hyperparameter_dict.items():
            for param in param_list:
                if not param.is_fixed:
                    if node_name in self.tunable_parameters:
                        self.tunable_parameters[node_name].append(param)
                    else:
                        self.tunable_parameters[node_name] = [param]

        if len(self.tunable_parameters) == 0:
            raise ValueError("Can not tune a problem without tunable parameters.")

    def run(self):
        self.trainer_object.reset()  # clean up trainer to allow copying

        trials = math.ceil(self.trial_number / self.process_number)

        if self.tuning_strategy == TuningStrategy.GRID:
            grid_params = self._build_parameter_grid()

        print("--- Start Tuning ---")
        for i in range(trials):
            current_n = self.process_number * i
            print(f"Running trials {current_n} to {self.process_number + current_n}")
            current_params = []

            if self.tuning_strategy == TuningStrategy.RANDOM:
                for _k in range(self.process_number):
                    current_params.append(self._sample_parameters_random())
            elif self.tuning_strategy == TuningStrategy.GRID:
                for k in range(self.process_number):
                    current_params.append(
                        self._sample_parameters_grid(grid_params, current_n + k)  # type: ignore
                    )

            results = self._run_generation(current_params)
            print("Saving current results")
            self._write_to_csv(results)

    def _run_generation(self, params):
        # TODO: Can maybe be made a bit smarter at the end to not start additional tries
        # when we already reached the final number of runs
        with Pool(
            processes=self.process_number,
            initializer=_init_worker,
            initargs=(self.trainer_object,),
        ) as pool:
            results = list(pool.imap_unordered(worker_eval, params))
        return results

    def _write_to_csv(self, results):
        flat_results = [self._flatten_result_data(r) for r in results]
        write_header = not os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())

            if write_header:
                writer.writeheader()

            writer.writerows(flat_results)

    def _flatten_result_data(self, results):
        # TODO: this is highly depending on the output of worker_eval and really unsafe...
        # Parameter are first:
        flat_dict = {}
        for _, all_params in results[0].items():
            for node_name, param_dict in all_params.items():
                for param_name, param_value in param_dict.items():
                    flat_dict[node_name + "_" + param_name] = param_value

        for key, value in results[1].items():
            flat_dict[key] = value

        return flat_dict

    def _sample_parameters_random(self) -> dict[str, dict[str, Any]]:
        sampled_parameters: dict[str, dict[str, Any]] = {}
        for key, param_list in self.tunable_parameters.items():
            sampled_parameters[key] = {}
            for param in param_list:
                sampled_parameters[key][param.name] = param.sample_parameter_random()
        return sampled_parameters

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

        # Resample the grid if the above division yielded to many points
        size = len(grid)
        if size <= self.trial_number:
            return grid
        return [grid[i * size // self.trial_number] for i in range(self.trial_number)]

    def _sample_parameters_grid(self, grid, index: int) -> dict[str, dict[str, Any]]:
        sampled_parameters: dict[str, dict[str, Any]] = {}
        param_counter: int = 0
        index = min(index, self.trial_number - 1)
        for key, param_list in self.tunable_parameters.items():
            sampled_parameters[key] = {}
            for param in param_list:
                sampled_parameters[key][param.name] = grid[index][param_counter]
                param_counter += 1
        return sampled_parameters
