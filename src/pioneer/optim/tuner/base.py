import csv
import os
import math

from multiprocessing import Pool
from copy import deepcopy
from typing import Any

from ..trainer.base import Trainer
from ..hyperparameter.base import HyperParameter

# Helper methods for parallel execution
# TODO: Maybe move into class with @static (could break when two tuners are run?)
_WORKER_OBJ = None


def _init_worker(obj):
    global _WORKER_OBJ
    _WORKER_OBJ = obj


def worker_eval(params):
    local_obj: Trainer = deepcopy(_WORKER_OBJ)  # type: ignore
    local_obj.set_hyperparameter(params)
    return [{"params": params}, local_obj.run()]


# TODO: Implement stuff like early stopping? And also stop when some given
# constrain is reached
#
# TODO: We want to save the best performing results?
#
# TODO: Add constraints for memory usage and speed
#
# TODO: Add different tuning strategies
#
# TODO: Move data to different devices (i think best if this happens in the trainer!)
#
# TODO: Enable to restart tuning from a given point
class Tuner:

    def __init__(
        self,
        trainer: Trainer,
        trial_number: int = 10,
        process_number: int = 1,
        save_path: str = "tuner_results",
    ) -> None:
        self.trainer_object = trainer
        self.process_number = process_number
        self.trial_number = trial_number

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

    def _sample_parameters(self) -> dict[str, dict[str, Any]]:
        sampled_parameters: dict[str, dict[str, Any]] = {}
        for key, param_list in self.tunable_parameters.items():
            sampled_parameters[key] = {}
            for param in param_list:
                sampled_parameters[key][param.name] = param.sample_parameter_random()  # type: ignore
        return sampled_parameters

    def run(self):
        self.trainer_object.reset()  # clean up trainer to allow copying

        trials = math.ceil(self.trial_number / self.process_number)
        print("--- Start Tuning ---")
        for i in range(trials):
            current_n = self.process_number * i
            print(f"Running trials {current_n} to {self.process_number + current_n}")
            current_params = []
            # TODO: this only makes sense for random sampling
            for _k in range(self.process_number):
                current_params.append(self._sample_parameters())

            results = self._run_generation(current_params)
            print("Saving current results")
            self._write_to_csv(results)

    def _run_generation(self, params):
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

        for pipeline_key, pipeline_results in results[1].items():
            for key, value in pipeline_results.items():
                flat_dict[pipeline_key + "_" + key] = value

        return flat_dict
