import csv
import os
import math
from multiprocessing import Pool
from copy import deepcopy
from typing import Any


from ..trainer.base import Trainer
from ...constraints.base import Constraint
from ..hyperparameter.base import HyperParameter


# Helper methods for parallel execution
# TODO: Maybe move into class with @static (could break when two tuners are run?)
class _WorkerState:
    trainer = None


def _init_worker(trainer):
    _WorkerState.trainer = trainer


def worker_eval(jobs):
    local_trainer: Trainer = deepcopy(_WorkerState.trainer)  # type: ignore
    params, device = jobs
    if isinstance(device, str):
        local_trainer.set_device(device)
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
# TODO: Enable to restart tuning from a given point
#
# TODO: Is passing the data via dictionaries the best way?
#
# TODO: In "devices" also allow for something like "auto" or "all" to automatically
#       use all available CPUs/GPUs? Currently passing in an int just copies
#       the trainer to the device given by the trainer!
class Tuner:

    def __init__(
        self,
        trainer: Trainer,
        tuning_constraints: list[Constraint],
        trial_number: int = 10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        save_path: str = "tuner_results",
    ) -> None:
        self.trainer_object = trainer
        self.trial_number = trial_number
        self.tuning_constraints = tuning_constraints
        self.trainer_object.set_tuning_constraints(self.tuning_constraints)

        if isinstance(devices, str):
            devices = [devices]
        self.devices = [device for device in devices for _ in range(trials_per_device)]
        self.process_number = len(self.devices)

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

        print("--- Start Tuning ---")
        for i in range(trials):
            current_n = self.process_number * i
            print(f"Running trials {current_n} to {self.process_number + current_n}")
            current_params = self._get_trial_parameters(current_n)
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
            jobs = list(zip(params, self.devices))
            results = list(pool.imap_unordered(worker_eval, jobs))
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

    def _get_trial_parameters(
        self, current_trials: int
    ) -> list[dict[str, dict[str, Any]]]:
        raise NotImplementedError(
            "The base Tuner does not implement a search strategy, \
                use one of the child classes."
        )
