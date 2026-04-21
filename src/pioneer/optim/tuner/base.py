import csv
import os
import math
import json
import platform
import multiprocessing as mp
from typing import Any, Callable
from enum import Enum
from datetime import datetime
import psutil

from ...constraints.base import Constraint
from ..trainer.base_trainer import Trainer
from ..parameters.hyperparameter_base import HyperParameter
from ..hyperparameter.dag import HyperParameterDAG


class TunerLoggingKeys(Enum):
    FIXEDPARAMS = "Fixed Hyperparameters"
    TUNABLEPARAMS = "Tunable Hyperparameters"
    TUNEMETRICS = "Tuning Metrics"
    TRAINMETRICS = "Train Metrics"
    # TODO: Maybe also add a section to point to the logs of loss curves.
    #       Here, maybe just point to the general save location and in the
    #       csv-file save the correct loss curve locations?


def worker_eval(jobs):
    trainer_factory, params, device = jobs
    local_trainer: Trainer = trainer_factory()
    if isinstance(device, str):
        local_trainer.set_device(device)
    local_trainer.set_hyperparameter(params)
    local_trainer.run(show_progress=False)
    results = local_trainer.get_tuning_results()

    # Cleanup
    # TODO: Add clean up in trainer, so it can be backend dependent
    del local_trainer  # remove references

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
#       And maybe better switch to os["set_visible_devices"]?
class Tuner:

    def __init__(
        self,
        trainer_factory: Callable[[], Trainer],
        trial_number: int = 10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        save_path: str = "tuner_results",
    ) -> None:
        self.trainer_factory = trainer_factory
        self.trial_number = trial_number

        if isinstance(devices, str):
            devices = [devices]
        self.devices = [device for device in devices for _ in range(trials_per_device)]
        self.process_number = len(self.devices)

        # check trainer factory
        trainer_dummy = self.trainer_factory()
        assert (
            len(trainer_dummy.tuning_constraints) > 0
        ), "The trainer object does not contain any constraints for tuning. \
            Set them via trainer.set_tuning_constraints(...)."
        # Build saving path
        self.save_path = save_path
        # TODO: does this work for all systems? Windows and Mac?
        file_path_extension = self.save_path + "/" + trainer_dummy.save_path + "/"
        counter = 0
        while os.path.exists(file_path_extension):
            counter += 1
            file_path_extension = (
                self.save_path + "/" + trainer_dummy.save_path + str(counter) + "/"
            )
        self.file_path = file_path_extension
        self.csv_path = f"{self.file_path}/tuning_results.csv"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Find what parameters can be tuned
        param_log = self._get_tuneable_parameters(trainer_dummy)
        self.write_constrain_info(
            param_log,
            trainer_dummy.tuning_constraints,
            TunerLoggingKeys.TUNEMETRICS,
        )
        self.write_constrain_info(
            param_log,
            trainer_dummy.training_constraints,
            TunerLoggingKeys.TRAINMETRICS,
        )

        with open(f"{self.file_path}/tuning_setup.json", "w", encoding="utf-8") as f:
            json.dump(param_log, f, indent=4)

        self.write_system_info()

    def _get_tuneable_parameters(self, trainer: Trainer):
        hyperparameter_set = trainer.hyperparameters
        tunable_parameters = set[HyperParameter]()
        # Save all parameters also to a file:
        param_log = {
            TunerLoggingKeys.FIXEDPARAMS.value: {},
            TunerLoggingKeys.TUNABLEPARAMS.value: {},
        }
        for hp in hyperparameter_set:
            if not hp.is_fixed:
                tunable_parameters.add(hp)
                param_log[TunerLoggingKeys.TUNABLEPARAMS.value][hp.name] = type(
                    hp
                ).__name__
            else:
                param_log[TunerLoggingKeys.FIXEDPARAMS.value][hp.name] = hp.current_value

        if len(tunable_parameters) == 0:
            raise ValueError("Can not tune a problem without tunable parameters.")

        self.hp_dag = HyperParameterDAG(tunable_parameters)
        return param_log

    def run(self):
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
        # TODO: Using the same pools and not recreating everything would be nice, but:
        # - Data stays on the GPUs and can only be cleared by backend dependent calls
        with mp.Pool(
            processes=self.process_number,
            # initializer=_init_worker,
            # initargs=(self.trainer_object,),
        ) as pool:
            jobs = [(self.trainer_factory, p, d) for p, d in zip(params, self.devices)]
            results = list(pool.imap(worker_eval, jobs))
        return results

    def _write_to_csv(
        self, results, trial: Any | None = None
    ):  # pylint: disable=unused-argument
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
            for hp in self.hp_dag.sorted_nodes:
                if hp.name in all_params:
                    flat_dict[hp.name] = all_params[hp.name]
                else:
                    flat_dict[hp.name] = ""

        for key, value in results[1].items():
            flat_dict[key] = value

        return flat_dict

    def _get_trial_parameters(
        self, current_trial: int
    ) -> list[dict[str, dict[str, Any]]]:
        raise NotImplementedError(
            "The base Tuner does not implement a search strategy, \
                use one of the child classes."
        )

    def write_system_info(self):
        system_specs = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        system_specs["used_devices"] = self.devices

        with open(f"{self.file_path}/system_specs.json", "w", encoding="utf-8") as f:
            json.dump(system_specs, f, indent=4)

    def write_constrain_info(
        self, param_log, constraints: list[Constraint], key: TunerLoggingKeys
    ):
        param_log[key.value] = {}
        for constraint in constraints:
            constraint_dic = {"objective": constraint.objective}
            if hasattr(constraint, "relative"):
                rel_value = constraint.relative  # type: ignore
                if isinstance(rel_value, HyperParameter):
                    if rel_value.is_fixed:
                        constraint_dic["relative"] = rel_value.value
                    else:
                        constraint_dic["relative"] = type(rel_value).__name__
                else:
                    constraint_dic["relative"] = rel_value

            param_log[key.value][constraint.name] = constraint_dic
