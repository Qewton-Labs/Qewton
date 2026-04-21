import csv
import os
import math
import multiprocessing as mp
from typing import Any, Callable, Tuple

from ..base import EvaluationPhase
from ..trainer.base_trainer import Trainer
from ..trainer.training_controllers import TrainerState
from ..parameters.hyperparameter_base import HyperParameter
from ..parameters.dag import HyperParameterDAG


def worker_eval(jobs) -> Tuple[dict[str, Any], TrainerState]:
    trainer_factory, params, device = jobs
    local_trainer: Trainer = trainer_factory()
    if isinstance(device, str):
        local_trainer.set_device(device)
    local_trainer.set_hyperparameter(params)
    local_trainer.run(show_progress=False)
    local_trainer.evaluate_tuning_constraints()
    local_trainer.train_state.detach_data()  # detach data to avoid memory issues
    return (params, local_trainer.train_state)


# TODO: Implement callbacks for early stopping (loss or time based) and saving
#       best results
#
# TODO: Add constraints for memory usage and speed
#
# TODO: Enable to restart tuning from a given point
#
# TODO: In "devices" also allow for something like "auto" or "all" to automatically
#       use all available CPUs/GPUs? Currently passing in an int just copies
#       the trainer to the device given by the trainer!
#       And maybe better switch to os["set_visible_devices"]?
#
# TODO: Save more information about the system and tuning setup, e.g. in a json file,
#       to be able to better compare different tuning runs.
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
        # Distribute trials on devices
        self.devices = [device for device in devices for _ in range(trials_per_device)]
        self.process_number = len(self.devices)

        # Check trainer factory and if tuning data is set:
        trainer_dummy = self.trainer_factory()
        trainer_dummy.populate_state_dict()
        assert (
            len(trainer_dummy.train_state.losses[EvaluationPhase.TUNE])
            + len(trainer_dummy.train_state.metrics[EvaluationPhase.TUNE])
            > 0
        ), "The trainer object does not contain any constraints for tuning. \
            Set them via trainer.set_tuning_constraints(...)."

        # Find what parameters can be tuned
        self.hp_dag = self._get_tuneable_parameters(trainer_dummy)

        # Build saving path
        self.save_path = save_path
        self.file_path = self.build_save_path(trainer_dummy)
        self.csv_path = os.path.join(self.file_path, "tuning_results.csv")
        self._setup_csv_file(trainer_dummy.train_state)

    def build_save_path(self, trainer: Trainer) -> str:
        base_path = os.path.join(self.save_path, trainer.train_state.save_path)

        file_path = base_path
        counter = 0

        while os.path.exists(file_path):
            counter += 1
            file_path = f"{base_path}_{counter}"

        os.makedirs(file_path, exist_ok=True)
        return file_path

    def _get_tuneable_parameters(self, trainer: Trainer) -> HyperParameterDAG:
        hyperparameter_set = trainer.hyperparameters
        tunable_parameters = set[HyperParameter]()
        for hp in hyperparameter_set:
            if not hp.is_fixed:
                tunable_parameters.add(hp)

        if len(tunable_parameters) == 0:
            raise ValueError("Can not tune a problem without tunable parameters.")

        return HyperParameterDAG(tunable_parameters)

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

    def _setup_csv_file(self, trainer_state_dummy: TrainerState):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                param_names = [hp.name for hp in self.hp_dag.sorted_nodes]
                loss_names = list(trainer_state_dummy.losses[EvaluationPhase.TUNE].keys())
                metric_names = list(
                    trainer_state_dummy.metrics[EvaluationPhase.TUNE].keys()
                )
                self.csv_columns = param_names + loss_names + metric_names
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

    def _write_to_csv(
        self, results: list[Tuple[dict[str, Any], TrainerState]], trial: Any | None = None
    ):  # pylint: disable=unused-argument
        flat_results = [self._flatten_result_data(r) for r in results]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
            writer.writerows(flat_results)

    def _flatten_result_data(self, result: Tuple[dict[str, Any], TrainerState]):
        flat_dict = {}
        tune_loss = result[1].losses[EvaluationPhase.TUNE]
        tune_metrics = result[1].metrics[EvaluationPhase.TUNE]
        for name in self.csv_columns:
            if name in result[0]:
                if isinstance(result[0][name], type):
                    flat_dict[name] = result[0][name].__name__
                else:
                    flat_dict[name] = result[0][name]
            elif name in tune_loss:
                flat_dict[name] = tune_loss[name]
            elif name in tune_metrics:
                flat_dict[name] = tune_metrics[name]
            else:
                flat_dict[name] = ""
        return flat_dict

    def _get_trial_parameters(
        self, current_trial: int
    ) -> list[dict[str, dict[str, Any]]]:
        raise NotImplementedError(
            "The base Tuner does not implement a search strategy, \
                use one of the child classes."
        )
