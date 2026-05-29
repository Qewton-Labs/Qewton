from copy import deepcopy
import csv
import os
import multiprocessing as mp
import sys
from typing import Any, Tuple

import torch

from ..base import EvaluationPhase
from ..trainer.base_trainer import Trainer
from ..trainer.training_controllers import TrainerState
from ..parameters.hyperparameter_base import HyperParameter
from ..parameters.dag import HyperParameterDAG

# TODO:
# if that works, move tuning objective into tuner
# Add tuningcallbacks (Memory, Trainingtime, evaluation time, earlystopping, ...)
# Add TuningState (saves all TrainerStates (maybe with reduced resolution))
# TuningCallbacks have access to the TuningState


def worker(trainer, device, task_queue, result_queue):
    while True:
        params = task_queue.get()

        if params is None:
            break

        try:
            local_trainer: Trainer = deepcopy(trainer)
            if isinstance(device, str):
                local_trainer.set_device(device)
            local_trainer.set_hyperparameter(params)
            local_trainer.run(show_progress=False)
            local_trainer.evaluate_tuning_constraints()
            local_trainer.train_state.losses = local_trainer.train_state.detach_data(
                local_trainer.train_state.losses
            )
            result_queue.put((params, local_trainer.train_state))

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        finally:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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
    """
    Base class for hyperparameter tuners.

    Manages the execution of multiple training trials with different hyperparameters
    across various devices, and logs the results.
    """

    def __init__(
        self,
        trainer: Trainer,
        trial_number: int = 10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        save_path: str = "tuner",
        save_interval: int = 10,
    ) -> None:
        """
        Initializes the Tuner.
        Args:
            trainer (Trainer): A callable that returns a new Trainer instance.
            trial_number (int, optional): The total number of hyperparameter trials to run. Defaults to 10.
            devices (str | list[str], optional): A single device name (e.g., "cpu", "cuda:0") or a list of device names.
                                                 Defaults to "cpu".
            trials_per_device (int, optional): The number of trials to run concurrently on each device. Defaults to 1.
            save_path (str, optional): The base directory to save tuning results. Defaults to "tuner_results".
        """
        self.trainer = trainer
        self.trial_number = trial_number

        if isinstance(devices, str):
            devices = [devices]
        # Distribute trials on devices
        self.devices = [device for device in devices for _ in range(trials_per_device)]
        self.process_number = len(self.devices)

        # Check trainer factory and if tuning data is set:
        self.trainer.populate_state_dict()
        assert (
            len(self.trainer.train_state.losses[EvaluationPhase.TUNE])
            + len(self.trainer.train_state.metrics[EvaluationPhase.TUNE])
            > 0
        ), "The trainer object does not contain any constraints for tuning. \
            Set them via trainer.set_tuning_constraints(...)."

        # Find what parameters can be tuned
        self.hp_dag = self._get_tuneable_parameters(self.trainer)

        # Build saving path
        self.save_path = save_path
        self.file_path = self.build_save_path(self.trainer)
        self.csv_path = os.path.join(self.file_path, "study.csv")
        self._setup_csv_file(self.trainer.train_state)

        # Queues for parallel processing:
        self.save_interval = save_interval
        self.task_queue: mp.Queue
        self.result_queue: mp.Queue
        self.workers: list[Any]

    def build_save_path(self, trainer: Trainer) -> str:
        """
        Constructs a unique save path for the tuning results.
        Args:
            trainer (Trainer): A trainer instance to get its save_path.
        Returns:
            str: The unique file path for saving results.
        """
        # base_path = os.path.join(self.save_path, trainer.train_state.save_path)

        file_path = self.save_path
        counter = 0

        while os.path.exists(file_path):
            counter += 1
            file_path = f"{file_path}_{counter}"

        os.makedirs(file_path, exist_ok=True)
        trainer.train_state.save_path = os.path.join(
            file_path, trainer.train_state.save_path
        )
        return file_path

    def _get_tuneable_parameters(self, trainer: Trainer) -> HyperParameterDAG:
        """
        Identifies and collects all tunable hyperparameters from the trainer.
        Args:
            trainer (Trainer): A dummy trainer instance to inspect its hyperparameters.
        Returns:
            HyperParameterDAG: A DAG representing the dependencies and structure of tunable hyperparameters.
        """
        hyperparameter_set = trainer.hyperparameters
        tunable_parameters = set[HyperParameter]()
        for hp in hyperparameter_set:
            if not hp.is_fixed:
                tunable_parameters.add(hp)

        if len(tunable_parameters) == 0:
            raise ValueError("Can not tune a problem without tunable parameters.")

        return HyperParameterDAG(tunable_parameters)

    def run(self):
        if sys.platform == "linux" or sys.platform == "linux2":
            context_str = "fork"
        else:
            context_str = "spawn"

        try:
            ctx = mp.get_context(context_str)

            self.task_queue = ctx.Queue()
            self.result_queue = ctx.Queue()

            self.workers = [
                ctx.Process(
                    target=worker,
                    args=(self.trainer, device, self.task_queue, self.result_queue),
                )
                for device in self.devices
            ]
            for w in self.workers:
                w.start()

            trial_params = self._get_trial_parameters()

            print("--- Start Tuning ---")
            for params in trial_params:
                self.task_queue.put(params)

            current_results = []
            done_counter = 0
            self.print_update_text(done_counter, len(trial_params))
            for _ in range(len(trial_params)):
                result = self.result_queue.get()
                current_results.append(result)
                if len(current_results) % self.save_interval == 0:
                    self._write_to_csv(current_results)
                    current_results = []
                    done_counter += self.save_interval
                    self.print_update_text(done_counter, len(trial_params))

            if len(current_results) > 0:
                self._write_to_csv(current_results)

            print("--- Cleaning up ---")
            for _ in self.workers:
                self.task_queue.put(None)

        finally:
            for w in self.workers:
                if w.is_alive():
                    w.join(timeout=5.0)
                    if w.is_alive():
                        w.terminate()
                w.join()
            print("--- Finished Tuning ---")

    def print_update_text(self, done_counter, len_trial_params):
        upper_limit = min(done_counter + self.save_interval, len_trial_params)
        print(f"Working on trials {done_counter} - {upper_limit}")

    def _setup_csv_file(self, trainer_state_dummy: TrainerState):
        """
        Sets up the CSV file for logging tuning results, including writing the header.
        Args:
            trainer_state_dummy (TrainerState): A dummy trainer state to extract loss and metric names.
        """
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
        """
        Writes the results of a batch of trials to the CSV file.
        Args:
            results (list[Tuple[dict[str, Any], TrainerState]]): A list of (parameters, trainer_state) tuples.
            trial (Any | None, optional): Placeholder for potential future trial object. Defaults to None.
        """
        flat_results = [self._flatten_result_data(r) for r in results]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
            writer.writerows(flat_results)

    def _flatten_result_data(self, result: Tuple[dict[str, Any], TrainerState]):
        """Flattens the result data (hyperparameters, losses, metrics) into a single
        dictionary for CSV writing.

        Args:
            result (Tuple[dict[str, Any], TrainerState]): A tuple containing
                the parameters and trainer state for a trial.
        Returns:
            dict: A flattened dictionary suitable for CSV row.
        """
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

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        """
        Abstract method to generate the next set of hyperparameters for trials.
        Args:
            current_trial (int): The current trial number (used for seeding or progress tracking).
        Raises:
            NotImplementedError: This method must be implemented by subclasses to define a search strategy.
        """
        raise NotImplementedError(
            "The base Tuner does not implement a search strategy, \
                use one of the child classes."
        )
