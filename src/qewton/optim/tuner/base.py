from copy import deepcopy
import csv
import os
import multiprocessing as mp
import sys
from typing import Any, Tuple

from qewton.optim.tuner.tuning_callbacks.state import TuningState
from qewton.optim.tuner.tuning_callbacks.tuning_callback import TuningCallback
from qewton.optim.base import EvaluationPhase
from qewton.optim.trainer.base_trainer import Trainer
from qewton.optim.trainer.training_controllers import TrainerState
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.dag import HyperParameterDAG


def worker(
    trainer,
    device,
    tune_state: TuningState,
    task_queue,
    result_queue,
    stop_event,
):
    while True:
        params = task_queue.get()
        local_trainer: None | Trainer = None  # type: ignore

        if params is None or stop_event.is_set():
            break

        try:
            local_trainer: Trainer = deepcopy(trainer)
            if isinstance(device, str):
                local_trainer.set_device(device)

            # Dont copy the state of the tuner. Its only set once here
            # and all processes have the same one.
            # TODO: This does not work currently, the tune_state is now local
            # Maybe do a system where the main process has the callbacks and
            # is coupled to child callbacks that only request information.
            if tune_state is not None:
                for cb in local_trainer.callbacks:
                    if isinstance(cb, TuningCallback):
                        cb.set_tune_state(tune_state)

            local_trainer.set_hyperparameter(params)
            local_trainer.run(show_progress=False)
        except Exception as e:
            if local_trainer is not None:
                local_trainer.train_state.termination_reason = f"Exception: {e}"
            else:
                result_queue.put((params, None))
        finally:
            if local_trainer is not None:
                local_trainer.train_state.losses = local_trainer.train_state.detach_data(
                    local_trainer.train_state.losses
                )
                result_queue.put((params, local_trainer.train_state))
                local_trainer.cleanup()


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

    save_keys = ["Termination Reason", "Training Time [s]", "Save Path"]

    def __init__(
        self,
        trainer: Trainer,
        tuning_objectives: list,
        trial_number: int = 10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        track_tune_state: bool | TuningState = True,
        tuning_callbacks: list[TuningCallback] | None = None,
        save_path: str = "tuner",
        save_interval: int = 10,
        use_multiprocessing: bool = True,
    ) -> None:
        """
        Initializes the Tuner.
        Args:
            trainer (Trainer): A callable that returns a new Trainer instance.
            trial_number (int, optional): The total number of hyperparameter trials
                to run. Defaults to 10.
            devices (str | list[str], optional): A single device name (e.g., "cpu",
                "cuda:0") or a list of device names. Defaults to "cpu".
            trials_per_device (int, optional): The number of trials to run concurrently
                on each device. Defaults to 1.
            save_path (str, optional): The base directory to save tuning results.
                Defaults to "tuner_results".
        """
        self.tuning_state = None
        if isinstance(track_tune_state, TuningState):
            self.tuning_state = track_tune_state
        if track_tune_state:
            self.tuning_state = TuningState(save_path)
        if not trainer.train_state.enable_logging and self.tuning_state is not None:
            raise RuntimeError(
                "Tuner can not log the tuning process, because logging "
                "is disabled in the trainer."
            )
        if tuning_callbacks is None:
            tuning_callbacks = []
        if len(tuning_callbacks) > 0 and self.tuning_state is None:
            raise RuntimeError(
                "Tuner can not use callbacks to track the tuning process, "
                "because no TuningState is provided."
            )

        trainer.callbacks.extend(tuning_callbacks)
        trainer.check_tuning_constraints_exist(tuning_objectives)

        self.trainer = trainer
        self.trial_number = trial_number
        self.tuning_objectives = tuning_objectives
        self.use_multiprocessing = use_multiprocessing

        if isinstance(devices, str):
            devices = [devices]
        # Distribute trials on devices
        self.devices = [device for device in devices for _ in range(trials_per_device)]
        self.process_number = len(self.devices)

        # Check trainer factory and if tuning data is set:
        self.trainer.populate_state_dict()
        assert (
            len(tuning_objectives) > 0
        ), "The trainer object does not contain any constraints for tuning. \
            Set them via trainer.set_tuning_constraints(...)."

        # Find what parameters can be tuned
        self.hp_dag = self._get_tuneable_parameters(self.trainer)

        # Build saving path
        self.save_path = save_path
        self.file_path = self.build_save_path(self.trainer)
        self.csv_path = os.path.join(self.file_path, "study.csv")
        self._setup_csv_file()

        # Queues for parallel processing:
        self.save_interval = save_interval
        self.task_queue: mp.Queue
        self.result_queue: mp.Queue
        self.stop_event: mp.Event  # type: ignore
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
            file_path = f"{self.save_path}_{counter}"

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
            HyperParameterDAG: A DAG representing the dependencies and structure of
                tunable hyperparameters.
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

        trial_params = self._get_trial_parameters()

        if not self.use_multiprocessing:
            print("--- Start Tuning (Sequential) ---")
            current_results = []
            done_counter = 0
            self.print_update_text(done_counter, len(trial_params))
            for params in trial_params:
                local_trainer = deepcopy(self.trainer)
                if self.devices:
                    local_trainer.set_device(self.devices[0])

                if self.tuning_state is not None:
                    for cb in local_trainer.callbacks:
                        if isinstance(cb, TuningCallback):
                            cb.set_tune_state(self.tuning_state)

                local_trainer.set_hyperparameter(params)
                local_trainer.run(show_progress=False)
                local_trainer.train_state.losses = local_trainer.train_state.detach_data(
                    local_trainer.train_state.losses
                )
                result = (params, local_trainer.train_state)
                current_results.append(result)

                if self.tuning_state:
                    self.tuning_state.finished_trials += 1
                    self.tuning_state.add_trial_history(result[1].history)
                    if self.tuning_state.stop_tuning:
                        break

                if len(current_results) % self.save_interval == 0:
                    self._write_to_csv(current_results)
                    current_results = []
                    done_counter += self.save_interval
                    self.print_update_text(done_counter, len(trial_params))
                local_trainer.cleanup()

            if len(current_results) > 0:
                self._write_to_csv(current_results)
            print("--- Finished Tuning ---")
            return

        try:
            ctx = mp.get_context(context_str)

            self.task_queue = ctx.Queue()
            self.result_queue = ctx.Queue()
            self.stop_event = ctx.Event()

            self.workers = [
                ctx.Process(
                    target=worker,
                    args=(
                        self.trainer,
                        device,
                        self.tuning_state,
                        self.task_queue,
                        self.result_queue,
                        self.stop_event,
                    ),
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

                # Log the current results:
                if self.tuning_state:
                    self.tuning_state.finished_trials += 1
                    self.tuning_state.add_trial_history(result[1].history)

                    if self.tuning_state.stop_tuning:
                        print("Stopping tuning...")
                        self.stop_event.set()
                        break

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

    def _setup_csv_file(self):
        """
        Sets up the CSV file for logging tuning results, including writing the header.
        Args:
            trainer_state_dummy (TrainerState): A dummy trainer state to extract loss and metric names.
        """
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                param_names = [hp.name for hp in self.hp_dag.sorted_nodes]
                objective_names = [con.name for con in self.tuning_objectives]
                # TODO: maybe add callback info here?
                self.csv_columns = param_names + objective_names + self.save_keys
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()

    def _write_to_csv(
        self, results: list[Tuple[dict[str, Any], TrainerState]], trial: Any | None = None
    ):  # pylint: disable=unused-argument
        """
        Writes the results of a batch of trials to the CSV file.
        Args:
            results (list[Tuple[dict[str, Any], TrainerState]]): A list of (parameters,
                trainer_state) tuples.
            trial (Any | None, optional): Placeholder for potential future trial object.
                Defaults to None.
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
        result_dict: dict[str, Any] = {}
        for obj in self.tuning_objectives:
            phase = obj.evaluated_in_mode
            if phase == EvaluationPhase.ALWAYS:
                if obj in result[1].losses[EvaluationPhase.VALIDATION]:
                    result_dict[obj.name] = result[1].losses[EvaluationPhase.VALIDATION][
                        obj.name
                    ]
                else:
                    result_dict[obj.name] = result[1].losses[EvaluationPhase.TRAIN][
                        obj.name
                    ]
            else:
                result_dict[obj.name] = result[1].losses[phase][obj.name]

        flat_dict = {}
        for name in self.csv_columns:
            if name == self.save_keys[0]:
                flat_dict[name] = result[1].termination_reason
            elif name == self.save_keys[1]:
                flat_dict[name] = result[1].total_train_time
            elif name == self.save_keys[2]:
                flat_dict[name] = result[1].save_path
            elif name in result[0]:
                if isinstance(result[0][name], type):
                    flat_dict[name] = result[0][name].__name__
                else:
                    flat_dict[name] = result[0][name]
            elif name in result_dict:
                flat_dict[name] = result_dict[name]
            else:
                flat_dict[name] = ""
        return flat_dict

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        """
        Abstract method to generate the next set of hyperparameters for trials.
        Args:
            current_trial (int): The current trial number (used for seeding or progress
                tracking).
        Raises:
            NotImplementedError: This method must be implemented by subclasses to
                define a search strategy.
        """
        raise NotImplementedError("The base Tuner does not implement a search strategy, \
                use one of the child classes.")
