from copy import deepcopy
import os
import multiprocessing as mp
import sys
import gc

from typing import Any

from qewton.optim.tuner.tuning_callbacks.state import TuningState
from qewton.optim.tuner.tuning_callbacks.tuning_callback import TuningCallback
from qewton.optim.trainer.base_trainer import Trainer
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.dag import HyperParameterDAG
from qewton.optim.tuner.results.tune_results import TrainResult, TuneResultCollector


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
                result_queue.put(TrainResult(params=params, train_state=None))
        finally:
            if local_trainer is not None:
                local_trainer.train_state.losses = local_trainer.train_state.detach_data(
                    local_trainer.train_state.losses
                )
                result_queue.put(
                    TrainResult(params=params, train_state=local_trainer.train_state)
                )
                gpu_cleanup_fn = local_trainer.cleanup()
                del local_trainer
                gc.collect()
                gpu_cleanup_fn()


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
        result_collector: TuneResultCollector = TuneResultCollector(),
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
        save_path = self.build_save_path(save_path, self.trainer)
        self.result_collector = result_collector
        self.result_collector.save_path = save_path
        self.result_collector.set_hp_and_conditions(self.hp_dag, self.tuning_objectives)

        # Queues for parallel processing
        self.task_queue: mp.Queue
        self.result_queue: mp.Queue
        self.stop_event: mp.Event  # type: ignore
        self.workers: list[Any]

    def build_save_path(self, save_path: str, trainer: Trainer) -> str:
        """
        Constructs a unique save path for the tuning results.
        Args:
            trainer (Trainer): A trainer instance to get its save_path.
        Returns:
            str: The unique file path for saving results.
        """
        # base_path = os.path.join(self.save_path, trainer.train_state.save_path)

        file_path = save_path
        counter = 0

        while os.path.exists(file_path):
            counter += 1
            file_path = f"{save_path}_{counter}"

        trainer.train_state.save_path = os.path.join(
            file_path + "/train_results", trainer.train_state.save_path
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
        self.result_collector.setup_file_tree()
        done_counter = 0

        if not self.use_multiprocessing:
            print("--- Start Tuning (Sequential) ---")
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
                result = TrainResult(params, local_trainer.train_state)
                self.result_collector.add_result(result)

                if self.tuning_state:
                    self.tuning_state.finished_trials += 1
                    self.tuning_state.add_trial_history(local_trainer.train_state.history)
                    if self.tuning_state.stop_tuning:
                        break

                cleanup_fn = local_trainer.cleanup()
                del local_trainer
                gc.collect()
                cleanup_fn()

                done_counter += 1
                if done_counter % self.result_collector.save_interval == 0:
                    self.print_update_text(done_counter, len(trial_params))

            self.result_collector.finish_tuning()
            print("--- Finished Tuning ---")
            return

        self._multiprocess_tune(context_str, done_counter)

    def _multiprocess_tune(self, context_str, done_counter):
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

            self.print_update_text(done_counter, len(trial_params))
            for _ in range(len(trial_params)):
                result = self.result_queue.get()
                self.result_collector.add_result(result)

                done_counter += 1
                if done_counter % self.result_collector.save_interval == 0:
                    self.print_update_text(done_counter, len(trial_params))

                # Log the current results:
                if self.tuning_state:
                    self.tuning_state.finished_trials += 1
                    self.tuning_state.add_trial_history(result.train_state.history)

                    if self.tuning_state.stop_tuning:
                        print("Stopping tuning...")
                        self.stop_event.set()
                        break

            print("--- Cleaning up ---")
            self.result_collector.finish_tuning()
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
        upper_limit = min(
            done_counter + self.result_collector.save_interval, len_trial_params
        )
        print(f"Working on trials {done_counter} - {upper_limit}")

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
