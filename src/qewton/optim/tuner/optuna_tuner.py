from copy import deepcopy
import math
import queue
import multiprocessing as mp
import sys
import os
from typing import Any
import optuna

from qewton.optim.tuner.base import Tuner
from qewton.optim.tuner.tuning_callbacks.state import TuningState
from qewton.optim.tuner.tuning_callbacks.tuning_callback import TuningCallback
from qewton.optim.base import EvaluationPhase
from qewton.optim.trainer.base_trainer import Trainer
from qewton.optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
    BooleanHyperparameter,
)
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
    ContinuousHyperparameter,
    HyperParameterScale,
)
from qewton.optim.parameters.dag import HyperParameterDAG
from qewton.optim.tuner.results.tune_results import TrainResult, TuneResultCollector
from qewton.constraints.base import Constraint

# TODO: Just a first version to try this out


def optuna_worker(
    optuna_study, trainer, objective, device, n_trials, hp_dag, result_queue
):
    optuna_study.optimize(
        lambda trial: optuna_objective(
            trial,
            trainer=trainer,
            objective=objective,
            hp_dag=hp_dag,
            device=device,
            result_queue=result_queue,
        ),
        n_trials=n_trials,
        # callbacks=callbacks,
    )


def optuna_objective(
    trial: optuna.Trial,
    trainer: Trainer,
    objective: Constraint,
    hp_dag: HyperParameterDAG,
    device: str,
    result_queue,
):
    # Sample hyperparameters from trial
    config = {}
    for hp in hp_dag.sorted_nodes:
        if not hp.is_active(config):
            continue
        if isinstance(hp, DiscreteHyperparameter):
            config[hp.name] = trial.suggest_int(
                hp.name,
                hp.parameter_range[0],
                hp.parameter_range[1],
                log=hp.scale == HyperParameterScale.LOG,
            )
        elif isinstance(hp, ContinuousHyperparameter):
            config[hp.name] = trial.suggest_float(
                hp.name,
                hp.parameter_range[0],
                hp.parameter_range[1],
                log=hp.scale == HyperParameterScale.LOG,
            )
        elif isinstance(hp, BooleanHyperparameter):
            config[hp.name] = trial.suggest_categorical(hp.name, hp.parameter_range)
        elif isinstance(hp, CategoricalHyperparameter):
            config[hp.name] = trial.suggest_categorical(hp.name, hp.categories)

    # Run evaluation
    local_trainer: Trainer = deepcopy(trainer)
    if isinstance(device, str):
        local_trainer.set_device(device)
    local_trainer.set_hyperparameter(config)
    local_trainer.run(show_progress=False)
    local_trainer.train_state.losses = local_trainer.train_state.detach_data(
        local_trainer.train_state.losses
    )
    result_queue.put(TrainResult(config, local_trainer.train_state))

    # TODO: Allow for multiple constraints to be optimized
    total_loss = None
    if objective.name in local_trainer.train_state.losses[EvaluationPhase.VALIDATION]:
        total_loss = local_trainer.train_state.losses[EvaluationPhase.VALIDATION][
            objective.name
        ]
    if total_loss is None:
        total_loss = local_trainer.train_state.losses[EvaluationPhase.TRAIN][
            objective.name
        ]
    return total_loss


class OptunaTuner(Tuner):
    def __init__(
        self,
        trainer: Trainer,
        tuning_objectives: list,
        optuna_study: optuna.Study,
        trial_number: int = 10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        track_tune_state: bool | TuningState = True,
        tuning_callbacks: list[TuningCallback] | None = None,
        save_path: str = "tuner",
        result_collector: TuneResultCollector = TuneResultCollector(),
        use_multiprocessing: bool = True,
    ):
        super().__init__(
            trainer=trainer,
            tuning_objectives=tuning_objectives,
            trial_number=trial_number,
            devices=devices,
            trials_per_device=trials_per_device,
            track_tune_state=track_tune_state,
            tuning_callbacks=tuning_callbacks,
            save_path=save_path,
            result_collector=result_collector,
            use_multiprocessing=use_multiprocessing,
        )
        self.study = optuna_study
        assert len(self.tuning_objectives) == 1, "Currently only one objective supported!"

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        return []

    def run(self):
        print("--- Start Optuna Tuning ---")
        self.result_collector.setup_file_tree()
        if not self.use_multiprocessing:
            res_queue = queue.Queue()
            self.study.optimize(
                lambda trial: optuna_objective(
                    trial,
                    self.trainer,
                    self.tuning_objectives[0],
                    self.hp_dag,
                    self.devices[0] if self.devices else "cpu",
                    res_queue,
                ),
                n_trials=self.trial_number,
            )
            current_results = [res_queue.get() for _ in range(res_queue.qsize())]
            for result in current_results:
                self.result_collector.add_result(result)

            self.result_collector.finish_tuning()
            print("--- Finished Tuning ---")
            return

        if sys.platform == "linux" or sys.platform == "linux2":
            context_str = "fork"
        else:
            context_str = "spawn"

        try:
            trials = math.ceil(self.trial_number / self.process_number)

            ctx = mp.get_context(context_str)
            self.result_queue = ctx.Queue()
            self.stop_event = ctx.Event()

            self.workers = [
                ctx.Process(
                    target=optuna_worker,
                    args=(
                        self.study,
                        self.trainer,
                        self.tuning_objectives[0],
                        device,
                        trials,
                        self.hp_dag,
                        self.result_queue,
                    ),
                )
                for device in self.devices
            ]
            for w in self.workers:
                w.start()

            done_counter = 0
            self.print_update_text(done_counter, trials * self.process_number)
            for _ in range(trials * self.process_number):
                result = self.result_queue.get()
                self.result_collector.add_result(result)
                done_counter += 1
                # Log the current results:
                if self.tuning_state:
                    self.tuning_state.finished_trials += 1
                    self.tuning_state.add_trial_history(result.train_state.history)

                    if self.tuning_state.stop_tuning:
                        print("Stopping tuning...")
                        self.stop_event.set()
                        break

                if done_counter % self.result_collector.save_interval == 0:
                    self.print_update_text(done_counter, trials * self.process_number)

            self.result_collector.finish_tuning()

        finally:
            for w in self.workers:
                if w.is_alive():
                    w.join(timeout=10.0)
                    if w.is_alive():
                        w.terminate()
                w.join()
            print("--- Finished Tuning ---")
        print("Best params:", self.study.best_params)

    def build_save_path(self, save_path: str, trainer: Trainer) -> str:
        """
        Constructs a unique save path for the tuning results.
        Args:
            trainer (Trainer): A trainer instance to get its save_path.
        Returns:
            str: The unique file path for saving results.
        """
        # base_path = os.path.join(self.save_path, trainer.train_state.save_path)

        trainer.train_state.save_path = os.path.join(
            save_path, trainer.train_state.save_path
        )
        return save_path
