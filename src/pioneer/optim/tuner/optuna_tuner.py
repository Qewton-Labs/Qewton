from copy import deepcopy
import math
import multiprocessing as mp
import os
import sys
from typing import Any
import optuna

from .base import Tuner
from .tuning_callbacks.state import TuningState
from .tuning_callbacks.tuning_callback import TuningCallback
from ..base import EvaluationPhase
from ..trainer.base_trainer import Trainer
from ..parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
    BooleanHyperparameter,
)
from ..parameters.number_hyperparameter import (
    DiscreteHyperparameter,
    ContinuousHyperparameter,
    HyperParameterScale,
)
from ..parameters.dag import HyperParameterDAG
from ...constraints.base import Constraint

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
    result_queue.put((config, local_trainer.train_state))

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
        trial_number=10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        track_tune_state: bool | TuningState = True,
        tuning_callbacks: list[TuningCallback] | None = None,
        save_path="tuner_results",
    ):
        super().__init__(
            trainer,
            tuning_objectives=tuning_objectives,
            trial_number=trial_number,
            devices=devices,
            trials_per_device=trials_per_device,
            track_tune_state=track_tune_state,
            tuning_callbacks=tuning_callbacks,
            save_path=save_path,
        )
        self.study = optuna_study
        assert len(self.tuning_objectives) == 1, "Currently only one objective supported!"

    def _get_trial_parameters(self) -> list[dict[str, dict[str, Any]]]:
        return []

    def run(self):
        print("--- Start Optuna Tuning ---")

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

            current_results = []
            done_counter = 0
            self.print_update_text(done_counter, trials * self.process_number)
            for _ in range(trials * self.process_number):
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
                    self.print_update_text(done_counter, trials * self.process_number)

            if len(current_results) > 0:
                self._write_to_csv(current_results)

        finally:
            for w in self.workers:
                if w.is_alive():
                    w.join(timeout=10.0)
                    if w.is_alive():
                        w.terminate()
                w.join()
            print("--- Finished Tuning ---")
        print("Best params:", self.study.best_params)

    def build_save_path(self, trainer: Trainer) -> str:
        """
        Constructs a unique save path for the tuning results.
        Args:
            trainer (Trainer): A trainer instance to get its save_path.
        Returns:
            str: The unique file path for saving results.
        """
        # base_path = os.path.join(self.save_path, trainer.train_state.save_path)

        trainer.train_state.save_path = os.path.join(
            self.save_path, trainer.train_state.save_path
        )
        return self.save_path
