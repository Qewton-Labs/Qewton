import csv
import math
import multiprocessing as mp
from typing import Any, Callable
import optuna

from .base import Tuner, worker_eval
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


def run_optimization(job):
    job[0].optimize(
        lambda trial: optuna_objective(trial, job[1], job[2], job[3]),
        n_trials=job[4],
        callbacks=job[5],
    )


def optuna_objective(
    trial: optuna.Trial,
    trainer_factory: Callable[[], Trainer],
    hp_dag: HyperParameterDAG,
    device: str,
):
    # Sample hyperparameters from trial
    config = {}
    for hp in hp_dag.sorted_nodes:
        config = {}
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
    _, train_state = worker_eval((trainer_factory, config, device))
    # TODO: adapt to multiple metrics and losses also for minimum/maximum
    total_loss = 0.0
    for loss_name, loss in train_state.losses[EvaluationPhase.TUNE].items():
        trial.set_user_attr(loss_name, loss)
        total_loss += loss
    return total_loss


class OptunaTuner(Tuner):
    def __init__(
        self,
        trainer_factory: Callable[[], Trainer],
        optuna_study: optuna.Study,
        trial_number=10,
        devices: str | list[str] = "cpu",
        trials_per_device: int = 1,
        save_path="tuner_results",
    ):
        super().__init__(
            trainer_factory,
            trial_number=trial_number,
            devices=devices,
            trials_per_device=trials_per_device,
            save_path=save_path,
        )
        self.study = optuna_study

    def _get_trial_parameters(
        self, current_trial: int
    ) -> list[dict[str, dict[str, Any]]]:
        return []

    def run(self):
        print("--- Start Optuna Tuning ---")
        trials = math.ceil(self.trial_number / self.process_number)
        jobs = [
            (
                self.study,
                self.trainer_factory,
                self.hp_dag,
                d,
                trials,
                [self._write_to_csv],  # callbacks
            )
            for d in self.devices
        ]
        with mp.Pool(processes=self.process_number) as pool:
            _a = list(pool.imap(run_optimization, jobs))
        print("Best params:", self.study.best_params)

    def _write_to_csv(self, results, trial: Any = None):
        flat_dict = {}
        params = trial.params if trial is not None else {}
        for name in self.csv_columns:
            if name in params:
                if isinstance(params[name], type):
                    flat_dict[name] = params[name].__name__
                else:
                    flat_dict[name] = params[name]
            elif trial is not None and name in trial.user_attrs:
                flat_dict[name] = trial.user_attrs[name]
            else:
                flat_dict[name] = ""

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_dict.keys())
            writer.writerow(flat_dict)
