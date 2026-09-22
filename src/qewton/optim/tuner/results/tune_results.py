from dataclasses import dataclass
import os
import csv
import json
from typing import Any

from qewton.optim.trainer.training_controllers import TrainerState
from qewton.optim.parameters.dag import HyperParameterDAG
from qewton.constraints.base import Constraint
from qewton.optim.base import EvaluationPhase


@dataclass
class TrainResult:
    """
    A class to store the results of a training process.

    Attributes:
        params (dict): The parameters used for training.
        score (float): The score achieved with these parameters.
    """

    params: dict
    train_state: TrainerState | None = None


class TuneResultCollector:
    """
    A class to store and save the results of a tuning process.
    """

    file_names = {"csv": "results.csv", "json": "setup.json"}
    termination_key = "Termination Reason"
    time_key = "Training Time [s]"
    save_key = "Save Path"
    hp_key = "Hyperparameters"
    hp_type_key = "Hyperparameter Types"
    objective_names_key = "Objective Names"
    objective_values_key = "Objective"

    def __init__(self, save_path="tuning_results", save_interval=10):
        self.save_interval = save_interval
        self.save_path = save_path
        self.current_results: list[TrainResult] = []

        self.hp_dag: HyperParameterDAG
        self.param_names: list[str]
        self.tuning_objectives: list[Constraint]
        self.objective_names: list[str]
        self.csv_path: str
        self.csv_columns: list[str]

    def set_hp_and_conditions(
        self, hp_dag: HyperParameterDAG, tuning_objectives: list[Constraint]
    ):
        """
        Set the hyperparameter DAG and tuning objectives for the collector.

        Args:
            hp_dag (HyperParameterDAG): The hyperparameter DAG.
            tuning_objectives (list): The list of tuning objectives.
        """
        self.hp_dag = hp_dag
        self.param_names = [hp.name for hp in hp_dag.sorted_nodes]
        self.tuning_objectives = tuning_objectives
        self.objective_names = [con.name for con in tuning_objectives]

    def add_result(self, new_result: TrainResult):
        """
        Add a new result to the collector.

        Args:
            new_result (TrainResult): The new result to be added.
        """
        self.current_results.append(new_result)
        if len(self.current_results) >= self.save_interval:
            self.save_results()
            self.current_results = []

    def finish_tuning(self):
        """
        Finalizes the tuning process by saving any remaining results.
        """
        if len(self.current_results) > 0:
            self.save_results()
            self.current_results = []

    def save_results(self):
        """
        Writes the results of a batch of trials to the CSV file.
        """
        flat_results = [self._flatten_result_data(r) for r in self.current_results]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
            writer.writerows(flat_results)

    def setup_file_tree(self):
        """
        Sets up the files for logging tuning results
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            self.csv_path = os.path.join(self.save_path, self.file_names["csv"])
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                info_keys = [self.termination_key, self.time_key, self.save_key]
                self.csv_columns = self.param_names + self.objective_names + info_keys
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writeheader()
            json_path = os.path.join(self.save_path, self.file_names["json"])
            with open(json_path, "w", encoding="utf-8") as f:
                info_dict = {
                    self.hp_key: self.param_names,
                    self.hp_type_key: [
                        str(hp.__class__.__name__) for hp in self.hp_dag.sorted_nodes
                    ],
                    self.objective_names_key: self.objective_names,
                    self.objective_values_key: [
                        str(obj.objective) for obj in self.tuning_objectives
                    ],
                }
                json.dump(info_dict, f, indent=4)

    def _flatten_result_data(self, result: TrainResult) -> dict[str, Any]:
        """Flattens the result data (hyperparameters, losses, metrics) into a single
        dictionary for CSV writing.

        Args:
            result (TrainResult): The result to be flattened.
        Returns:
            dict[str, Any]: A dictionary containing the flattened result data.
        """
        result_dict: dict[str, Any] = {k: "" for k in self.csv_columns}

        if result.train_state is None:
            result_dict = {
                **result.params,
                self.termination_key: "Setup of training failed",
            }
            return result_dict

        # Add the training constraints
        for obj in self.tuning_objectives:
            phase = obj.evaluated_in_mode
            if phase == EvaluationPhase.ALWAYS:
                if obj in result.train_state.losses[EvaluationPhase.VALIDATION]:
                    phase = EvaluationPhase.VALIDATION
                else:
                    phase = EvaluationPhase.TRAIN
            result_dict[obj.name] = result.train_state.losses[phase][obj.name]

        # Other information about the training process
        result_dict[self.termination_key] = result.train_state.termination_reason
        result_dict[self.time_key] = result.train_state.total_train_time
        result_dict[self.save_key] = result.train_state.save_path

        # Add current hyperparameter values to the result dictionary
        for hp_name in self.param_names:
            if hp_name in result.params:
                if isinstance(result.params[hp_name], type):
                    result_dict[hp_name] = result.params[hp_name].__name__
                else:
                    result_dict[hp_name] = result.params[hp_name]

        return result_dict
