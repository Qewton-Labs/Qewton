import warnings
from typing import Any

from ..base import EvaluationMode
from ..hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    CategoricalHyperparameter,
)
from ...pipeline.base import Pipeline
from ...nodes.base import Node
from ...constraints.base import Constraint


###############################
# TODO: This trainer is just some first idea.
# I think for more general optimizers this does not work (e.g. LBFGS)
#
# TODO: What when the user does not want to work with the pipelines in the
# backend?
#
# TODO: Add parallelization?
#
# TODO: Add stuff callbacks
#
# TODO: What should the trainer return in run?
###############################
class Trainer:
    def __init__(
        self,
        pipelines: list[Pipeline],
        training_constraints: list[Constraint],
        validation_constraints: list[Constraint],
        optimizer_cls,
        max_iterations: int | DiscreteHyperparameter | CategoricalHyperparameter,
        device="cpu",
        validation_check: int = 100,
        save_path: str = "train_results",
    ):

        self.optimizer_cls = optimizer_cls
        self.max_iterations: HyperParameter = HyperParameter.from_value(
            max_iterations, "Max. Iterations"
        )
        self.validation_check = validation_check
        self.device = device
        self.save_path: str
        self.set_file_path(save_path)

        self.training_constraints = training_constraints
        self.validation_constraints = validation_constraints
        self.tuning_constraints: list[Constraint]

        # Since different pipelines may have different nodes we
        # also collect all of them (without duplicates)
        self.all_nodes = set[Node]()
        for pipeline in pipelines:
            for node in pipeline.nodes:
                self.all_nodes.add(node)

        self.all_pipelines = pipelines
        self.tune_pipelines = set[Pipeline]()
        self.train_pipelines = self._register_pipelines(self.training_constraints)
        self.validation_pipelines = self._register_pipelines(self.validation_constraints)

    def _register_pipelines(self, constraints: list[Constraint]) -> set[Pipeline]:
        found_constraints = set()
        pipeline_set = set[Pipeline]()
        for pipeline in self.all_pipelines:
            for constraint_node in pipeline.constrain_nodes:
                if constraint_node not in constraints:
                    continue

                found_constraints.add(constraint_node)
                pipeline_set.add(pipeline)

        # Check for missing constraints
        missing = set(constraints) - found_constraints
        if missing:
            warnings.warn(
                f"The Trainer has no pipelines containing constraints {missing}",
                RuntimeWarning,
            )
        return pipeline_set

    def set_tuning_constraints(self, constraints: list[Constraint]):
        self.tuning_constraints = constraints
        self.tune_pipelines = self._register_pipelines(self.tuning_constraints)

    def _get_trainable_parameters(self):
        return []

    def _move_to_device(self):
        pass

    def run(self, show_progress: bool = True):
        pass

    def set_device(self, device: str):
        self.device = device

    def set_file_path(self, path: str):
        self.save_path = path
        # TODO: Set this path also for all callbacks and models

    def _run_pipeline(self, pipeline: Pipeline, mode: EvaluationMode):
        pipeline.set_mode(mode)
        run_time = pipeline.create_runtime()
        run_time.run()

    def _evaluate_constraints(
        self, constraint_list: list[Constraint]
    ) -> dict[str, float]:

        pipeline_loss_dict: dict[str, float] = {}
        for constraint_node in constraint_list:
            pipeline_loss_dict[constraint_node.name] = constraint_node.get_loss()
        return pipeline_loss_dict

    def get_tuning_results(self):
        for constraint in self.tuning_constraints:
            constraint.set_mode(EvaluationMode.TUNE)
        for pipeline in self.tune_pipelines:
            self._run_pipeline(pipeline, EvaluationMode.TUNE)
        return self._evaluate_constraints(self.tuning_constraints)

    def _compute_loss(
        self,
        pipelines: set[Pipeline],
        mode: EvaluationMode,
        constraints: list[Constraint],
    ):
        for constraint in constraints:
            constraint.set_mode(mode)

        for pipeline in pipelines:
            self._run_pipeline(pipeline, mode)

        total_loss = 0.0
        for constraint in constraints:
            total_loss += constraint.get_loss()
        return total_loss

    def get_hyperparameter(self) -> dict[str, list[HyperParameter]]:
        hyperparameter_dict: dict[str, list[HyperParameter]] = {}
        # TODO: Does not work if we have conditional parameters?
        # We would need some kind of tree-structure
        for node in self.all_nodes:
            node_params = node.hyperparameters
            if len(node_params) > 0:
                hyperparameter_dict[node.name] = node.hyperparameters
        # TODO: Not completely correct, since maybe we want to try different
        # Optimizers, saved as Hyperparameters?! E.g. Adam, LBFGS or a combination
        # of them...
        # This then also needs to change the iterations accordingly
        hyperparameter_dict["trainer"] = [self.max_iterations]
        return hyperparameter_dict

    def set_hyperparameter(self, param_dict: dict[str, dict[str, Any]]):
        # TODO: Here also update the stuff for the trainer
        if "trainer" in param_dict:
            self.max_iterations.set_value(param_dict["trainer"][self.max_iterations.name])

        for node in self.all_nodes:
            if node.name in param_dict:
                node_hyperparameters = node.hyperparameters
                for param in node_hyperparameters:
                    if param.name in param_dict[node.name]:
                        param.set_value(param_dict[node.name][param.name])

    def reset(self):
        for node in self.all_nodes:
            node.reset()
