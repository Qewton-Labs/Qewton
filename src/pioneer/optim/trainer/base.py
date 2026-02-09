from typing import Any

from ..base import EvaluationMode
from ..hyperparameter.base import HyperParameter, DiscreteHyperparameter
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
        optimizer_cls,
        max_iterations: int | DiscreteHyperparameter,
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

        self.train_pipelines = set[Pipeline]()
        self.validation_pipelines = set[Pipeline]()
        self.test_pipelines = set[Pipeline]()
        # Since different pipelines may have different nodes we
        # also collect all of them (without duplicates)
        self.all_nodes = set[Node]()
        self._register_pipelines(pipelines)

    def _register_pipelines(self, pipelines):
        for pipeline in pipelines:
            for node in pipeline.nodes:
                self.all_nodes.add(node)
                if isinstance(node, Constraint):
                    match node.mode:
                        case EvaluationMode.TRAIN:
                            self.train_pipelines.add(pipeline)
                        case EvaluationMode.VALIDATION:
                            self.validation_pipelines.add(pipeline)
                        case EvaluationMode.TEST:
                            self.test_pipelines.add(pipeline)
                        case EvaluationMode.TEST_AND_VALIDATION:
                            self.validation_pipelines.add(pipeline)
                            self.test_pipelines.add(pipeline)
                        case EvaluationMode.ALWAYS:
                            self.train_pipelines.add(pipeline)
                            self.validation_pipelines.add(pipeline)
                            self.test_pipelines.add(pipeline)

    def _get_trainable_parameters(self):
        return []

    def _move_to_device(self):
        pass

    def run(self) -> dict[str, dict[str, float]]:
        return {}

    def set_device(self, device: str):
        self.device = device

    def set_file_path(self, path: str):
        self.save_path = path
        # TODO: Set this path also for all callbacks and models

    def _run_pipeline(
        self, pipeline: Pipeline, mode: EvaluationMode
    ) -> tuple[float, dict[str, float]]:
        pipeline.set_mode(mode)
        run_time = pipeline.create_runtime()
        run_time.run()
        pipeline_loss_dict: dict[str, float] = {}
        for constrain_node in pipeline.constrain_nodes:
            if (
                constrain_node.mode == mode
                or constrain_node.mode == EvaluationMode.ALWAYS
                or (constrain_node.mode == EvaluationMode.TEST_AND_VALIDATION)
                and (mode == EvaluationMode.TEST or mode == EvaluationMode.VALIDATION)
            ):
                pipeline_loss_dict[constrain_node.name] = constrain_node.get_loss()
        pipeline_loss = sum(pipeline_loss_dict.values())
        return pipeline_loss, pipeline_loss_dict

    def get_hyperparameter(self) -> dict[str, list[HyperParameter]]:
        hyperparameter_dict: dict[str, list[HyperParameter]] = {}
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
