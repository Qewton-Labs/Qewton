from .backend import BackendOptimizer
from ..base import EvaluationMode
from ..hyperparameter.base import HyperParameter, DiscreteHyperparameter
from ...pipeline.base import Pipeline
from ...nodes.base import Node
from ...constraints.base import Constraint


###############################
# TODO: This trainer is just some first idea.
# I think for more general optimizers this does not work (e.g. LBFGS)
# Also not sure about different backends, highly orientated on PyTorch

# TODO: What when the user does not want to work with the pipelines in the
# backend?

# TODO: Add parallelization?


# TODO: Also add stuff like callbacks
###############################
class Trainer:
    def __init__(
        self,
        backend: BackendOptimizer,
        pipelines: list[Pipeline],
        max_iterations: int | DiscreteHyperparameter,
        device="cpu",
        validation_check: int = 100,
    ):

        self.backend: BackendOptimizer = backend
        self.max_iterations: HyperParameter = HyperParameter.from_value(
            max_iterations, "Max. Iterations"
        )
        self.validation_check = validation_check
        self.device = device

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
        trainable_parameters = []
        for node in self.all_nodes:
            node_params = node.trainable_parameters
            if node_params is not None:
                trainable_parameters.append(node_params)
        return trainable_parameters

    def _move_to_device(self):
        for node in self.all_nodes:
            node.to(self.device)

    def run(self):
        # Setup all data inside the problem and create models
        all_pipelines = self.train_pipelines.union(self.validation_pipelines)
        all_pipelines = all_pipelines.union(self.test_pipelines)
        for pipeline in all_pipelines:
            pipeline.setup()
        # Register trainable parameters
        self._move_to_device()
        trainable_parameters = self._get_trainable_parameters()
        self.backend.setup(trainable_parameters)
        # Start training loop
        for step in self.max_iterations.current_value:

            total_loss = 0.0
            # Run all pipelines that contain training constraints
            for pipeline in self.train_pipelines:
                total_loss += self._run_pipeline(pipeline, EvaluationMode.TRAIN)

            # Update parameters
            self.backend.compute_gradients(total_loss)
            self.backend.apply_gradients()

            # Check validation data
            if step % self.validation_check == 0:
                validation_loss = 0.0
                for pipeline in self.validation_pipelines:
                    validation_loss += self._run_pipeline(
                        pipeline, EvaluationMode.VALIDATION
                    )
                print(
                    f"Training loss at {step}/{self.max_iterations.current_value}: \
                      {total_loss}"
                )
                print(
                    f"Validation loss at {step}/{self.max_iterations.current_value}: \
                    {validation_loss}"
                )

        # Run test at the end
        test_loss = 0.0
        for pipeline in self.test_pipelines:
            test_loss += self._run_pipeline(pipeline, EvaluationMode.TEST)
        print("Final testing loss is", test_loss)

    def _run_pipeline(self, pipeline: Pipeline, mode: EvaluationMode) -> float:
        pipeline.set_mode(mode)
        run_time = pipeline.create_runtime()
        run_time.run()
        pipeline_loss = 0.0
        # TODO: Not helpful if we want to track each loss independently.
        for constrain_node in pipeline.constrain_nodes:
            if constrain_node.mode == mode:
                pipeline_loss += constrain_node.get_loss()
        return pipeline_loss

    def get_hyperparameter(self) -> dict[str, list[HyperParameter]]:
        hyperparameter_dict: dict[str, list[HyperParameter]] = {}
        for node in self.all_nodes:
            node_params = node.hyperparameters
            if len(node_params) > 0:
                hyperparameter_dict[node.name] = node.hyperparameters
        # TODO: Not completely correct, since maybe we want to try different
        # Optimizers, saved as Hyperparameters?!
        hyperparameter_dict["trainer"] = [self.max_iterations]
        if len(self.backend.hyperparameters) > 0:
            hyperparameter_dict["backend"] = self.backend.hyperparameters
        return hyperparameter_dict
