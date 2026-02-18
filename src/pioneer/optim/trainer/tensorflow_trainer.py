import tensorflow as tf

from ..base import EvaluationMode
from ..hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    ContinuousHyperparameter,
    CategoricalHyperparameter,
)
from ...pipelines.base import Pipeline
from ...constraints.base import Constraint
from .base import Trainer


class TensorFlowTrainer(Trainer):
    def __init__(
        self,
        pipelines: list[Pipeline],
        training_constraints: list[Constraint],
        optimizer_cls: type[tf.keras.optimizers.Optimizer],  # type: ignore
        max_iterations: int | DiscreteHyperparameter | CategoricalHyperparameter,
        learning_rate: float | ContinuousHyperparameter | CategoricalHyperparameter,
        device="/CPU:0",
        validation_constraints: list[Constraint] = [],
        validation_check: int = 100,
    ):
        super().__init__(
            pipelines=pipelines,
            training_constraints=training_constraints,
            validation_constraints=validation_constraints,
            optimizer_cls=optimizer_cls,
            max_iterations=max_iterations,
            device=device,
            validation_check=validation_check,
        )
        self.lr = HyperParameter.from_value(learning_rate, "Learning Rate")
        self.optimizer: tf.keras.optimizers.Optimizer  # type: ignore

    def _get_trainable_parameters(self):
        trainable_parameters = []
        for node in self.all_nodes:
            node_params = node.trainable_parameters
            if node_params is not None:
                trainable_parameters.extend(node_params)
        return trainable_parameters

    def _move_to_device(self):
        # TensorFlow automatically handles devices,
        pass

    def run(self, show_progress: bool = True):
        with tf.device(self.device):
            all_pipelines = self.train_pipelines.union(self.validation_pipelines)
            for pipeline in all_pipelines:
                pipeline.setup()

            # Register trainable parameters
            trainable_parameters = self._get_trainable_parameters()
            self.optimizer = self.optimizer_cls(learning_rate=self.lr.value)

            for step in range(self.max_iterations.value):
                total_loss = 0.0
                # Run training pipelines inside GradientTape
                with tf.GradientTape() as tape:
                    for pipeline in self.train_pipelines:
                        # Assume _run_pipeline returns a scalar loss
                        self._run_pipeline(pipeline, EvaluationMode.TRAIN)

                    for train_constraint in self.training_constraints:
                        total_loss += train_constraint.get_loss()

                gradients = tape.gradient(total_loss, trainable_parameters)
                self.optimizer.apply_gradients(zip(gradients, trainable_parameters))
                # type: ignore

                # Check validation data
                if step % self.validation_check == 0:
                    for pipeline in self.validation_pipelines:
                        self._run_pipeline(pipeline, EvaluationMode.VALIDATION)

                    validation_loss = 0.0
                    for validation_constraint in self.validation_constraints:
                        validation_loss += validation_constraint.get_loss()

                    if show_progress:
                        print(
                            f"Training loss at {step}/{self.max_iterations.value}: \
                            {total_loss}"
                        )
                        print(
                            f"Validation loss at {step}/{self.max_iterations.value}:\
                            {validation_loss}"
                        )
