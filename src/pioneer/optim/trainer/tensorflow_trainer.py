import tensorflow as tf

from ..base import EvaluationMode
from ..hyperparameter.base import HyperParameter, DiscreteHyperparameter, ContinuousHyperparameter
from ...pipeline.base import Pipeline
from .base import Trainer

class TensorFlowTrainer(Trainer):
    def __init__(
        self,
        pipelines: list[Pipeline],
        optimizer_cls: type[tf.keras.optimizers.Optimizer], # type: ignore
        max_iterations: int | DiscreteHyperparameter,
        learning_rate: float | ContinuousHyperparameter,
        device="/CPU:0",
        validation_check: int = 100,
    ):
        super().__init__(
            pipelines=pipelines,
            optimizer_cls=optimizer_cls,
            max_iterations=max_iterations,
            device=device,
            validation_check=validation_check
        )
        self.lr = HyperParameter.from_value(learning_rate, "Learning Rate")
        self.optimizer: tf.keras.optimizers.Optimizer # type: ignore

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

    def run(self):
        with tf.device(self.device):
            all_pipelines = self.train_pipelines.union(self.validation_pipelines)
            all_pipelines = all_pipelines.union(self.test_pipelines)
            for pipeline in all_pipelines:
                pipeline.setup()

            # Register trainable parameters
            trainable_parameters = self._get_trainable_parameters()
            self.optimizer = self.optimizer_cls(learning_rate=self.lr.current_value)

            for step in range(self.max_iterations.current_value):
                total_loss = 0.0
                # Run training pipelines inside GradientTape
                with tf.GradientTape() as tape:
                    for pipeline in self.train_pipelines:
                        # Assume _run_pipeline returns a scalar loss
                        total_loss += self._run_pipeline(pipeline, EvaluationMode.TRAIN)

                gradients = tape.gradient(total_loss, trainable_parameters)
                self.optimizer.apply_gradients(zip(gradients, trainable_parameters)) # type: ignore

                # Validation
                if step % self.validation_check == 0:
                    validation_loss = 0.0
                    for pipeline in self.validation_pipelines:
                        validation_loss += self._run_pipeline(
                            pipeline, EvaluationMode.VALIDATION
                        )
                    print(
                        f"Training loss at {step}/{self.max_iterations.current_value}: {total_loss}"
                    )
                    print(
                        f"Validation loss at {step}/{self.max_iterations.current_value}: {validation_loss}"
                    )

            test_loss = 0.0
            for pipeline in self.test_pipelines:
                test_loss += self._run_pipeline(pipeline, EvaluationMode.TEST)
            print("Final testing loss is", test_loss)
