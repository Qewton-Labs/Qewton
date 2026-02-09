import torch

from ..base import EvaluationMode
from ..hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    ContinuousHyperparameter,
)
from ...pipeline.base import Pipeline
from .base import Trainer


class PyTorchTrainer(Trainer):
    def __init__(
        self,
        pipelines: list[Pipeline],
        optimizer,
        max_iterations: int | DiscreteHyperparameter,
        learning_rate: float | ContinuousHyperparameter,
        device="cpu",
        validation_check: int = 100,
    ):

        super().__init__(
            pipelines=pipelines,
            optimizer_cls=optimizer,
            max_iterations=max_iterations,
            device=device,
            validation_check=validation_check,
        )
        self.optimizer: torch.optim.Optimizer
        self.lr = HyperParameter.from_value(learning_rate, "Learning Rate")

    def _get_trainable_parameters(self):
        trainable_parameters = []
        for node in self.all_nodes:
            node_params = node.trainable_parameters
            if node_params is not None:
                trainable_parameters.append({"params": node_params})
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
        self.optimizer: torch.optim.Optimizer = self.optimizer_cls(
            trainable_parameters, lr=self.lr.current_value
        )
        # Start training loop
        for step in range(self.max_iterations.current_value):

            total_loss = 0.0
            # Run all pipelines that contain training constraints
            for pipeline in self.train_pipelines:
                total_loss += self._run_pipeline(pipeline, EvaluationMode.TRAIN)

            # Update parameters
            total_loss.backward()  # type: ignore
            self.optimizer.step()
            self.optimizer.zero_grad()

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
