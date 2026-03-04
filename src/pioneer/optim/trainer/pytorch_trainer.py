import torch

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


class PyTorchTrainer(Trainer):
    def __init__(
        self,
        pipelines: list[Pipeline],
        training_constraints: list[Constraint],
        optimizer,
        max_iterations: int | DiscreteHyperparameter | CategoricalHyperparameter,
        learning_rate: float | ContinuousHyperparameter | CategoricalHyperparameter,
        device="cpu",
        validation_constraints: list[Constraint] = [],
        validation_check: int = 100,
        save_path="pytorch_trainer",
    ):

        super().__init__(
            pipelines=pipelines,
            training_constraints=training_constraints,
            validation_constraints=validation_constraints,
            optimizer_cls=optimizer,
            max_iterations=max_iterations,
            device=device,
            validation_check=validation_check,
            save_path=save_path,
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

    def run(self, show_progress: bool = True):
        # Setup all data inside the problem and create models
        all_pipelines = self.train_pipelines.union(self.validation_pipelines)
        for pipeline in all_pipelines:
            pipeline.setup()
        # Register trainable parameters
        self._move_to_device()
        trainable_parameters = self._get_trainable_parameters()
        self.optimizer: torch.optim.Optimizer = self.optimizer_cls(
            trainable_parameters, lr=self.lr.value
        )
        # Start training loop
        for step in range(self.max_iterations.value):

            # Run all pipelines and compute the training loss
            total_loss = self._compute_loss(
                self.train_pipelines, EvaluationMode.TRAIN, self.training_constraints
            )

            # Update parameters
            total_loss.backward()  # type: ignore
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Check validation data
            if step % self.validation_check == 0:
                validation_loss = self._compute_loss(
                    self.validation_pipelines,
                    EvaluationMode.VALIDATION,
                    self.validation_constraints,
                )

                if show_progress:
                    print(
                        f"Training loss at {step}/{self.max_iterations.value}: \
                        {total_loss}"
                    )
                    if len(self.validation_constraints) > 0:
                        print(
                            f"Validation loss at {step}/{self.max_iterations.value}:\
                            {validation_loss}"
                        )

    def _evaluate_constraints(
        self, constraint_list: list[Constraint]
    ) -> dict[str, float]:

        pipeline_loss_dict: dict[str, float] = {}
        for constraint_node in constraint_list:
            loss = constraint_node.get_loss().detach().item()  # type: ignore
            pipeline_loss_dict[constraint_node.name] = loss
        return pipeline_loss_dict
