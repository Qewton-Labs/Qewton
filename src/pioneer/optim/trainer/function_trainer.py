from typing import Any, Callable

from .base_trainer import Trainer
from .training_controllers import TrainerState
from .callbacks.base_callback import Callback
from .callbacks.training_callbacks import FunctionEvalCallback
from .callbacks.progressbar_callback import ProgressBarCallback
from .training_controllers import OptimizationPhase
from ..parameters.hyperparameter_base import HyperParameter
from ..base import EvaluationPhase
from ...graphs.nodes import Node


class FunctionBasedTrainer(Trainer):

    def __init__(
        self,
        optimization_phases: OptimizationPhase | list[OptimizationPhase],
        training_functions: list[Callable[[int, TrainerState], Any]],
        model_nodes: list[Node],
        hyperparameters: set[HyperParameter] | None = None,
        validation_functions: list[Callable[[int, TrainerState], Any]] | None = None,
        validation_interval: int = 100,
        callbacks: Callback | list[Callback] | None = None,
        device="cpu",
        save_path: str = "train_results",
        progress_bar: ProgressBarCallback = ProgressBarCallback(),
    ) -> None:
        if hyperparameters is None:
            hyperparameters = set[HyperParameter]()
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]

        self.model_nodes = model_nodes
        self.training_functions = training_functions
        if validation_functions is None:
            validation_functions = []
        self.validation_functions = validation_functions
        self.tuning_functions = self.validation_functions

        # Add functions to be evaluated in training
        callbacks.append(
            FunctionEvalCallback(
                self.training_functions,
                EvaluationPhase.TRAIN,
            )
        )
        if self.validation_functions:
            callbacks.append(
                FunctionEvalCallback(
                    self.validation_functions,
                    EvaluationPhase.VALIDATION,
                    evaluation_interval=validation_interval,
                )
            )

        super().__init__(
            optimization_phases=optimization_phases,
            callbacks=callbacks,
            hyperparameters=hyperparameters,
            device=device,
            save_path=save_path,
            progress_bar=progress_bar,
        )

    def set_tuning_constraints(
        self, constraints: list[Callable[[int, TrainerState], Any]]
    ):
        self.tuning_functions = constraints

    def on_training_start(self):
        # setup all graphs and then move all nodes to the correct device:
        for node in self.model_nodes:
            node.setup()
            node.to(self.device)
        # Now also collect all trainable parameters inside the graph
        parameters = None
        for node in self.model_nodes:
            node_params = node.trainable_parameters
            if parameters is None:
                parameters = node_params
            elif not node_params.empty:
                parameters = parameters.combine(node_params)
        if parameters is None:
            raise ValueError("Did not find any trainable parameters in this setup!")
        self.set_trainable_parameters(parameters=parameters)

        return super().on_training_start()

    def populate_state_dict(self):
        """Collect all relevant loss and metric names into the state dict, to
        know at the start of training which values are to be expected."""
        constraints_list = [
            self.training_functions,
            self.validation_functions,
            self.tuning_functions,
        ]
        evaluation_phases = [
            EvaluationPhase.TRAIN,
            EvaluationPhase.VALIDATION,
            EvaluationPhase.TUNE,
        ]
        for constraints, eval_phase in zip(constraints_list, evaluation_phases):
            for constraint in constraints:
                self.train_state.losses[eval_phase][constraint.__name__] = 0.0

    def evaluate_tuning_constraints(self):
        # Evaluate all graphs that have some tuning constraint
        for constraint in self.tuning_functions:
            loss = constraint(0, self.train_state)
            self.train_state.losses[EvaluationPhase.TUNE][constraint.__name__] = loss
