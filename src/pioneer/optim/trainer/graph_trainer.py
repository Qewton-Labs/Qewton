import warnings

from .base_trainer import Trainer
from .callbacks.base_callback import Callback
from .callbacks.progressbar_callback import ProgressBarCallback
from .callbacks.training_callbacks import GraphEvalCallback
from .training_controllers import OptimizationPhase
from ..parameters.hyperparameter_base import HyperParameter
from ..base import EvaluationPhase
from ...graphs import Graph
from ...graphs.nodes import Node
from ...constraints.base import Constraint, ConstraintType


class GraphBasedTrainer(Trainer):
    def __init__(
        self,
        optimization_phases: OptimizationPhase | list[OptimizationPhase],
        graphs: list[Graph],
        training_constraints: list[Constraint],
        validation_constraints: list[Constraint] | None = None,
        callbacks: Callback | list[Callback] | None = None,
        validation_interval: int = 100,
        device="cpu",
        save_path: str = "train_results",
        progress_bar: ProgressBarCallback = ProgressBarCallback(),
    ):
        if callbacks is None:
            callbacks = []
        if validation_constraints is None:
            validation_constraints = []
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]

        # First find all nodes from all graphs (without duplicates)
        self.graphs = graphs
        self.all_nodes = set[Node]()
        for graph in graphs:
            for node in graph.nodes:
                self.all_nodes.add(node)

        # For all constraints check if they belong to some graph and order them
        self.tune_graphs = set[Graph]()
        self.training_constraints = training_constraints
        self.validation_constraints = validation_constraints
        self.tuning_constraints: list[Constraint] = []
        self.train_graphs = self._register_graphs(training_constraints)
        self.validation_graphs = self._register_graphs(validation_constraints)

        # Add callbacks that evaluate the graphs
        train_callback = GraphEvalCallback(
            self.train_graphs, EvaluationPhase.TRAIN, training_constraints
        )
        callbacks.append(train_callback)
        if len(validation_constraints) > 0:
            validation_callback = GraphEvalCallback(
                self.validation_graphs,
                EvaluationPhase.VALIDATION,
                validation_constraints,
                evaluation_interval=validation_interval,
            )
            callbacks.append(validation_callback)
        # by default tuning = validation
        self.set_tuning_constraints(validation_constraints)

        super().__init__(
            optimization_phases=optimization_phases,
            callbacks=callbacks,
            hyperparameters=self.collect_graph_hyperparameters(),
            device=device,
            save_path=save_path,
            progress_bar=progress_bar,
        )

    def _register_graphs(self, constraints: list[Constraint]) -> set[Graph]:
        found_constraints = set()
        graph_set = set[Graph]()
        for graph in self.graphs:
            for constraint_node in graph.nodes:
                if constraint_node not in constraints:
                    continue

                found_constraints.add(constraint_node)
                graph_set.add(graph)

        # Check for missing constraints
        missing = set(constraints) - found_constraints
        if missing:
            warnings.warn(
                f"The Trainer has no graphs containing constraints {missing}",
                RuntimeWarning,
            )
        return graph_set

    def set_tuning_constraints(self, constraints: list[Constraint]):
        self.tuning_constraints = constraints
        self.tune_graphs = self._register_graphs(self.tuning_constraints)

    def collect_graph_hyperparameters(self) -> set[HyperParameter]:
        hyperparameter_set = set[HyperParameter]()
        for node in self.all_nodes:
            node_params = node.hyperparameters
            for hp in node_params:
                hyperparameter_set.add(hp)
        return hyperparameter_set

    def on_training_start(self):
        # setup all graphs and then move all nodes to the correct device:
        for graph in self.graphs:
            graph.setup()
        for node in self.all_nodes:
            node.to(self.device)
        # Now also collect all trainable parameters inside the graph
        parameters = None
        for node in self.all_nodes:
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
            self.training_constraints,
            self.validation_constraints,
            self.tuning_constraints,
        ]
        evaluation_phases = [
            EvaluationPhase.TRAIN,
            EvaluationPhase.VALIDATION,
            EvaluationPhase.TUNE,
        ]
        for constraints, eval_phase in zip(constraints_list, evaluation_phases):
            for constraint in constraints:
                if constraint.constraint_type == ConstraintType.LOSS:
                    self.train_state.losses[eval_phase][constraint.name] = 0.0
                elif constraint.constraint_type == ConstraintType.MONITOR:
                    self.train_state.metrics[eval_phase][constraint.name] = 0.0

    def evaluate_tuning_constraints(self):
        # Evaluate all graphs that have some tuning constraint
        for graph in self.tune_graphs:
            graph.run(EvaluationPhase.TUNE)

        # Write out the loss
        for constraint in self.tuning_constraints:
            if constraint.constraint_type == ConstraintType.LOSS:
                self.train_state.losses[EvaluationPhase.TUNE][constraint.name] = (
                    constraint.get_loss(add_weight=False)
                )
            elif constraint.constraint_type == ConstraintType.MONITOR:
                self.train_state.metrics[EvaluationPhase.TUNE][constraint.name] = (
                    constraint.get_loss(add_weight=False)
                )
