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
from ...constraints.base import Constraint
from ...data.dataloaders.base import DataNode


class GraphBasedTrainer(Trainer):
    def __init__(
        self,
        optimization_phases: OptimizationPhase | list[OptimizationPhase],
        graphs: list[Graph],
        training_objectives: list[Constraint],
        callbacks: Callback | list[Callback] | None = None,
        validation_interval: int = 100,
        device="cpu",
        save_path: str = "train_results",
        progress_bar: ProgressBarCallback = ProgressBarCallback(),
        enable_logging=True,
        log_interval=100,
    ):
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        for train_obj in training_objectives:
            assert train_obj.evaluated_in_mode in [
                EvaluationPhase.TRAIN,
                EvaluationPhase.ALWAYS,
            ], f"Train objective {train_obj.name} is in mode \
                {train_obj.evaluated_in_mode}."

        # First find all nodes from all graphs (without duplicates)
        self.graphs = graphs
        self.all_nodes = set[Node]()
        for graph in graphs:
            for node in graph.nodes:
                self.all_nodes.add(node)

        # For all constraints check if they belong to some graph and order them
        self.training_objectives = training_objectives
        self.training_constraints: set[Constraint] = set(training_objectives)
        self.train_graphs = self._find_training_graphs(training_objectives)

        # Now also find all other constraints inside the graphs and
        # register them as validation constraints.
        self.validation_graphs = set[Graph]()
        self.validation_constraints = set[Constraint]()
        self._find_all_constraints()

        # Add callbacks that evaluate the graphs
        train_callback = GraphEvalCallback(
            self.train_graphs, EvaluationPhase.TRAIN, self.training_constraints
        )
        callbacks.append(train_callback)
        if len(self.validation_constraints) > 0:
            validation_callback = GraphEvalCallback(
                self.validation_graphs,
                EvaluationPhase.VALIDATION,
                self.validation_constraints,
                evaluation_interval=validation_interval,
            )
            callbacks.append(validation_callback)

        super().__init__(
            optimization_phases=optimization_phases,
            callbacks=callbacks,
            hyperparameters=self.collect_graph_hyperparameters(),
            device=device,
            save_path=save_path,
            progress_bar=progress_bar,
            enable_logging=enable_logging,
            log_interval=log_interval,
        )

    def _find_all_constraints(self):
        for graph in self.graphs:
            constraint_list: list[Constraint] = []

            constraint_modes: dict[EvaluationPhase, bool] = {
                EvaluationPhase.TRAIN: False,
                EvaluationPhase.VALIDATION: False,
                EvaluationPhase.TEST: False,
                EvaluationPhase.ALWAYS: False,
                EvaluationPhase.NEVER: False,
            }
            data_modes: dict[EvaluationPhase, bool] = {
                EvaluationPhase.TRAIN: True,
                EvaluationPhase.VALIDATION: True,
                EvaluationPhase.TEST: True,
                EvaluationPhase.ALWAYS: True,
            }

            for node in graph.nodes:
                if isinstance(node, Constraint):
                    constraint_modes[node.evaluated_in_mode] = True
                    constraint_list.append(node)
                if isinstance(node, DataNode):
                    for mode, active in data_modes.items():
                        data_modes[mode] = active & node.provides_data_in_phase(mode)

            if data_modes[EvaluationPhase.ALWAYS]:
                # We have data for all phases, so we can evaluate all constraints in
                # all phases
                pass
            else:
                # We need to check that for each constraint, we have data in the
                # corresponding phase, otherwise we cannot evaluate this
                # constraint at all and need to raise an error.
                phases = [
                    EvaluationPhase.TRAIN,
                    EvaluationPhase.VALIDATION,
                    EvaluationPhase.TEST,
                ]
                for eval_phase in phases:
                    if constraint_modes[eval_phase] and not data_modes[eval_phase]:
                        raise ValueError(
                            f"The graph {graph} does not \
                            provide data for evaluation phase {eval_phase}, \
                            but there are constraints in this graph that are \
                            evaluated in this phase."
                        )

            self._register_constraints(graph, constraint_list, data_modes)

    def _register_constraints(
        self,
        graph: Graph,
        constraint_list: list[Constraint],
        data_modes: dict[EvaluationPhase, bool],
    ):
        for con in constraint_list:
            phase = con.evaluated_in_mode

            add_train = False
            add_val = False

            if phase == EvaluationPhase.ALWAYS:
                if data_modes[EvaluationPhase.ALWAYS]:
                    add_train = add_val = True
                else:
                    add_train = data_modes[EvaluationPhase.TRAIN]
                    add_val = data_modes[EvaluationPhase.VALIDATION]

            elif phase == EvaluationPhase.TRAIN:
                add_train = True

            elif phase == EvaluationPhase.VALIDATION:
                add_val = True

            if add_train:
                self.training_constraints.add(con)
                self.train_graphs.add(graph)

            if add_val:
                self.validation_constraints.add(con)
                self.validation_graphs.add(graph)

    def _find_training_graphs(self, constraints: list[Constraint]) -> set[Graph]:
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

    def check_tuning_constraints_exist(self, constraints: list[Constraint]):
        for c in constraints:
            assert (
                c in self.validation_constraints or c in self.training_constraints
            ), f"Constraint {c} is not part of the training or validation constraints \
                  of this trainer."

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
        constraints_list = [self.training_constraints, self.validation_constraints]
        evaluation_phases = [
            EvaluationPhase.TRAIN,
            EvaluationPhase.VALIDATION,
        ]
        for constraints, eval_phase in zip(constraints_list, evaluation_phases):
            for constraint in constraints:
                self.train_state.losses[eval_phase][constraint.name] = None
