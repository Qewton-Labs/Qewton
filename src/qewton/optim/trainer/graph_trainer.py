"""Trainer that trains models defined as graphs."""

import warnings

from qewton.optim.trainer.base_trainer import Trainer
from qewton.optim.trainer.callbacks.base_callback import Callback
from qewton.optim.trainer.callbacks.progressbar_callback import ProgressBarCallback
from qewton.optim.trainer.callbacks.training_callbacks import GraphEvalCallback
from qewton.optim.trainer.training_controllers import OptimizationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.base import EvaluationPhase
from qewton.graphs import Graph
from qewton.graphs.nodes import Node
from qewton.constraints.base import Constraint
from qewton.config.devices import Device, cpu
from qewton.data.dataloaders.base import DataNode


class GraphBasedTrainer(Trainer):
    """Trainer that evaluates graphs to train the models inside them.

    The trainer collects the constraints of all graphs. Each graph
    containing a training objective is evaluated in the training phase.
    Other constraints are assigned to training and/or validation according
    to their ``evaluated_in_mode`` and the phases for which the graph's data
    nodes provide data. Constraints with the same name are renamed
    (``"PDE"``, ``"PDE_2"``, ...) in the order of `training_objectives`,
    then of creation, since losses are recorded by constraint name.

    Args:
        optimization_phases: One or more optimization phases defining the
            training workflow.
        graphs: All graphs considered by this trainer.
        training_objectives: The constraints whose losses are optimized. Their
            ``evaluated_in_mode`` must be ``TRAIN`` or ``ALWAYS``.
        callbacks: Additional callbacks.
        validation_interval: The number of iterations between validation
            evaluations.
        device: The device to train on, e.g. ``"cpu"`` or ``"cuda"``.
        save_path: Directory or file path where training results are stored.
        progress_bar: The progress bar callback. If ``None``, a default
            :class:`~qewton.optim.trainer.callbacks.progressbar_callback.ProgressBarCallback`
            is used.
        enable_logging (bool, optional): Whether training logs are recorded.
            Defaults to ``True``.
        log_interval (int, optional): The number of iterations between log
            outputs. Defaults to 100.

    Raises:
        AssertionError: If a training objective is not evaluated in the
            training phase.
        ValueError: If a graph has constraints for a phase in which its data
            nodes provide no data.

    Attributes:
        graphs (list of Graph): All graphs of this trainer.
        all_nodes (set of Node): All nodes of all graphs.
        training_objectives (list of Constraint): The optimized constraints.
        training_constraints (set of Constraint): The constraints evaluated in
            the training phase.
        validation_constraints (set of Constraint): The constraints evaluated
            in the validation phase.
        train_graphs (set of Graph): The graphs evaluated in the training
            phase.
        validation_graphs (set of Graph): The graphs evaluated in the
            validation phase.
    """

    def __init__(
        self,
        optimization_phases: OptimizationPhase | list[OptimizationPhase],
        graphs: list[Graph],
        training_objectives: list[Constraint],
        callbacks: Callback | list[Callback] | None = None,
        validation_interval: int = 100,
        device: str | Device = cpu,
        save_path: str = "train_results",
        progress_bar: ProgressBarCallback | None = None,
        enable_logging=True,
        log_interval=100,
    ):
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        # Duplicate names are made unique by _make_constraint_names_unique()
        # below, once all constraints are known.
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
        self._make_constraint_names_unique()

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
                        raise ValueError(f"The graph {graph} does not \
                            provide data for evaluation phase {eval_phase}, \
                            but there are constraints in this graph that are \
                            evaluated in this phase.")

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

    def _make_constraint_names_unique(self):
        # Renames constraints sharing a name by appending an index ("PDE",
        # "PDE_2", ...), in the order of `training_objectives`, then of
        # creation (node_id).
        in_graphs = sorted(
            (n for graph in self.graphs for n in graph.nodes if isinstance(n, Constraint)),
            key=lambda n: n.node_id,
        )
        ordered: list[Constraint] = []
        for constraint in [*self.training_objectives, *in_graphs]:
            if not any(constraint is c for c in ordered):
                ordered.append(constraint)
        taken: set[str] = set()
        for constraint in ordered:
            name, index = constraint.name, 2
            while name in taken:
                name, index = f"{constraint.name}_{index}", index + 1
            constraint.name = name
            taken.add(name)

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
        """Check that the constraints are evaluated by this trainer.

        Args:
            constraints: The constraints to check.

        Raises:
            AssertionError: If a constraint is neither a training nor a
                validation constraint.
        """
        for c in constraints:
            assert (
                c in self.validation_constraints or c in self.training_constraints
            ), f"Constraint {c} is not part of the training or validation constraints \
                  of this trainer."

    def collect_graph_hyperparameters(self) -> set[HyperParameter]:
        """Collect the hyperparameters of all nodes.

        Returns:
            set of HyperParameter: The hyperparameters of all nodes of all
                graphs.
        """
        hyperparameter_set = set[HyperParameter]()
        for node in self.all_nodes:
            node_params = node.hyperparameters
            for hp in node_params:
                hyperparameter_set.add(hp)
        return hyperparameter_set

    def on_training_start(self):
        """Prepare the graphs for training.

        Sets up all graphs, moves all nodes to the training device and
        registers the trainable parameters of all nodes.

        Raises:
            ValueError: If no node has trainable parameters.
        """
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
        """Register the loss names of all training and validation
        constraints in the train state, so the expected values are known
        at the start of training."""
        constraints_list = [self.training_constraints, self.validation_constraints]
        evaluation_phases = [
            EvaluationPhase.TRAIN,
            EvaluationPhase.VALIDATION,
        ]
        for constraints, eval_phase in zip(constraints_list, evaluation_phases):
            for constraint in constraints:
                self.train_state.losses[eval_phase][constraint.name] = None
