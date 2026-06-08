from qewton.constraints.base import Constraint, ConstraintObjective
from qewton.config.data_configurations import DataConfiguration
from qewton.optim.parameters.categorical_hyperparameter import (
    HyperParameter,
    BooleanHyperparameter,
)
from qewton.optim.parameters.number_hyperparameter import ContinuousHyperparameter
from qewton.graphs.nodes import InputPort
from qewton.graphs.graphs import Graph
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.algorithms.building_blocks.math import Subtract, Square, Mean, Divide, Sum
from qewton.optim.base import EvaluationPhase


class MetricConstraint(Constraint):
    def __init__(
        self,
        input_config: DataConfiguration = DataConfiguration.empty(),
        name="MetricConstraint",
        relative: bool | BooleanHyperparameter = False,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        weight: float | ContinuousHyperparameter = 1,
        backend=DEFAULT_DL_BACKEND,
        epsilon=1e-8,
        **kwargs,
    ):
        super().__init__(
            name=name,
            weight=weight,
            backend=backend,
            objective=ConstraintObjective.MINIMIZE,
            evaluated_in_mode=evaluated_in_mode,
            **kwargs,
        )
        self.relative = HyperParameter.from_value(relative, "Relative Constraint")
        self.epsilon = epsilon  # for computation of the relative loss.

        if len(self._input_ports) == 0:
            self._build_input_ports(input_config)

    def _build_input_ports(self, input_config: DataConfiguration):
        if not hasattr(self, "_input_ports"):
            self.input_1 = InputPort(input_config, self, name="input1")
            self.input_2 = InputPort(input_config, self, name="input2")
            self._input_ports = [self.input_1, self.input_2]


class MSEConstraint(MetricConstraint, GraphNode):
    # TODO: Add different kind of norms

    def __init__(
        self,
        name="MSEConstraint",
        relative: bool | BooleanHyperparameter = False,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        weight: float | ContinuousHyperparameter = 1,
        backend=DEFAULT_DL_BACKEND,
        epsilon=1e-8,
    ):
        self.subtract_node = Subtract(backend=backend)
        self.sum_node = Sum(backend=backend, axis=-1)
        self.square_node = Square(backend=backend)

        self.sum_node_relative = Sum(backend=backend, axis=-1)
        self.square_node_relative = Square(backend=backend)
        self.divide_node = Divide(backend=backend)

        self.mean_node = Mean(backend=backend)

        new_graph, in_ports, out_ports = self._build_graph(
            relative.value if isinstance(relative, BooleanHyperparameter) else relative
        )

        super().__init__(
            graph=new_graph,
            weight=weight,
            relative=relative,
            input_ports=in_ports,
            output_ports=out_ports,
            name=name,
            backend=backend,
            epsilon=epsilon,
            evaluated_in_mode=evaluated_in_mode,
        )

        self._graph.setup()

    def _build_graph(self, use_relative: bool):
        new_graph = Graph()
        new_graph.connect(self.subtract_node, self.square_node)
        new_graph.connect(self.square_node, self.sum_node)
        if use_relative:
            new_graph.connect(self.square_node_relative, self.sum_node_relative)
            new_graph.connect(self.sum_node_relative, self.divide_node.input_ports[1])
            new_graph.connect(self.sum_node, self.divide_node.input_ports[0])
            new_graph.connect(self.divide_node, self.mean_node)
        else:
            new_graph.connect(self.sum_node, self.mean_node)

        self._build_input_ports(self.subtract_node.input_ports[0].data_configuration)

        in_ports = {
            self.input_1: [self.subtract_node.input_ports[0]],
            self.input_2: [self.subtract_node.input_ports[1]],
        }
        if use_relative:
            in_ports[self.input_2].append(self.square_node_relative.input_ports[0])

        return new_graph, in_ports, self.mean_node.output_ports

    def setup(self):
        new_graph, in_ports, out_ports = self._build_graph(self.relative.value)
        self.setup_graph(new_graph, input_ports=in_ports, output_ports=out_ports)
