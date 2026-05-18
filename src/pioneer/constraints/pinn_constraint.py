import inspect

from ..algorithms.building_blocks.norms import MSN

from ..config.axes import FeatureAxes, EllipsisAxes
from ..config.data_configurations import DataConfiguration
from ..config.variables import Variable

from ..graphs.graphs import Graph

from ..graphs.control_nodes.graph_node import GraphNode

from ..config.backend import DEFAULT_DL_BACKEND, Backend, TensorType
from ..constraints.base import ConstraintObjective, ConstraintType
from ..optim.base import EvaluationPhase
from ..optim.parameters.hyperparameter_base import HyperParameter
from ..graphs.nodes import Node, OutputPort

from .base import Constraint


class PINNConstraint(Constraint, GraphNode):

    def __init__(
        self,
        residual,
        reduction=MSN(),
        name="PINNConstraint",
        weight: float | HyperParameter = 1,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        constraint_type: ConstraintType = ConstraintType.LOSS,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        Constraint.__init__(self, weight, objective, constraint_type, evaluated_in_mode)
        self.backend = backend

        # construct residual node
        if isinstance(residual, Node):
            self.residual_node = residual
        else:
            assert callable(residual)
            residual_graph, residual_input_ports, residual_output_ports = (
                self.build_graph_from_function(residual)
            )

            sig = inspect.signature(residual).parameters.values()
            for var, p in zip(sig, residual_input_ports.keys()):
                if isinstance(var, Variable):
                    p.name = var.name
                    p.data_configuration = DataConfiguration(
                        EllipsisAxes(), FeatureAxes(var)
                    )
            print(residual_output_ports)
            assert (
                len(residual_output_ports) == 1
            ), "Residual functions should return a single value."
            self.residual_node = GraphNode(
                residual_graph,
                residual_input_ports,
                residual_output_ports,
                backend=backend,
            )

        # construct reduction node
        if isinstance(reduction, Node):
            self.reduction_node = reduction
        else:
            assert callable(reduction)
            reduction_graph, reduction_input_ports, reduction_output_ports = (
                self.build_graph_from_function(reduction)
            )
            assert len(reduction_input_ports) == 1, "Reduction takes only a single input."
            assert (
                len(reduction_output_ports) == 1
            ), "Reduction function should return a single value."
            self.reduction_node = GraphNode(
                reduction_graph,
                reduction_input_ports,
                reduction_output_ports,
                backend=backend,
            )

        # build own graph
        graph = Graph()
        graph.connect(self.residual_node, self.reduction_node)

        self.loss_port = OutputPort(
            DataConfiguration(FeatureAxes(shape=(1,)), dtype=backend.standard_datatype()),
            self,
            name="loss",
        )

        GraphNode.__init__(
            self,
            graph,
            self.residual_node.input_ports,
            {self.loss_port: self.reduction_node.output_ports[0]},
            name,
            backend=backend,
        )

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return GraphNode.hyperparameters.__get__(
            self, type(self)
        ) + Constraint.hyperparameters.__get__(self, type(self))
