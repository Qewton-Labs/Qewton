import inspect

from ..algorithms.building_blocks.norms import MSN

from ..config.axes import FeatureAxes, EllipsisAxes
from ..config.data_configurations import DataConfiguration
from ..config.variables import Variable

from ..graphs.graphs import Graph

from ..graphs.control_nodes.graph_node import GraphNode, FromFunctionNode

from ..config.backend import DEFAULT_DL_BACKEND, Backend, TensorType
from ..constraints.base import ConstraintObjective, ConstraintType
from ..optim.base import EvaluationPhase
from ..optim.parameters.hyperparameter_base import HyperParameter
from ..graphs.nodes import Node

from .base import Constraint


class PINNConstraint(Constraint, GraphNode):

    def __init__(
        self,
        residual,
        reduction=None,
        name="PINNConstraint",
        weight: float | HyperParameter = 1,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        constraint_type: ConstraintType = ConstraintType.LOSS,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        if reduction is None:
            reduction = MSN()
        # construct residual node
        if isinstance(residual, Node):
            self.residual_node = residual
        else:
            assert callable(residual)
            self.residual_node = FromFunctionNode(residual, backend=backend)

            sig = inspect.signature(residual).parameters.values()
            for var, p in zip(sig, self.residual_node.input_ports):
                if isinstance(var, Variable):
                    p.name = var.name
                    p.data_configuration = DataConfiguration(
                        EllipsisAxes(), FeatureAxes(var)
                    )
        assert (
            len(self.residual_node.output_ports) == 1
        ), "Residual functions should return a single value."

        # construct reduction node
        if isinstance(reduction, Node):
            self.reduction_node = reduction
        else:
            assert callable(reduction)
            self.reduction_node = FromFunctionNode(reduction, backend=backend)
        assert (
            len(self.reduction_node.input_ports) == 1
        ), "Reduction takes only a single input."
        assert (
            len(self.reduction_node.output_ports) == 1
        ), "Reduction function should return a single value."

        # build own graph
        graph = Graph()
        graph.connect(self.residual_node, self.reduction_node)

        super().__init__(
            graph=graph,
            weight=weight,
            objective=objective,
            constraint_type=constraint_type,
            evaluated_in_mode=evaluated_in_mode,
            input_ports=self.residual_node.input_ports,
            output_ports=self.reduction_node.output_ports,
            name=name,
            backend=backend,
        )
