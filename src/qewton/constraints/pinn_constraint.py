import inspect

from qewton.algorithms.building_blocks.norms import MSN

from qewton.config.axes import FeatureAxes, EllipsisAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable

from qewton.graphs.graphs import Graph

from qewton.graphs.control_nodes.graph_node import GraphNode, FromFunctionNode

from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.constraints.base import ConstraintObjective
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.nodes import Node

from qewton.constraints.base import Constraint


class PINNConstraint(Constraint, GraphNode):
    """Represents a physics informed constraint, which is defined by a
    residual function. The residual function takes the necessary data as input
    and computes the residual of the PDE, which is then reduced to a
    single value by a reduction function, e.g. the mean squared error of
    the residual.

    The residual function should use variables as a type annotation for its
    arguments, which will be used to automatically construct the input
    ports of this constraint and to feed the correct data into the residual
    function. E.g.

        def residual(x: Variable("x"), u: Variable("u")):
            return u.gradient(x) - f(x)

    Args:
        residual (Callable): The residual function, which computes the
            residual of the physics conditions.
        reduction (_type_, optional): A custom reduction function.
            Defaults to None and the MSE.

    Raises:
        AssertionError: If the residual function does not return a single value.
        AssertionError: If the reduction function does not take a single input.
        AssertionError: If the reduction function does not return a single value.
    """

    def __init__(
        self,
        residual,
        reduction=None,
        name="PINNConstraint",
        weight: float | HyperParameter = 1,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
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
                if isinstance(var.annotation, Variable):
                    p.name = var.name
                    p.data_configuration = DataConfiguration(
                        EllipsisAxes(),
                        FeatureAxes(var.annotation),
                        dtype=backend.default_dtype,
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
            evaluated_in_mode=evaluated_in_mode,
            input_ports=self.residual_node.input_ports,
            output_ports=self.reduction_node.output_ports,
            name=name,
            backend=backend,
        )
