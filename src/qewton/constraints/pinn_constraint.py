"""Physics-informed constraint defined by a residual function."""

import inspect

from qewton.algorithms.building_blocks.norms import MSN

from qewton.config.axes import FeatureAxes, EllipsisAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable

from qewton.constraints.weight_functions import ConstraintWeightFunction
from qewton.graphs.control_nodes.wrapper_node import FunctionWrappingNode
from qewton.graphs.graphs import Graph

from qewton.graphs.control_nodes.graph_node import GraphNode, FromFunctionNode

from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.constraints.base import ConstraintObjective
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.nodes import Node

from qewton.constraints.base import Constraint


class PINNConstraint(Constraint, GraphNode):
    """Physics-informed constraint defined by a residual function.

    The residual function computes the residual of the PDE from its inputs;
    a reduction function then reduces it to a single value, by default the
    mean squared norm. Both are wrapped as nodes in the internal graph of
    this :class:`~qewton.graphs.control_nodes.graph_node.GraphNode`.

    The arguments of the residual function are annotated with variables.
    Each annotated argument becomes an input port named after the argument,
    expecting data with that variable as feature axis::

        def residual(x: Variable("x"), u: Variable("u")):
            return u.gradient(x) - f(x)

    Args:
        residual (callable or Node): The residual function, or a node computing
            the residual. Must have a single output.
        reduction (callable or Node, optional): Reduces the residual to a
            single value. Must have a single input and a single output.
            Defaults to :class:`~qewton.algorithms.building_blocks.norms.MSN`.
        name: The name under which the loss is recorded. Defaults to the name
            of the residual function, or ``"PINNConstraint"`` for a lambda or a
            node.
        weight: A weight the loss is multiplied with.
        activation_condition: Maps the training iteration to an additional
            weight factor. If ``None``, the constraint is always active.
        track_residual: If ``True``, the residual function is traced into a
            graph
            (:class:`~qewton.graphs.control_nodes.graph_node.FromFunctionNode`);
            otherwise it is called directly as an opaque function
            (:class:`~qewton.graphs.control_nodes.wrapper_node.FunctionWrappingNode`).
            Ignored if `residual` is a node.
        objective: Whether the constraint is minimized or maximized.
        evaluated_in_mode: The evaluation phase in which the constraint is
            evaluated.
        backend: The backend used for computations.

    Raises:
        AssertionError: If the residual has more than one output, or the
            reduction does not have exactly one input and one output.

    Attributes:
        residual_node (Node): The node computing the residual.
        reduction_node (Node): The node reducing the residual to the loss.
        track_residual (bool): Whether the residual function is traced.
    """

    def __init__(
        self,
        residual,
        reduction=None,
        name: str | None = None,
        weight: float | HyperParameter = 1,
        activation_condition: ConstraintWeightFunction | None = None,
        track_residual: bool = True,
        objective: ConstraintObjective = ConstraintObjective.MINIMIZE,
        evaluated_in_mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.track_residual = track_residual
        if name is None:
            name = getattr(residual, "__name__", "<lambda>")
            if isinstance(residual, Node) or name == "<lambda>":
                name = "PINNConstraint"

        if reduction is None:
            reduction = MSN()
        # construct residual node
        if isinstance(residual, Node):
            self.residual_node = residual
        else:
            assert callable(residual)
            node_cls = FromFunctionNode if track_residual else FunctionWrappingNode
            self.residual_node = node_cls(residual, backend=backend)

            sig = inspect.signature(residual).parameters.values()
            for var, p in zip(sig, self.residual_node.input_ports):
                if isinstance(var.annotation, Variable):
                    p.name = var.name
                    p.update_static_data_configuration(
                        DataConfiguration(
                            EllipsisAxes(),
                            FeatureAxes(var.annotation),
                            dtype=backend.default_dtype,
                        )
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
            activation_condition=activation_condition,
            name=name,
            backend=backend,
        )
