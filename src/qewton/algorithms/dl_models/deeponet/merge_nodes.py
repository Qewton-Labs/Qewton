from typing import Literal

from qewton.algorithms.building_blocks.math import Inner, Multiply, Sum
from qewton.algorithms.building_blocks.array_operations import ReshapeAtDim, Unsqueeze
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.graphs.nodes import NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.config.variables import Variable


class DefaultMerger(GraphNode[TensorType]):
    """A default merger node that merges the outputs of the trunk and branch networks
    in a DeepONet architecture. The merging is done by multiplying the outputs of
    the trunk and branch networks and then summing over the last dimension.

    Args:
        output_dim (int | Variable): The dimension of the final output of the DeepONet.
            If the output dimension is greater than 1, the merge node will
            be configured following the specified output strategy.
        intermediate_neurons (int | HyperParameter): The number of intermediate neurons
            in the trunk and branch networks.
        output_strategy (Literal["split", "split_branch", "split_trunk"], optional):
            The strategy that is used to merge the branch and trunk net.
            In case of a single output, the merge node will be an Inner product node.
            In case of multiple outputs, the merge node will be a graph that
            reshapes the outputs of the branch and trunk networks and then performs
            an element-wise multiplication followed by a summation. The reshaping
            is done according to the specified output strategy:
            - "split": Both the branch and trunk networks will have
              output_dim * intermediate_neurons neurons in the last layer, which will
              be reshaped to (output_dim, intermediate_neurons) before merging.
            - "split_branch": Only the branch network will have
              output_dim * intermediate_neurons, the trunk net output will be
              multiplied with all branch net outputs.
            - "split_trunk": Only the trunk network will have
              output_dim * intermediate_neurons, the branch net output will be
              multiplied with all trunk net outputs.
            Defaults to "split".
        backend (type[DeepLearningBackend[TensorType]]): The deep learning backend to use
            for the merger node. Defaults to DEFAULT_DL_BACKEND.
    """

    MERGE_TYPES = Literal["split", "split_branch", "split_trunk"]

    def __init__(
        self,
        output_dim: int | Variable,
        intermediate_neurons: int | HyperParameter,
        output_strategy: MERGE_TYPES = "split",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        name: str = "DefaultMerger",
        **kwargs,
    ):
        self.output_dim = output_dim if isinstance(output_dim, int) else output_dim.dim
        self.intermediate_neurons = HyperParameter.from_value(
            intermediate_neurons, "Intermediate neurons"
        )
        self.output_strategy = output_strategy
        self.trunk_out = 1
        self.branch_out = 1
        merge_graph, in_ports, out_ports = self._build_merge_graph(backend)
        super().__init__(
            graph=merge_graph,
            input_ports=in_ports,
            output_ports=out_ports,
            backend=backend,
            name=name,
            **kwargs,
        )
        self._graph.setup()
        self.set_state(NodeState.UNINITIALIZED)

    def _build_merge_graph(self, backend):
        self.trunk_out = self.intermediate_neurons.value
        self.branch_out = self.intermediate_neurons.value
        merge_graph = Graph()
        in_ports = []
        out_ports = []
        if self.output_dim == 1:
            merge_node = Inner(backend=backend, keepdims=True)
            merge_graph.add_node(merge_node)
            in_ports = merge_node.input_ports
            out_ports = merge_node.output_ports
        else:
            if self.output_strategy == "split":
                # Both trunk and branch network will have
                # output_dim * intermediate_neurons neurons in the last layer.
                # Which we just reshape to (output_dim, intermediate_neurons).
                reshape_branch = ReshapeAtDim(
                    dim=-1,
                    new_shape=(self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                reshape_trunk = ReshapeAtDim(
                    dim=-1,
                    new_shape=(self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                self.branch_out *= self.output_dim
                self.trunk_out *= self.output_dim
            elif self.output_strategy == "split_branch":
                reshape_branch = ReshapeAtDim(
                    dim=-1,
                    new_shape=(self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                reshape_trunk = Unsqueeze(dim=-2, backend=backend)
                self.branch_out *= self.output_dim
            else:  # self.output_strategy == "split_trunk"
                reshape_branch = Unsqueeze(dim=-2, backend=backend)
                reshape_trunk = ReshapeAtDim(
                    dim=-1,
                    new_shape=(self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                self.trunk_out *= self.output_dim

            multiply_node = Multiply(backend=backend)
            sum_node = Sum(axis=-1, backend=backend)

            merge_graph.connect(reshape_branch, multiply_node.input_ports[0])
            merge_graph.connect(reshape_trunk, multiply_node.input_ports[1])
            merge_graph.connect(multiply_node, sum_node)
            in_ports = [reshape_branch.input_ports[0], reshape_trunk.input_ports[0]]
            out_ports = sum_node.output_ports

        return merge_graph, in_ports, out_ports

    def reset(self):
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    def setup(self) -> None:
        if self.state == NodeState.UNINITIALIZED:
            merge_graph, in_ports, out_ports = self._build_merge_graph(self.backend)
            self.setup_graph(
                merge_graph,
                input_ports=in_ports,
                output_ports=out_ports,
            )
            self._graph.setup()
            self.set_state(NodeState.INITIALIZED)
