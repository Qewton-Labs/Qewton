from typing import Literal

from qewton.algorithms.dl_models.deeponet.base import DeepONet
from qewton.algorithms.dl_models.fcn import FCN
from qewton.algorithms.building_blocks.math import Inner, Multiply, Sum
from qewton.algorithms.building_blocks.array_operations import Reshape, Unsqueeze
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.backends import DEFAULT_DL_BACKEND, ComputingBackend, TensorType
from qewton.graphs.nodes import Node
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.config.variables import Variable
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class FCNDeepONet(DeepONet[TensorType]):
    """A DeepONet architecture that uses fully connected networks (FCNs)
    for both the branch and trunk networks.
    The merge node can be configured to handle different output strategies,
    allowing for flexibility in how the outputs of the branch and trunk networks
    are combined.

    Args:
        trunk_input (int | Variable): The input dimension for
            the trunk network.
        branch_input (int | Variable): The input dimension for
            the branch network.
        output (int | Variable): The dimension of the final output of the DeepONet.
            If the output dimension is greater than 1, the merge node will
            be configured following the specified output strategy.
        trunk_hidden_neurons (int | HyperParameter): The number of hidden neurons
            in each layer of the trunk network.
        branch_hidden_neurons (int | HyperParameter): The number of hidden neurons
            in each layer of the branch network.
        trunk_n_hidden_layers (int | HyperParameter): The number of hidden layers
            in the trunk network.
        branch_n_hidden_layers (int | HyperParameter): The number of hidden layers
            in the branch network.
        intermediate_neurons (int | HyperParameter): The number of neurons in
            the intermediate layer which is the output of both the branch and
            trunk networks before merging.
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
        activations (type[Node] | HyperParameter, optional): Defaults to ReLU.
        name (str, optional): Defaults to "FCNDeepONet".
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        trunk_input: int | Variable | HyperParameter,
        branch_input: int | Variable | HyperParameter,
        output: int | Variable,
        trunk_hidden_neurons: int | HyperParameter,
        branch_hidden_neurons: int | HyperParameter,
        trunk_n_hidden_layers: int | HyperParameter,
        branch_n_hidden_layers: int | HyperParameter,
        intermediate_neurons: int | HyperParameter,
        output_strategy: Literal["split", "split_branch", "split_trunk"] = "split",
        activations: type[Node] | HyperParameter = ReLU,
        name="FCNDeepONet",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        self.output_dim = output if isinstance(output, int) else output.dim
        self.output_strategy = output_strategy
        self.intermediate_neurons = HyperParameter.from_value(
            intermediate_neurons, "Intermediate neurons"
        )
        self.trunk_input = trunk_input
        self.branch_input = branch_input
        self.trunk_hidden_neurons = HyperParameter.from_value(
            trunk_hidden_neurons, "Trunk hidden neurons"
        )
        self.branch_hidden_neurons = HyperParameter.from_value(
            branch_hidden_neurons, "Branch hidden neurons"
        )
        self.trunk_n_hidden_layers = HyperParameter.from_value(
            trunk_n_hidden_layers, "Trunk hidden layers"
        )
        self.branch_n_hidden_layers = HyperParameter.from_value(
            branch_n_hidden_layers, "Branch hidden layers"
        )
        self.activations = HyperParameter.from_value(activations, "Activations")

        merge_node, branch_net, trunk_net = self._build_network(backend)

        super().__init__(
            branch_net=branch_net,
            trunk_net=trunk_net,
            merge_node=merge_node,
            name=name,
            backend=backend,
            **kwargs,
        )

    def _build_network(self, backend):
        trunk_out = self.intermediate_neurons.value
        branch_out = self.intermediate_neurons.value
        if self.output_dim == 1:
            merge_node = Inner(backend=backend)
        else:
            merge_graph = Graph()
            if self.output_strategy == "split":
                # Both trunk and branch network will have
                # output_dim * intermediate_neurons neurons in the last layer.
                # Which we just reshape to (output_dim, intermediate_neurons).
                reshape_branch = Reshape(
                    new_shape=(-1, self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                reshape_trunk = Reshape(
                    new_shape=(-1, self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                branch_out *= self.output_dim
                trunk_out *= self.output_dim
            elif self.output_strategy == "split_branch":
                reshape_branch = Reshape(
                    new_shape=(-1, self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                reshape_trunk = Unsqueeze(dim=-2, backend=backend)
                branch_out *= self.output_dim
            else:  # self.output_strategy == "split_trunk"
                reshape_branch = Unsqueeze(dim=-2, backend=backend)
                reshape_trunk = Reshape(
                    new_shape=(-1, self.output_dim, self.intermediate_neurons.value),
                    backend=backend,
                )
                trunk_out *= self.output_dim

            multiply_node = Multiply(backend=backend)
            sum_node = Sum(axis=-1, backend=backend)

            merge_graph.connect(reshape_branch, multiply_node.input_ports[0])
            merge_graph.connect(reshape_trunk, multiply_node.input_ports[1])
            merge_graph.connect(multiply_node, sum_node)

            merge_node = GraphNode(
                graph=merge_graph,
                input_ports=[reshape_branch.input_ports[0], reshape_trunk.input_ports[0]],
                output_ports=sum_node.output_ports,
                name="Merge",
                backend=backend,
            )

        trunk_net = FCN(
            in_neurons=self.trunk_input,
            hidden_neurons=self.trunk_hidden_neurons,
            out_neurons=trunk_out,
            n_hidden_layers=self.trunk_n_hidden_layers,
            activation=self.activations,
            backend=backend,
        )
        branch_net = FCN(
            in_neurons=self.branch_input,
            hidden_neurons=self.branch_hidden_neurons,
            out_neurons=branch_out,
            n_hidden_layers=self.branch_n_hidden_layers,
            activation=self.activations,
            backend=backend,
        )

        return merge_node, branch_net, trunk_net

    def setup(self) -> None:
        self.merge_node, self.branch_net, self.trunk_net = self._build_network(
            self.backend
        )

        new_graph = Graph()
        new_graph.connect(self.branch_net, self.merge_node.input_ports[0])
        new_graph.connect(self.trunk_net, self.merge_node.input_ports[1])
        self.setup_graph(
            new_graph,
            input_ports=self.branch_net.input_ports + self.trunk_net.input_ports,
            output_ports=self.merge_node.output_ports,
        )
