import math
from typing import Annotated

from qewton.algorithms.building_blocks.math import Multiply, Cos, Sin
from qewton.algorithms.building_blocks.array_operations import ConcatNode
from qewton.algorithms.building_blocks.creation import Identity
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.algorithms.dl_models.fcn import FCN
from qewton.backends import DEFAULT_DL_BACKEND, ComputingBackend, TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import FeatureAxes, EllipsisAxes, AxesDim
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import NodeConfig, NodeState, Node
from qewton.config.variables import Variable
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class HarmonicEmbedding(GraphNode[TensorType]):
    """Adds a harmonic embedding to the input data.
    The embedding consists of sine and cosine functions of the input data,
    with frequencies determined by the `max_harmonic` parameter.
    Optionally, the original input can be included in the output.


    Args:
        max_harmonic (int | HyperParameter): The maximum harmonic frequency
            to include in the embedding.
        include_input (bool | HyperParameter, optional): If True, the original input will
            be included in the output. Defaults to True.
        name (str, optional): Defaults to "HarmonicEmbedding".
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        max_harmonic: int | HyperParameter,
        include_input: bool | HyperParameter = True,
        name: str = "HarmonicEmbedding",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        self.max_harmonic = HyperParameter.from_value(max_harmonic, "Max. Harmonic")
        self.include_input = HyperParameter.from_value(include_input, "Include input")

        self._graph, in_port, out_port = self._build_embedding_graph(backend)
        self.ellipsis_axes: EllipsisAxes
        self.feature_dim = AxesDim()
        super().__init__(
            name=name,
            graph=self._graph,
            input_ports=[in_port],
            output_ports=[out_port],
            backend=backend,
            **kwargs,
        )
        self._graph.setup()
        self._state = NodeState.UNINITIALIZED

    def _build_embedding_graph(self, backend):
        graph = Graph()
        identity_node = Identity(backend=backend)
        embedding_list = []

        harmonic_output_dim = 2 * self.max_harmonic.value
        if self.include_input.value:
            concat_node = ConcatNode(
                concat_dim=-1,
                num_of_input_ports=harmonic_output_dim + 1,
                backend=backend,
            )
            embedding_list.append(identity_node)
        else:
            concat_node = ConcatNode(
                concat_dim=-1,
                num_of_input_ports=harmonic_output_dim,
                backend=backend,
            )

        # Build the sin/cos transformations
        for i in range(1, self.max_harmonic.value + 1):
            multiply_i = Multiply(backend=backend)
            multiply_i.input_ports[1].default = 2 * math.pi * i
            cos_i = Cos(backend=backend)
            sin_i = Sin(backend=backend)

            graph.connect(identity_node, multiply_i.input_ports[0])
            graph.connect(multiply_i, cos_i)
            graph.connect(multiply_i, sin_i)

            embedding_list.append(cos_i)
            embedding_list.append(sin_i)

        # Concatenate them all
        for i, node in enumerate(embedding_list):
            graph.connect(node, concat_node.input_ports[i])

        return graph, identity_node.input_ports[0], concat_node.output_ports[0]

    def setup(self):
        new_graph, new_in, new_out = self._build_embedding_graph(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=[new_in],
            output_ports=[new_out],
        )
        self.set_state(NodeState.INITIALIZED)

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [self.include_input, self.max_harmonic]

    def x_data_config(self):
        self.ellipsis_axes = EllipsisAxes()
        return DataConfiguration(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.feature_dim,)),
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self):
        output_dim = 2 * self.max_harmonic.value
        if self.include_input.value:
            output_dim = 1 + output_dim
        output_dim = output_dim * self.feature_dim
        return DataConfiguration(
            self.ellipsis_axes,
            FeatureAxes(shape=(output_dim,)),
            dtype=self.backend.default_dtype,
        )

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore


class HarmonicFCN(GraphNode[TensorType]):
    """A fully connected neural network with harmonic embeddings.

    Args:
        input_dim (int | Variable): The dimension of the
            original input data, or a variable representing it.
        hidden_neurons (int | HyperParameter): The number of neurons in
            each hidden layer.
        output_dim (int | Variable): The dimension of the
            output data or a variable representing it.
        n_hidden_layers (int | HyperParameter): Number of hidden layers.
        max_harmonic (int | HyperParameter): The maximum harmonic frequency to
            include in the embedding.
        bias (bool | HyperParameter, optional): If a bias should be included.
            Defaults to True.
        activation (type[Node] | HyperParameter, optional): The activation
            function in each layer. Defaults to ReLU.
        include_input (bool | HyperParameter, optional): If True, the original
            input will be included in the output. Defaults to True.
        name (str, optional): Defaults to "HarmonicFCN".
        backend (type[ComputingBackend[TensorType]], optional):
            The backend to use for computations. Defaults to DEFAULT_DL_BACKEND.
    """

    _type_identifier = "HarmonicFCN"

    def __init__(
        self,
        input_dim: int | Variable | HyperParameter,
        hidden_neurons: int | HyperParameter,
        output_dim: int | HyperParameter | Variable,
        n_hidden_layers: int | HyperParameter,
        max_harmonic: int | HyperParameter,
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        include_input: bool | HyperParameter = True,
        name: str = "HarmonicFCN",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        if isinstance(input_dim, Variable):
            self.in_neurons = input_dim.dim
        self.in_neurons = HyperParameter.from_value(input_dim, "Input dimension")

        self.embedding = HarmonicEmbedding(
            max_harmonic=max_harmonic,
            include_input=include_input,
            backend=backend,
        )
        self.fcn = FCN(
            in_neurons=self._compute_network_input_dim(),
            hidden_neurons=hidden_neurons,
            out_neurons=output_dim,
            n_hidden_layers=n_hidden_layers,
            bias=bias,
            activation=activation,
            backend=backend,
        )
        graph = Graph()
        self._state = NodeState.UNINITIALIZED
        self.setup()
        graph.connect(self.embedding, self.fcn)
        super().__init__(
            name=name,
            graph=graph,
            input_ports=self.embedding.input_ports,
            output_ports=self.fcn.output_ports,
            backend=backend,
            **kwargs,
        )
        self._graph.setup()
        self.set_state(NodeState.UNINITIALIZED)

    def _compute_network_input_dim(self) -> int:
        embedding_multiplier = 2 * self.embedding.max_harmonic.value
        if self.embedding.include_input.value:
            embedding_multiplier += 1
        return embedding_multiplier * self.in_neurons.value

    def reset(self):
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return self.fcn.hyperparameters + self.embedding.hyperparameters

    def setup(self) -> None:
        if self.state == NodeState.UNINITIALIZED:
            self.embedding.setup()
            old_in_neurons = self.fcn.in_neurons.current_value
            self.fcn.in_neurons.current_value = self._compute_network_input_dim()
            self.fcn.reset()
            self.fcn.setup()
            self.fcn.in_neurons.current_value = old_in_neurons
            self.set_state(NodeState.INITIALIZED)

    def config_dict(self) -> NodeConfig:
        other_args = {
            "name": self.name,
            "backend": self.backend,
        }
        hyperparameters = {
            "input_dim": self.fcn.in_neurons,
            "hidden_neurons": self.fcn.hidden_neurons,
            "output_dim": self.fcn.out_neurons,
            "n_hidden_layers": self.fcn.n_hidden_layers,
            "max_harmonic": self.embedding.max_harmonic,
            "bias": self.fcn.bias,
            "activation": self.fcn.activation,
            "include_input": self.embedding.include_input,
        }
        return NodeConfig(
            node_identifier=self._type_identifier,
            node_id=self.node_id,
            mode=self.mode,
            hyperparameters=hyperparameters,  # type: ignore
            other_args=other_args,
            state=self.state,
            nested_graphs={"graph": self._graph},
        )

    @classmethod
    def load_from_config(cls, config: NodeConfig) -> Node:
        new_node: HarmonicFCN = super().load_from_config(config)  # type: ignore
        for node in new_node._graph.nodes:
            if isinstance(node, HarmonicEmbedding):
                new_node.embedding = node
            elif isinstance(node, FCN):
                new_node.fcn = node
        return new_node
