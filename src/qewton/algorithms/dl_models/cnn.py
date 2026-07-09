from typing import Generic

from qewton.algorithms.building_blocks.conv import Conv
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.config.variables import Variable
from qewton.graphs.graphs import SequentialGraph
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class CNN(GraphNode, Generic[TensorType]):
    """Implements a classical convolutional neural network. The input and output
    image dimension is assumed to be the same for this network, hence
    the kernel size should always be odd, so that the padding can be set
    to (kernel_size - 1) / 2.
    The network is a sequence of convolutional layers, each followed by
    an activation function, except for the last layer.

    Args:
        in_channels (int | HyperParameter | Variable): The number of input channels,
            or a variable that defines the input channels.
        hidden_channels (int | HyperParameter): The number of hidden channels in
            the convolutional layers.
        out_channels (int | HyperParameter | Variable): The number of output channels,
            or a variable that defines the output channels.
        n_hidden_layers (int | HyperParameter): The number of hidden convolutional
            layers in the network.
        kernel_size (int | HyperParameter | tuple[int  |  HyperParameter, ...]):
            The size of the convolutional kernel. Can be an int for 1D kernels,
            or a tuple for multi-dimensional kernels.
        bias (bool | HyperParameter, optional): If a bias should be included.
            Defaults to True.
        activation (type[Node] | HyperParameter, optional): What activation function
            should be used. Defaults to ReLU.
        name (str, optional): The name of this node. Defaults to "ccn".
        backend (type[DeepLearningBackend[TensorType]], optional): The computation
            backend. Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_channels: int | HyperParameter | Variable,
        hidden_channels: int | HyperParameter,
        out_channels: int | HyperParameter | Variable,
        n_hidden_layers: int | HyperParameter,
        kernel_size: int | HyperParameter | tuple[int | HyperParameter, ...],
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        name: str = "ccn",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        # Handle variable as input and output channels
        if isinstance(in_channels, Variable):
            self.input_var = in_channels
            in_channels = in_channels.dim
        else:
            self.input_var = None
        if isinstance(out_channels, Variable):
            self.output_var = out_channels
            out_channels = out_channels.dim
        else:
            self.output_var = None
        # Transform everything into HyperParameters
        self.in_channels = HyperParameter.from_value(in_channels, "CCN Input Channels")
        self.hidden_channels = HyperParameter.from_value(
            hidden_channels, "CCN Hidden Channels"
        )
        self.out_channels = HyperParameter.from_value(out_channels, "CCN Output Channels")
        self.n_hidden_layers = HyperParameter.from_value(
            n_hidden_layers, "CCN Hidden Layers"
        )
        self.bias = HyperParameter.from_value(bias, "CCN Bias")
        self.activation = HyperParameter.from_value(activation, "CCN Activations")
        if isinstance(kernel_size, int) or isinstance(kernel_size, HyperParameter):
            kernel_size = (kernel_size,)
        self.kernel_size: list[HyperParameter] = []
        for i, k in enumerate(kernel_size):
            self.kernel_size.append(HyperParameter.from_value(k, f"CCN Kernel Size {i}"))
            assert (
                self.kernel_size[i].current_value % 2 == 1
            ), f"Kernel size must be always odd, got {self.kernel_size[i]}"

        self._graph = self._build_network(backend)
        super().__init__(
            name=name,
            graph=self._graph,
            input_ports=self._graph.sorted_nodes[0].input_ports,
            output_ports=self._graph.sorted_nodes[-1].output_ports,
            backend=backend,
        )
        self._graph.setup()
        self._state = NodeState.UNINITIALIZED

    def _build_network(self, backend):
        nodes: list[Node] = []
        layers = self.n_hidden_layers.value + 1
        for i in range(layers):
            nodes.append(
                Conv(
                    in_channels=(
                        self.in_channels.value if i == 0 else self.hidden_channels.value
                    ),
                    out_channels=(
                        self.out_channels.value
                        if i == layers - 1
                        else self.hidden_channels.value
                    ),
                    kernel_size=tuple(k.value for k in self.kernel_size),
                    padding=tuple((k.value - 1) // 2 for k in self.kernel_size),
                    bias=self.bias.value,
                    name=f"conv_{i}",
                    backend=backend,
                )
            )
            # No activation after the last layer
            if i < layers - 1:
                nodes.append(self.activation.value(backend=backend))
        return SequentialGraph(*nodes)

    def setup(self):
        """Initializes the neural network itself for the current
        set of parameters.
        """
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=new_graph.sorted_nodes[0].input_ports,
            output_ports=new_graph.sorted_nodes[-1].output_ports,
        )

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
            self.n_hidden_layers,
            self.bias,
            self.activation,
        ]

    def forward(self, x):
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value
