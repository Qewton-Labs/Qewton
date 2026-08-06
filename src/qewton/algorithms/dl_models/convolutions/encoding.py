from typing import Generic, Literal
import math

from qewton.algorithms.building_blocks.conv import (
    Conv,
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
)
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.algorithms.building_blocks.array_operations import Flatten
from qewton.algorithms.dl_models.fcn import FCN
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.config.variables import Variable
from qewton.graphs.graphs import SequentialGraph
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class ConvolutionalEncoder(GraphNode, Generic[TensorType]):
    """A convolutional encoder that uses a series of convolutional
    layers followed by a fully connected layer to encode input data
    into a lower-dimensional representation.
    Expects input data of shape (batch_size, channels, *input_shape)
    and outputs data of shape (batch_size, out_channels).

    Args:
        in_channels (int | HyperParameter | Variable): The number of
            input channels.
        channels (tuple[int | HyperParameter, ...]): A tuple specifying
            the number of channels for each convolutional layer.
        out_channels (int | HyperParameter | Variable): The number of
            output channels after the fully connected layer.
        conv_kernel_size (HyperParameter | tuple[int, ...]): The size of
            the convolutional kernel. Has to be a tuple of length equal
            to the number of dimensions of the input data.
        fcn_hidden_neurons (int | HyperParameter): The number of neurons
            in the hidden layer of the fully connected network.
        fcn_hidden_layers (int | HyperParameter): The number of hidden
            layers in the fully connected network.
        input_shape (tuple[int, ...] | None): The shape of the input data
            excluding the batch and channel dimensions. If None, the input
            shape will be inferred from the connection in the graph or
            from the first input data received in forward.
        pooling_kernel_size (int | HyperParameter | tuple[int, ...]): The
            size of the pooling kernel. Can be an int or a tuple of ints.
            The stride in the pooling operation will be equal to the
            kernel size.
        pooling_type (Literal["average", "maximum"]): The type of pooling
            operation to use. Can be either "average" or "maximum".
        bias (bool | HyperParameter): Whether to include a bias term in the
            convolutional layers.
        activation (type[Node] | HyperParameter): The activation function
            to use after each convolutional layer. Should be a subclass of Node.
        name (str, optional): The name of the encoder node. Defaults to
            "ConvolutionalEncoder".
        backend (type[DeepLearningBackend[TensorType]], optional): The
            deep learning backend to use. Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_channels: int | HyperParameter | Variable,
        channels: tuple[int | HyperParameter, ...],
        out_channels: int | HyperParameter | Variable,
        conv_kernel_size: HyperParameter | tuple[int, ...],
        fcn_hidden_neurons: int | HyperParameter = 50,
        fcn_hidden_layers: int | HyperParameter = 1,
        input_shape: tuple[int, ...] | None = None,
        pooling_kernel_size: int | HyperParameter | tuple[int, ...] = 2,
        pooling_type: Literal["average", "maximum"] = "average",
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        name: str = "ConvolutionalEncoder",
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
        # Compute image dimension
        if isinstance(conv_kernel_size, HyperParameter):
            self.image_dim = len(conv_kernel_size.value)
        else:
            self.image_dim = len(conv_kernel_size)

        # Register all parameters
        self.in_channels = HyperParameter.from_value(
            in_channels, "Encoder Input Channels"
        )
        self.out_channels = HyperParameter.from_value(
            out_channels, "Encoder Output Channels"
        )
        self.channels: list[HyperParameter] = []
        for i, c in enumerate(channels):
            self.channels.append(
                HyperParameter.from_value(c, "Encoder Channel " + str(i))
            )
        self.bias = HyperParameter.from_value(bias, "Encoder Bias")
        self.conv_kernel_size = HyperParameter.from_value(
            conv_kernel_size, "Encoder Kernel Size"
        )
        self.pooling_kernel_size = HyperParameter.from_value(
            pooling_kernel_size, "Encoder Pooling Size"
        )
        self.activation = HyperParameter.from_value(activation, "UNet Activations")
        self.pooling_node = self._pick_pooling_type(pooling_type=pooling_type)
        self.fcn_hidden_neurons = HyperParameter.from_value(
            fcn_hidden_neurons, "Encoder FCN Hidden Neurons"
        )
        self.fcn_hidden_layers = HyperParameter.from_value(
            fcn_hidden_layers, "Encoder FCN Hidden Layers"
        )
        self.input_shape = input_shape
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

    def reset(self):
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    def _pick_pooling_type(self, pooling_type: Literal["average", "maximum"]):
        if pooling_type == "average":
            pool_list = [AvgPool1D, AvgPool2D, AvgPool3D]
        else:
            pool_list = [MaxPool1D, MaxPool2D, MaxPool3D]
        return pool_list[self.image_dim - 1]

    def _build_network(self, backend):
        nodes: list[Node] = []
        # First the convolutional layers
        for i, c in enumerate(self.channels):
            nodes.append(
                Conv(
                    in_channels=(
                        self.in_channels.value if i == 0 else self.channels[i - 1].value
                    ),
                    out_channels=c.value,
                    kernel_size=self.conv_kernel_size.value,
                    padding=tuple((k - 1) // 2 for k in self.conv_kernel_size.value),
                    bias=self.bias.value,
                    name=f"conv_{i}",
                    backend=backend,
                )
            )
            nodes.append(self.activation.value(backend=backend))
            nodes.append(self.pooling_node(self.pooling_kernel_size.value))
        # Then we flatten the output and add a fully connected layer
        nodes.append(Flatten(start_dim=1, backend=backend))

        fcn_input_dim = self.channels[-1].value
        if self.input_shape is not None:
            pooling_n = len(self.channels)  # Number of pooling operations
            if isinstance(self.pooling_kernel_size.value, int):
                for dim in self.input_shape[1:]:
                    fcn_input_dim *= math.floor(
                        dim / (self.pooling_kernel_size.value**pooling_n)
                    )
            elif isinstance(self.pooling_kernel_size.value, tuple):
                for dim, pool in zip(
                    self.input_shape[1:], self.pooling_kernel_size.value
                ):
                    fcn_input_dim *= math.floor(dim / (pool**pooling_n))

        fcn_encoding = FCN(
            in_neurons=fcn_input_dim,
            hidden_neurons=self.fcn_hidden_neurons,
            out_neurons=self.out_channels,
            n_hidden_layers=self.fcn_hidden_layers,
            activation=self.activation,
            backend=backend,
        )
        nodes.append(fcn_encoding)
        return SequentialGraph(*nodes)

    def setup(self):
        """Initializes the neural network itself for the current
        set of parameters.
        """
        if self.state == NodeState.UNINITIALIZED:
            new_graph = self._build_network(self.backend)
            self.setup_graph(
                new_graph,
                input_ports=new_graph.sorted_nodes[0].input_ports,
                output_ports=new_graph.sorted_nodes[-1].output_ports,
            )

            self.set_state(NodeState.INITIALIZED)

    def update_data_configs(self, updated_port, config_dict, dynamic_configs):
        ports = super().update_data_configs(updated_port, config_dict, dynamic_configs)
        # Read the expected input shape from the dynamic configs if it is not set yet
        if self.input_shape is None and updated_port == self._input_ports[0]:
            branch_config_shape = dynamic_configs[updated_port].shape[1:]
            if len(branch_config_shape) > 0 and all(
                isinstance(dim, int) for dim in branch_config_shape
            ):
                self.input_shape = branch_config_shape
        return ports

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.shape[1:]
        if self.state == NodeState.UNINITIALIZED:
            self.setup()
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value
