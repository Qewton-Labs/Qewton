from typing import Generic, Literal

from qewton.algorithms.building_blocks.conv import (
    Conv,
    DoubleConv,
    Interpolate,
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
)
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.algorithms.building_blocks.array_operations import (
    ConcatNode,
    GetShapeNode,
    Slice,
)
from qewton.algorithms.building_blocks.creation import Identity
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.config.variables import Variable
from qewton.graphs.graphs import SequentialGraph, Graph
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

    _type_identifier = "CNNNode"

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
                self.kernel_size[i].value % 2 == 1
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
                    kernel_size=tuple(self.kernel_size),
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
        ] + self.kernel_size

    def forward(self, x):
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value


class UNet(GraphNode, Generic[TensorType]):
    """A simple UNet as used in many imaging tasks. It consists of double
    convolution blocks with batch norms, followed by a pooling layer for down-
    sampling. The upsampling is carried out by a nearest interpolation.
    Expects data to be in the shape of: (batch, channels, width, [height, depth]),
    where height and depth are optional.

    Args:
        in_channels (int | HyperParameter | Variable): The number of input channels.
        channels (tuple[int  |  HyperParameter, ...]): The number of hidden channels.
            Should be given as an tuple. Only the encoding channels should be
            provided, the decoding use the channels backwards. E.g. a
            value channels=(16, 32, 64), would create a UNet with:
                input -> 16 -> 32 -> 64 -> 32 -> 16 -> output
        out_channels (int | HyperParameter | Variable): The number of output channels.
        conv_kernel_size (HyperParameter | tuple[int, ...]): The size of the kernels
            used in the convolution. They should be odd, for correct padding.
        pooling_kernel_size (int | HyperParameter | tuple[int, ...], optional):
            The pooling window size. Defaults to 2.
        pooling_type (Literal["average";, "maximum"], optional):
            What kind of pooling should be applied. Defaults to "average".
        skip_connections (bool | HyperParameter, optional): If the encoding
            layers should add information back to the decoding layers. This is
            done by a concatenation of both intermediate values. Defaults to True.
        bias (bool | HyperParameter, optional): If the convolution should include a bias.
            Defaults to True.
        activation (type[Node] | HyperParameter, optional): The activation function.
            Defaults to ReLU.
        name (str, optional): Defaults to "UNet".
        backend (type[DeepLearningBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_channels: int | HyperParameter | Variable,
        channels: tuple[int | HyperParameter, ...],
        out_channels: int | HyperParameter | Variable,
        conv_kernel_size: HyperParameter | tuple[int, ...],
        pooling_kernel_size: int | HyperParameter | tuple[int, ...] = 2,
        pooling_type: Literal["average", "maximum"] = "average",
        skip_connections: bool | HyperParameter = True,
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        name: str = "UNet",
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
        self.in_channels = HyperParameter.from_value(in_channels, "UNet Input Channels")
        self.out_channels = HyperParameter.from_value(
            out_channels, "UNet Output Channels"
        )
        self.channels: list[HyperParameter] = []
        for i, c in enumerate(channels):
            self.channels.append(
                HyperParameter.from_value(c, "UNetInner Channel " + str(i))
            )
        self.bias = HyperParameter.from_value(bias, "UNet Bias")
        self.skip_connections = HyperParameter.from_value(
            skip_connections, "UNet Skip Connection"
        )
        self.conv_kernel_size = HyperParameter.from_value(
            conv_kernel_size, "UNet Kernel Size"
        )
        self.pooling_kernel_size = HyperParameter.from_value(
            pooling_kernel_size, "UNet Pooling Size"
        )
        self.activation = HyperParameter.from_value(activation, "UNet Activations")
        self.pooling_node = self._pick_pooling_type(pooling_type=pooling_type)
        self.pooling_type = pooling_type
        # Build a starting graph/network
        self._graph, in_node, out_node = self._build_network(backend)
        super().__init__(
            name=name,
            graph=self._graph,
            input_ports=in_node.input_ports,
            output_ports=out_node.output_ports,
            backend=backend,
        )
        self._graph.setup()
        self._state = NodeState.UNINITIALIZED

    def _pick_pooling_type(self, pooling_type: Literal["average", "maximum"]):
        if pooling_type == "average":
            pool_list = [AvgPool1D, AvgPool2D, AvgPool3D]
        else:
            pool_list = [MaxPool1D, MaxPool2D, MaxPool3D]
        assert (
            self.image_dim >= 1 and self.image_dim <= 3
        ), f"UNet can only work on 1D, 2D or 3D images of the form \
            (batch, channels, image_dim), got image_dim={self.image_dim}"
        return pool_list[self.image_dim - 1]

    def _build_network(self, backend):
        graph = Graph()

        conv_block_list_down = []
        pooling_list = []

        conv_block_list_up = []
        interpolate_list: list[Interpolate] = []
        shape_node_list = []
        slice_node_list = []
        skip_list: list[ConcatNode] = []

        down_channels = [self.in_channels] + self.channels
        up_channels = [self.out_channels] + self.channels

        # Walk downwards the channels and create all needed nodes
        # The downward reduction we can also directly connect
        for i in range(len(down_channels) - 1):
            # Downsampling step:
            assert all(
                k % 2 == 1 for k in self.conv_kernel_size.value
            ), "Only odd kernel sizes are supported by default."
            padding = tuple((k - 1) // 2 for k in self.conv_kernel_size.value)
            conv_block_list_down.append(
                DoubleConv(
                    in_channels=down_channels[i].value,
                    out_channels=down_channels[i + 1].value,
                    kernel_size=self.conv_kernel_size.value,
                    padding=padding,
                    activation=self.activation.value,
                    bias=self.bias.value,
                    backend=backend,
                )
            )
            pooling_list.append(
                self.pooling_node(self.pooling_kernel_size.value, stride=2)
            )
            graph.connect(conv_block_list_down[i], pooling_list[i])

            # Upsampling step:
            interpolate_list.append(Interpolate(backend=backend))
            shape_node_list.append(GetShapeNode(backend=backend))
            slice_node_list.append(Slice(slice_config=slice(2, None), backend=backend))
            # In the upwards conv. we switch the order of the index, since
            # we want to invert the steps from above:
            if self.skip_connections.value:
                skip_list.append(ConcatNode(concat_dim=1, backend=backend))
                in_channels = 2 * up_channels[i + 1].value
            else:
                in_channels = up_channels[i + 1].value
            if i == 0:  # For the final step we want to just use a convolution
                conv_block_list_up.append(
                    Conv(
                        in_channels=in_channels,
                        out_channels=up_channels[i].value,
                        kernel_size=tuple(1 for _ in range(self.image_dim)),
                        bias=self.bias.value,
                        backend=backend,
                    )
                )
            else:
                conv_block_list_up.append(
                    DoubleConv(
                        in_channels=in_channels,
                        out_channels=up_channels[i].value,
                        kernel_size=tuple(1 for _ in range(self.image_dim)),
                        activation=self.activation.value,
                        bias=self.bias.value,
                        backend=backend,
                    )
                )

            # Build connections:
            if len(conv_block_list_down) > 1:
                # Downsampling straightforward:
                graph.connect(pooling_list[i - 1], conv_block_list_down[i])

                # For upsampling, we need to interpolate to the correct
                # shape
                graph.connect(conv_block_list_down[i], shape_node_list[i])
                graph.connect(shape_node_list[i], slice_node_list[i])
                graph.connect(slice_node_list[i], interpolate_list[i].size_port)
                self._add_skip_connection(
                    graph,
                    conv_block_list_down,
                    conv_block_list_up,
                    interpolate_list,
                    skip_list,
                    i,
                )
                # Note, this is backwards, so we connect the previous conv.
                # to the "next" interpolation which is given by a smaller index.
                graph.connect(conv_block_list_up[i], interpolate_list[i - 1].input_port)

        # Bottleneck connection:
        graph.connect(pooling_list[-1], interpolate_list[-1].input_port)

        # The output references the input size, for this add an artificial
        # identity node:
        id_node = Identity()
        graph.connect(id_node, conv_block_list_down[0])

        graph.connect(id_node, shape_node_list[0])
        graph.connect(shape_node_list[0], slice_node_list[0])
        graph.connect(slice_node_list[0], interpolate_list[0].size_port)
        self._add_skip_connection(
            graph,
            conv_block_list_down,
            conv_block_list_up,
            interpolate_list,
            skip_list,
            0,
        )

        return graph, id_node, conv_block_list_up[0]

    def _add_skip_connection(
        self,
        graph: Graph,
        conv_block_list_down,
        conv_block_list_up,
        interpolate_list,
        skip_list,
        i,
    ):
        if self.skip_connections.value:
            graph.connect(interpolate_list[i], skip_list[i].input_ports[0])
            graph.connect(conv_block_list_down[i], skip_list[i].input_ports[1])
            graph.connect(skip_list[i], conv_block_list_up[i])
        else:
            graph.connect(interpolate_list[i], conv_block_list_up[i])

    def setup(self):
        new_graph, in_node, out_node = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=in_node.input_ports,
            output_ports=out_node.output_ports,
        )

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_channels,
            self.out_channels,
            self.skip_connections,
            self.bias,
            self.activation,
            self.conv_kernel_size,
            self.pooling_kernel_size,
        ] + self.channels
