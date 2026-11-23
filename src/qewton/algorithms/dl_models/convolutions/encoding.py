from typing import Annotated, Generic, Literal
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
from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration
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
        in_channels (int | HyperParameter | Variable | None): The number of
            input channels, or None to infer it from the connected data.
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
        input_shape (tuple[int | HyperParameter, ...] | None): The spatial
            shape of the input data, i.e. without the batch and channel
            dimensions. If None, it is inferred from the connected data.
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
        in_channels: int | HyperParameter | Variable | None,
        channels: tuple[int | HyperParameter, ...],
        out_channels: int | HyperParameter | Variable,
        conv_kernel_size: HyperParameter | tuple[int, ...],
        fcn_hidden_neurons: int | HyperParameter = 50,
        fcn_hidden_layers: int | HyperParameter = 1,
        input_shape: tuple[int | HyperParameter, ...] | None = None,
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
        self.in_channels = HyperParameter.size_from(in_channels, "Encoder Input Channels")
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
        self.pooling_type = pooling_type
        self.pooling_node = self._pick_pooling_type(pooling_type=pooling_type)
        self.fcn_hidden_neurons = HyperParameter.from_value(
            fcn_hidden_neurons, "Encoder FCN Hidden Neurons"
        )
        self.fcn_hidden_layers = HyperParameter.from_value(
            fcn_hidden_layers, "Encoder FCN Hidden Layers"
        )
        self.input_shape = tuple(
            HyperParameter.size_from(
                None if input_shape is None else input_shape[i], f"Encoder Input Size {i}"
            )
            for i in range(self.image_dim)
        )
        self.batch_axes = BatchAxes(AxesDim(None))
        # With an open size, the network is built once it is inferred.
        graph = None if self.open_hyperparameters else self._build_network(backend)
        super().__init__(
            name=name,
            graph=graph,
            input_ports=[] if graph is None else graph.sorted_nodes[0].input_ports,
            output_ports=[] if graph is None else graph.sorted_nodes[-1].output_ports,
            backend=backend,
        )
        if graph is not None:
            self._graph.setup(plan=False)
        self._state = NodeState.UNINITIALIZED

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_channels,
            *self.channels,
            self.out_channels,
            self.conv_kernel_size,
            self.pooling_kernel_size,
            self.fcn_hidden_neurons,
            self.fcn_hidden_layers,
            self.bias,
            self.activation,
            *self.input_shape,
        ]

    def fcn_input_size(self) -> int:
        """The input size of the fully connected network: the channels of
        the last convolution times the spatial size left after pooling."""
        n_poolings = len(self.channels)
        pooling = self.pooling_kernel_size.value
        pools = pooling if isinstance(pooling, tuple) else (pooling,) * len(self.input_shape)
        return self.channels[-1].value * math.prod(
            math.floor(size.value / (pool**n_poolings))
            for size, pool in zip(self.input_shape, pools)
        )

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

        fcn_encoding = FCN(
            in_neurons=self.fcn_input_size(),
            hidden_neurons=self.fcn_hidden_neurons,
            out_neurons=self.out_channels,
            n_hidden_layers=self.fcn_hidden_layers,
            activation=self.activation,
            backend=backend,
        )
        nodes.append(fcn_encoding)
        return SequentialGraph(*nodes)

    def build(self):
        """Initializes the neural network itself for the current
        set of parameters.
        """
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=new_graph.sorted_nodes[0].input_ports,
            output_ports=new_graph.sorted_nodes[-1].output_ports,
        )


    def x_data_config(self) -> DataConfiguration:
        """The batch, the input channels and the spatial dims, bound to
        `in_channels` and `input_shape`."""
        channels = (
            FeatureAxes(self.input_var)
            if self.input_var is not None
            else FeatureAxes(shape=(AxesDim(self.in_channels),))
        )
        space = GeometryAxes(shape=tuple(AxesDim(size) for size in self.input_shape))
        return DataConfiguration(
            self.batch_axes, channels, space, dtype=self.backend.default_dtype
        )

    def out_data_config(self) -> DataConfiguration:
        """The batch and the output channels, bound to `out_channels`."""
        channels = (
            FeatureAxes(self.output_var)
            if self.output_var is not None
            else FeatureAxes(shape=(AxesDim(self.out_channels),))
        )
        return DataConfiguration(self.batch_axes, channels, dtype=self.backend.default_dtype)

    def upgrade(self, version: int) -> None:
        """See :meth:`~qewton.graphs.control_nodes.graph_node.GraphNode.upgrade`;
        files of version 1 have no batch axes yet and an input shape with the
        channels, as numbers.

        Args:
            version: The format version of the file.
        """
        if version < 2:
            self.batch_axes = BatchAxes(AxesDim(None))
            shape = self.input_shape or (None,) * (self.image_dim + 1)
            self.input_shape = tuple(
                HyperParameter.size_from(size, f"Encoder Input Size {i}")
                for i, size in enumerate(shape[1:])
            )
        super().upgrade(version)

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        return super().forward(x)
