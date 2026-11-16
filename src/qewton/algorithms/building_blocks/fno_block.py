"""Fourier layer of Fourier Neural Operators."""

from typing import Annotated, Generic, Literal

from qewton.algorithms.building_blocks.activation_functions import GELU
from qewton.algorithms.building_blocks.channel_mlp import ChannelMLP
from qewton.algorithms.building_blocks.creation import Identity
from qewton.algorithms.building_blocks.math import Add
from qewton.algorithms.building_blocks.skip_connections import SkipType, skip_connection
from qewton.algorithms.building_blocks.spectral_convolution import (
    FFTNorm,
    SpectralConv,
)
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node, NodeState, OutputPort
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class FNOBlock(GraphNode, Generic[TensorType]):
    """A single Fourier layer.

    Computes::

        x = non_linearity(conv(x) + fno_skip(x))
        if use_channel_mlp:
            x = channel_mlp(x) + channel_mlp_skip(x)
        x = non_linearity(x)

    where the skip connections are applied to the input of the block. If
    `non_linearity` is None, no non-linearity is applied, as in the last layer of an
    FNO. As in neuraloperator, the second non-linearity is applied also without a
    channel MLP. The channel MLP always uses GELU between its two layers.

    All arguments but `name` and `backend` can be hyperparameters. The network is
    rebuilt for their current values in `setup`, which re-initializes all weights.

    Attributes:
        in_channels, out_channels, n_modes, use_channel_mlp, channel_mlp_expansion,
            non_linearity, fno_skip, channel_mlp_skip, conv_bias, init_std, fft_norm
            (HyperParameter): The arguments of the same name, each stored as a
            hyperparameter.
        batch_axes, geo_axes, channel_dim_in, channel_dim_out: Shape information of the
            data configurations.
        input_node (Identity): Distributes the input to the parallel branches.
        conv (SpectralConv): The spectral convolution.
        fno_skip_connection (Node | None): Skip connection around the spectral
            convolution.
        fno_non_linearity (Node | None): Non-linearity after the spectral convolution.
        channel_mlp (ChannelMLP | None): The channel MLP.
        channel_mlp_skip_connection (Node | None): Skip connection around the channel
            MLP.
        post_non_linearity (Node | None): Non-linearity after the channel MLP, or
            after the spectral convolution if there is no channel MLP.

    Args:
        in_channels (int | HyperParameter): Number of input channels.
        out_channels (int | HyperParameter): Number of output channels.
        n_modes (tuple[int, ...] | HyperParameter): Number of Fourier modes kept in
            every spatial dimension.
        use_channel_mlp (bool | HyperParameter, optional): If a channel MLP follows
            the spectral convolution. Defaults to True.
        channel_mlp_expansion (float | HyperParameter, optional): The channel MLP has
            `round(out_channels * channel_mlp_expansion)` hidden channels.
            Defaults to 0.5.
        non_linearity (type[Node] | HyperParameter | None, optional): Non-linearity
            applied after the spectral convolution and after the channel MLP; None
            applies none. Defaults to GELU.
        fno_skip (SkipType | HyperParameter | None, optional): Skip connection around
            the spectral convolution; None for no skip connection. "soft-gating" and
            "identity" require `in_channels == out_channels`, also for
            `channel_mlp_skip`. Defaults to "linear".
        channel_mlp_skip (SkipType | HyperParameter | None, optional): Skip connection
            around the channel MLP; None for no skip connection. Ignored (and not
            checked) if `use_channel_mlp` is False. Defaults to "soft-gating".
        conv_bias (bool | HyperParameter, optional): If the spectral convolution has a
            bias. Defaults to True.
        init_std (float | Literal["auto"] | HyperParameter, optional): Standard
            deviation to initialize the spectral convolution. Defaults to "auto".
        fft_norm (FFTNorm | HyperParameter, optional): Normalization of the FFT of the
            spectral convolution. Defaults to "forward".
        name (str, optional): Name of the node. Defaults to "fno_block".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.

    Raises:
        TypeError: If `non_linearity` is neither None nor a node class.
        ValueError: If the channel MLP would have no hidden channels, if `fno_skip` or
            `channel_mlp_skip` is "soft-gating" or "identity" and the number of input
            and output channels differ, or if the spectral convolution gets invalid
            arguments (see `SpectralConv`).
    """

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        n_modes: tuple[int, ...] | HyperParameter,
        use_channel_mlp: bool | HyperParameter = True,
        channel_mlp_expansion: float | HyperParameter = 0.5,
        non_linearity: type[Node] | HyperParameter | None = GELU,
        fno_skip: SkipType | HyperParameter | None = "linear",
        channel_mlp_skip: SkipType | HyperParameter | None = "soft-gating",
        conv_bias: bool | HyperParameter = True,
        init_std: float | Literal["auto"] | HyperParameter = "auto",
        fft_norm: FFTNorm | HyperParameter = "forward",
        name: str = "fno_block",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.in_channels = HyperParameter.from_value(
            in_channels, "FNOBlock input channels"
        )
        self.out_channels = HyperParameter.from_value(
            out_channels, "FNOBlock output channels"
        )
        self.n_modes = HyperParameter.from_value(n_modes, "FNOBlock modes")
        self.use_channel_mlp = HyperParameter.from_value(
            use_channel_mlp, "FNOBlock use channel MLP"
        )
        self.channel_mlp_expansion = HyperParameter.from_value(
            channel_mlp_expansion, "FNOBlock channel MLP expansion"
        )
        self.non_linearity = HyperParameter.from_value(
            non_linearity, "FNOBlock non-linearity"
        )
        self.fno_skip = HyperParameter.from_value(fno_skip, "FNOBlock FNO skip")
        self.channel_mlp_skip = HyperParameter.from_value(
            channel_mlp_skip, "FNOBlock channel MLP skip"
        )
        self.conv_bias = HyperParameter.from_value(conv_bias, "FNOBlock conv bias")
        self.init_std = HyperParameter.from_value(init_std, "FNOBlock init std")
        self.fft_norm = HyperParameter.from_value(fft_norm, "FNOBlock FFT norm")

        # Shape information for the data configurations: the batch and the spatial
        # dimensions are the same for the input and the output.
        self.batch_axes = BatchAxes(AxesDim(None))
        self.channel_dim_in = AxesDim(self.in_channels.value)
        self.channel_dim_out = AxesDim(self.out_channels.value)
        self.geo_axes = GeometryAxes(
            shape=tuple(AxesDim(None) for _ in range(len(self.n_modes.value)))
        )

        graph, output_port = self._build_network(backend)
        super().__init__(
            graph=graph,
            input_ports=[self.input_node.input_ports[0]],
            output_ports=[output_port],
            name=name,
            backend=backend,
        )
        self._graph.setup()
        self._state = NodeState.UNINITIALIZED

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_channels,
            self.out_channels,
            self.n_modes,
            self.use_channel_mlp,
            self.channel_mlp_expansion,
            self.non_linearity,
            self.fno_skip,
            self.channel_mlp_skip,
            self.conv_bias,
            self.init_std,
            self.fft_norm,
        ]

    def setup(self) -> None:
        """Rebuilds the network, which re-initializes all of its weights."""
        self.channel_dim_in.update_size(self.in_channels.value)
        self.channel_dim_out.update_size(self.out_channels.value)
        graph, output_port = self._build_network(self.backend)
        self.setup_graph(
            graph,
            input_ports=[self.input_node.input_ports[0]],
            output_ports=[output_port],
        )

    def _build_network(
        self, backend: type[Backend[TensorType]]
    ) -> tuple[Graph, OutputPort]:
        """Builds the nodes and the graph of the block for the current hyperparameters.

        Args:
            backend (type[Backend[TensorType]]): The backend used for the computations.

        Returns:
            tuple[Graph, OutputPort]: The graph and the output port of the block.

        Raises:
            TypeError: If the non-linearity is neither None nor a node class.
            ValueError: If the channel MLP would have no hidden channels, or if the
                spectral convolution or a skip connection gets invalid arguments.
        """
        in_channels, out_channels = self.in_channels.value, self.out_channels.value
        n_dim = len(self.n_modes.value)
        non_linearity = self.non_linearity.value
        if non_linearity is not None and not (
            isinstance(non_linearity, type) and issubclass(non_linearity, Node)
        ):
            raise TypeError(
                f"non_linearity must be a Node class or None, got {non_linearity!r}."
            )
        channel_mlp_channels = round(out_channels * self.channel_mlp_expansion.value)
        if self.use_channel_mlp.value and channel_mlp_channels < 1:
            raise ValueError(
                f"channel_mlp_expansion={self.channel_mlp_expansion.value} gives no "
                f"hidden channels for out_channels={out_channels}."
            )

        self.input_node = Identity(backend=backend)
        self.conv = SpectralConv(
            in_channels,
            out_channels,
            self.n_modes.value,
            bias=self.conv_bias.value,
            init_std=self.init_std.value,
            fft_norm=self.fft_norm.value,
            backend=backend,
        )
        graph = Graph()
        block_input = self.input_node.output_ports[0]
        graph.connect(block_input, self.conv.input_ports[0])
        current_port = self.conv.output_ports[0]

        self.fno_skip_connection = None
        if self.fno_skip.value is not None:
            self.fno_skip_connection = skip_connection(
                in_channels,
                out_channels,
                n_dim,
                skip_type=self.fno_skip.value,
                backend=backend,
            )
            current_port = self._add_skip_connection(
                graph, block_input, current_port, self.fno_skip_connection, backend
            )

        self.fno_non_linearity = None
        if non_linearity is not None:
            self.fno_non_linearity = non_linearity(backend=backend)
            current_port = self._connect_node(graph, current_port, self.fno_non_linearity)

        self.channel_mlp = None
        self.channel_mlp_skip_connection = None
        if self.use_channel_mlp.value:
            self.channel_mlp = ChannelMLP(
                in_channels=out_channels,
                out_channels=out_channels,
                hidden_channels=channel_mlp_channels,
                n_layers=2,
                backend=backend,
            )
            current_port = self._connect_node(graph, current_port, self.channel_mlp)

            if self.channel_mlp_skip.value is not None:
                self.channel_mlp_skip_connection = skip_connection(
                    in_channels,
                    out_channels,
                    n_dim,
                    skip_type=self.channel_mlp_skip.value,
                    backend=backend,
                )
                current_port = self._add_skip_connection(
                    graph,
                    block_input,
                    current_port,
                    self.channel_mlp_skip_connection,
                    backend,
                )

        self.post_non_linearity = None
        if non_linearity is not None:
            self.post_non_linearity = non_linearity(backend=backend)
            current_port = self._connect_node(
                graph, current_port, self.post_non_linearity
            )
        return graph, current_port

    @staticmethod
    def _connect_node(graph: Graph, source: OutputPort, node: Node) -> OutputPort:
        """Feeds `source` into the single input of `node`.

        Args:
            graph (Graph): The graph to add the connection to.
            source (OutputPort): The port providing the value.
            node (Node): A node with a single input and a single output port.

        Returns:
            OutputPort: The output port of `node`.
        """
        graph.connect(source, node.input_ports[0])
        return node.output_ports[0]

    @staticmethod
    def _add_skip_connection(
        graph: Graph,
        block_input: OutputPort,
        current_port: OutputPort,
        skip: Node,
        backend: type[Backend[TensorType]],
    ) -> OutputPort:
        """Adds the skip connection of the block input to the current output.

        Args:
            graph (Graph): The graph to add the nodes and connections to.
            block_input (OutputPort): The input of the block, fed into `skip`.
            current_port (OutputPort): The current output of the block, to which
                `skip(block_input)` is added.
            skip (Node): The skip connection, with a single input and output port.
            backend (type[Backend[TensorType]]): The backend of the addition.

        Returns:
            OutputPort: The output port of the sum.
        """
        graph.connect(block_input, skip.input_ports[0])
        add_node = Add(backend=backend)
        graph.connect(current_port, add_node.input_ports[0])
        graph.connect(skip.output_ports[0], add_node.input_ports[1])
        return add_node.output_ports[0]

    def x_data_config(self) -> DC:
        return DC(
            self.batch_axes,
            FeatureAxes(shape=(self.channel_dim_in,)),
            self.geo_axes,
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self) -> DC:
        return DC(
            self.batch_axes,
            FeatureAxes(shape=(self.channel_dim_out,)),
            self.geo_axes,
            dtype=self.backend.default_dtype,
        )

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        """Applies the Fourier layer.

        Args:
            x (TensorType): Input of shape `(batch, in_channels, *spatial_dims)`.

        Returns:
            TensorType: Output of shape `(batch, out_channels, *spatial_dims)`.
        """
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore
