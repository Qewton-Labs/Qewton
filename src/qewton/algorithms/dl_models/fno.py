"""Fourier Neural Operator (FNO)."""

from typing import Generic, Literal

from qewton.algorithms.building_blocks.activation_functions import GELU
from qewton.algorithms.building_blocks.channel_mlp import ChannelMLP
from qewton.algorithms.building_blocks.embeddings import GridEmbeddingND
from qewton.algorithms.building_blocks.fno_block import FNOBlock
from qewton.algorithms.building_blocks.skip_connections import SkipType
from qewton.algorithms.building_blocks.spectral_convolution import FFTNorm
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import SequentialGraph
from qewton.graphs.nodes import Node, NodeState
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class FNO(GraphNode, Generic[TensorType]):
    """Fourier Neural Operator [1].

    Based on the FNO of the neuraloperator library [2], with the same argument names
    but only a subset of its options. The input of shape
    `(batch, in_channels, *spatial_dims)` is optionally extended by a positional
    embedding, lifted to `hidden_channels` by a channel MLP, passed through `n_layers`
    Fourier layers (`FNOBlock`), and projected to `out_channels` by another channel MLP.
    The Fourier layers apply `non_linearity` except for the last one.

    All arguments but `name` and `backend` can be hyperparameters. The network is
    rebuilt for their current values in `setup`, which re-initializes all weights.

    Attributes:
        in_channels, out_channels, n_modes, hidden_channels, n_layers,
            lifting_channel_ratio, projection_channel_ratio, positional_embedding,
            non_linearity, use_channel_mlp, channel_mlp_expansion, channel_mlp_skip,
            fno_skip, conv_bias, init_std, fft_norm (HyperParameter): The arguments of
            the same name, each stored as a hyperparameter.
        lifting_channels (int): Number of hidden channels of the lifting channel MLP.
        projection_channels (int): Number of hidden channels of the projection channel
            MLP.

    Args:
        in_channels (int | HyperParameter): Number of input channels.
        out_channels (int | HyperParameter): Number of output channels.
        n_modes (tuple[int, ...] | HyperParameter): Number of Fourier modes kept in
            every spatial dimension. Its length is the number of spatial dimensions.
        hidden_channels (int | HyperParameter, optional): Number of channels of the
            Fourier layers. Defaults to 64.
        n_layers (int | HyperParameter, optional): Number of Fourier layers.
            Defaults to 4.
        lifting_channel_ratio (float | HyperParameter, optional): The hidden layer of
            the lifting channel MLP has `int(lifting_channel_ratio * hidden_channels)`
            channels. With a ratio of 0 the lifting is a single linear layer.
            Defaults to 2.
        projection_channel_ratio (float | HyperParameter, optional): The hidden layer
            of the projection channel MLP has
            `int(projection_channel_ratio * hidden_channels)` channels. Defaults to 2.
        positional_embedding (Literal["grid"] | HyperParameter | None, optional):
            "grid" appends the coordinates of a regular grid on the unit cube to the
            input channels; None adds no embedding. Defaults to "grid".
        non_linearity (type[Node] | HyperParameter, optional): Non-linearity used in
            the lifting and projection and between the Fourier layers. Defaults to GELU.
        use_channel_mlp (bool | HyperParameter, optional): If the Fourier layers
            contain a channel MLP. Defaults to True.
        channel_mlp_expansion (float | HyperParameter, optional): Ratio between the
            hidden channels of the channel MLPs inside the Fourier layers and
            `hidden_channels`. Defaults to 0.5.
        channel_mlp_skip (SkipType | HyperParameter | None, optional): Skip connection
            around the channel MLPs inside the Fourier layers. Defaults to
            "soft-gating".
        fno_skip (SkipType | HyperParameter | None, optional): Skip connection around
            the spectral convolutions. Defaults to "linear".
        conv_bias (bool | HyperParameter, optional): If the spectral convolutions
            have a bias. Defaults to True.
        init_std (float | Literal["auto"] | HyperParameter, optional): Standard
            deviation to initialize the spectral convolutions. Defaults to "auto".
        fft_norm (FFTNorm | HyperParameter, optional): Normalization of the FFT of the
            spectral convolutions. Defaults to "forward".
        name (str, optional): Name of the node. Defaults to "fno".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.

    Note:
        Differences from the FNO of neuraloperator: for odd spatial sizes in all but the
        last dimension the spectrum is shifted back correctly (neuraloperator shifts it
        by one frequency there); only the "grid" positional embedding is supported.

    Raises:
        ValueError: If `positional_embedding` is not supported, if `hidden_channels` or
            `n_layers` is not positive, if a channel ratio is out of range, or if a
            layer gets invalid arguments (see `FNOBlock`, `SpectralConv`).
        TypeError: If `non_linearity` is not a node class.

    References:
        [1] Z. Li et al., "Fourier Neural Operator for Parametric Partial Differential
            Equations", ICLR 2021. https://arxiv.org/abs/2010.08895
        [2] neuraloperator library, https://github.com/neuraloperator/neuraloperator
    """

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter,
        n_modes: tuple[int, ...] | HyperParameter,
        hidden_channels: int | HyperParameter = 64,
        n_layers: int | HyperParameter = 4,
        lifting_channel_ratio: float | HyperParameter = 2,
        projection_channel_ratio: float | HyperParameter = 2,
        positional_embedding: Literal["grid"] | HyperParameter | None = "grid",
        non_linearity: type[Node] | HyperParameter = GELU,
        use_channel_mlp: bool | HyperParameter = True,
        channel_mlp_expansion: float | HyperParameter = 0.5,
        channel_mlp_skip: SkipType | HyperParameter | None = "soft-gating",
        fno_skip: SkipType | HyperParameter | None = "linear",
        conv_bias: bool | HyperParameter = True,
        init_std: float | Literal["auto"] | HyperParameter = "auto",
        fft_norm: FFTNorm | HyperParameter = "forward",
        name: str = "fno",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.in_channels = HyperParameter.from_value(in_channels, "FNO input channels")
        self.out_channels = HyperParameter.from_value(out_channels, "FNO output channels")
        self.n_modes = HyperParameter.from_value(n_modes, "FNO modes")
        self.hidden_channels = HyperParameter.from_value(
            hidden_channels, "FNO hidden channels"
        )
        self.n_layers = HyperParameter.from_value(n_layers, "FNO layers")
        self.lifting_channel_ratio = HyperParameter.from_value(
            lifting_channel_ratio, "FNO lifting channel ratio"
        )
        self.projection_channel_ratio = HyperParameter.from_value(
            projection_channel_ratio, "FNO projection channel ratio"
        )
        self.positional_embedding = HyperParameter.from_value(
            positional_embedding, "FNO positional embedding"
        )
        self.non_linearity = HyperParameter.from_value(non_linearity, "FNO non-linearity")
        self.use_channel_mlp = HyperParameter.from_value(
            use_channel_mlp, "FNO use channel MLP"
        )
        self.channel_mlp_expansion = HyperParameter.from_value(
            channel_mlp_expansion, "FNO channel MLP expansion"
        )
        self.channel_mlp_skip = HyperParameter.from_value(
            channel_mlp_skip, "FNO channel MLP skip"
        )
        self.fno_skip = HyperParameter.from_value(fno_skip, "FNO FNO skip")
        self.conv_bias = HyperParameter.from_value(conv_bias, "FNO conv bias")
        self.init_std = HyperParameter.from_value(init_std, "FNO init std")
        self.fft_norm = HyperParameter.from_value(fft_norm, "FNO FFT norm")

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

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_channels,
            self.out_channels,
            self.n_modes,
            self.hidden_channels,
            self.n_layers,
            self.lifting_channel_ratio,
            self.projection_channel_ratio,
            self.positional_embedding,
            self.non_linearity,
            self.use_channel_mlp,
            self.channel_mlp_expansion,
            self.channel_mlp_skip,
            self.fno_skip,
            self.conv_bias,
            self.init_std,
            self.fft_norm,
        ]

    @property
    def lifting_channels(self) -> int:
        """Number of hidden channels of the lifting channel MLP."""
        return int(self.lifting_channel_ratio.value * self.hidden_channels.value)

    @property
    def projection_channels(self) -> int:
        """Number of hidden channels of the projection channel MLP."""
        return int(self.projection_channel_ratio.value * self.hidden_channels.value)

    def setup(self) -> None:
        """Rebuilds the network, which re-initializes all of its weights."""
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=new_graph.sorted_nodes[0].input_ports,
            output_ports=new_graph.sorted_nodes[-1].output_ports,
        )

    def _check_arguments(self) -> None:
        """Checks the current values of the hyperparameters.

        Raises:
            ValueError: If `hidden_channels` or `n_layers` is not positive, a channel
                ratio is out of range, or the positional embedding is not supported.
        """
        if self.hidden_channels.value < 1:
            raise ValueError(
                f"hidden_channels must be positive, got {self.hidden_channels.value}."
            )
        if self.n_layers.value < 1:
            raise ValueError(f"n_layers must be positive, got {self.n_layers.value}.")
        if self.lifting_channel_ratio.value < 0:
            raise ValueError(
                "lifting_channel_ratio must not be negative, "
                f"got {self.lifting_channel_ratio.value}."
            )
        if self.projection_channels < 1:
            raise ValueError(
                f"projection_channel_ratio={self.projection_channel_ratio.value} gives "
                f"no hidden channels for hidden_channels={self.hidden_channels.value}."
            )
        if self.positional_embedding.value not in ("grid", None):
            raise ValueError(
                f"Unexpected positional_embedding {self.positional_embedding.value}."
            )

    def _build_network(self, backend: type[Backend[TensorType]]) -> SequentialGraph:
        """Builds the chain embedding, lifting, Fourier layers and projection.

        Args:
            backend (type[Backend[TensorType]]): The backend used for the computations.

        Returns:
            SequentialGraph: The network, with the nodes in the order of evaluation.

        Raises:
            ValueError: If the numbers of channels or layers, the channel ratios or the
                positional embedding are not supported (see `_check_arguments`).
        """
        self._check_arguments()
        nodes: list[Node] = []
        lifting_in_channels = self.in_channels.value
        if self.positional_embedding.value == "grid":
            embedding = GridEmbeddingND(
                in_channels=lifting_in_channels,
                dim=len(self.n_modes.value),
                backend=backend,
            )
            nodes.append(embedding)
            lifting_in_channels = embedding.out_channels

        nodes.append(self._build_lifting(lifting_in_channels, backend))
        nodes.extend(self._build_fourier_layers(backend))
        nodes.append(
            ChannelMLP(
                in_channels=self.hidden_channels.value,
                out_channels=self.out_channels.value,
                hidden_channels=self.projection_channels,
                n_layers=2,
                non_linearity=self.non_linearity.value,
                backend=backend,
            )
        )
        return SequentialGraph(*nodes)

    def _build_lifting(
        self, in_channels: int, backend: type[Backend[TensorType]]
    ) -> ChannelMLP:
        """Builds the lifting to the hidden channels.

        Args:
            in_channels (int): Number of channels of the lifted input, including the
                channels of the positional embedding.
            backend (type[Backend[TensorType]]): The backend used for the computations.

        Returns:
            ChannelMLP: A channel MLP with `lifting_channels` hidden channels, or a
                single linear layer if there are none.
        """
        has_hidden_layer = self.lifting_channels > 0
        return ChannelMLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels.value,
            hidden_channels=(
                self.lifting_channels if has_hidden_layer else self.hidden_channels.value
            ),
            n_layers=2 if has_hidden_layer else 1,
            non_linearity=self.non_linearity.value,
            backend=backend,
        )

    def _build_fourier_layers(self, backend: type[Backend[TensorType]]) -> list[FNOBlock]:
        """Builds the Fourier layers.

        Args:
            backend (type[Backend[TensorType]]): The backend used for the computations.

        Returns:
            list[FNOBlock]: `n_layers` Fourier layers. As in neuralop, the last one has
                no non-linearity.
        """
        n_layers = self.n_layers.value
        return [
            FNOBlock(
                in_channels=self.hidden_channels.value,
                out_channels=self.hidden_channels.value,
                n_modes=self.n_modes.value,
                use_channel_mlp=self.use_channel_mlp.value,
                channel_mlp_expansion=self.channel_mlp_expansion.value,
                non_linearity=(
                    None if layer_idx == n_layers - 1 else self.non_linearity.value
                ),
                fno_skip=self.fno_skip.value,
                channel_mlp_skip=self.channel_mlp_skip.value,
                conv_bias=self.conv_bias.value,
                init_std=self.init_std.value,
                fft_norm=self.fft_norm.value,
                backend=backend,
            )
            for layer_idx in range(n_layers)
        ]

    def forward(self, x: TensorType) -> TensorType:
        """Applies the Fourier Neural Operator.

        Args:
            x (TensorType): Input of shape `(batch, in_channels, *spatial_dims)`.

        Returns:
            TensorType: Output of shape `(batch, out_channels, *spatial_dims)`.
        """
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value
