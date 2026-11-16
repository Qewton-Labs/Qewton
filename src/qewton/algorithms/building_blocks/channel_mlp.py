"""Channel-wise multi-layer perceptron."""

from typing import Generic

from qewton.algorithms.building_blocks.activation_functions import GELU
from qewton.algorithms.building_blocks.conv import Flattened1dConv
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import SequentialGraph
from qewton.graphs.nodes import Node, NodeState
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class ChannelMLP(GraphNode, Generic[TensorType]):
    """A multi-layer perceptron applied to the channels of an input.

    For an input of shape `(batch, in_channels, x_1, ..., x_n)` the same MLP is applied
    to the channels at every position (a stack of 1x1 convolutions).

    Attributes:
        in_channels, out_channels, hidden_channels, n_layers, non_linearity
            (HyperParameter): The arguments of the same name, each stored as a
            hyperparameter.

    Args:
        in_channels (int | HyperParameter): Number of input channels.
        out_channels (int | HyperParameter | None, optional): Number of output
            channels. Defaults to `in_channels`.
        hidden_channels (int | HyperParameter | None, optional): Number of channels
            between the layers. Defaults to `in_channels`.
        n_layers (int | HyperParameter, optional): Number of linear layers. The
            non-linearity is applied between the layers, not after the last one. With
            a single layer the MLP is just one linear map from `in_channels` to
            `out_channels`. Defaults to 2.
        non_linearity (type[Node] | HyperParameter, optional): The non-linearity
            between the layers. Defaults to GELU.
        name (str, optional): Name of the node. Defaults to "channel_mlp".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.

    Raises:
        ValueError: If a number of channels or layers is not positive.
        TypeError: If `non_linearity` is not a node class.
    """

    def __init__(
        self,
        in_channels: int | HyperParameter,
        out_channels: int | HyperParameter | None = None,
        hidden_channels: int | HyperParameter | None = None,
        n_layers: int | HyperParameter = 2,
        non_linearity: type[Node] | HyperParameter = GELU,
        name: str = "channel_mlp",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.in_channels = HyperParameter.from_value(
            in_channels, "ChannelMLP input channels"
        )
        self.hidden_channels = HyperParameter.from_value(
            hidden_channels, "ChannelMLP hidden channels"
        )
        self.out_channels = HyperParameter.from_value(
            out_channels, "ChannelMLP output channels"
        )
        self.n_layers = HyperParameter.from_value(n_layers, "ChannelMLP layers")
        self.non_linearity = HyperParameter.from_value(
            non_linearity, "ChannelMLP non-linearity"
        )

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
        return list(
            dict.fromkeys(
                [
                    self.in_channels,
                    self.hidden_channels,
                    self.out_channels,
                    self.n_layers,
                    self.non_linearity,
                ]
            )
        )

    def setup(self) -> None:
        """Rebuilds the network, which re-initializes all of its weights."""
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=new_graph.sorted_nodes[0].input_ports,
            output_ports=new_graph.sorted_nodes[-1].output_ports,
        )

    def _build_network(self, backend: type[Backend[TensorType]]) -> SequentialGraph:
        """Builds the chain of linear layers with the non-linearity in between.

        Args:
            backend (type[Backend[TensorType]]): The backend used for the computations.

        Returns:
            SequentialGraph: The layers in the order of evaluation.

        Raises:
            ValueError: If a number of channels or layers is not positive.
            TypeError: If the non-linearity is not a node class.
        """
        n_layers = self.n_layers.value
        non_linearity = self.non_linearity.value
        channels = (
            self.in_channels.value,
            self.hidden_channels.value,
            self.out_channels.value,
        )
        if min(channels) < 1 or n_layers < 1:
            raise ValueError(
                "All numbers of channels and n_layers must be positive, got "
                f"{channels} channels and {n_layers} layers."
            )
        if not (isinstance(non_linearity, type) and issubclass(non_linearity, Node)):
            raise TypeError(f"non_linearity must be a Node class, got {non_linearity!r}.")

        nodes: list[Node] = []
        for layer_idx in range(n_layers):
            is_last_layer = layer_idx == n_layers - 1
            nodes.append(
                Flattened1dConv(
                    in_channels=(
                        self.in_channels if layer_idx == 0 else self.hidden_channels
                    ),
                    out_channels=(
                        self.out_channels if is_last_layer else self.hidden_channels
                    ),
                    kernel_size=1,
                    bias=True,
                    backend=backend,
                )
            )
            if not is_last_layer:
                nodes.append(non_linearity(backend=backend))
        return SequentialGraph(*nodes)

    def forward(self, x: TensorType) -> TensorType:
        """Applies the MLP to the channels at every position.

        Args:
            x (TensorType): Input of shape `(batch, in_channels, *spatial_dims)`.

        Returns:
            TensorType: Output of shape `(batch, out_channels, *spatial_dims)`.
        """
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value
