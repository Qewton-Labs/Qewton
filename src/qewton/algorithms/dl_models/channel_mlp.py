from typing import Annotated, Generic

from qewton.algorithms.backend_node import BackendNode, TensorType
from qewton.algorithms.building_blocks.activation_functions import GELU
from qewton.algorithms.building_blocks.conv import Conv1d
from qewton.config.backend import DEFAULT_DL_BACKEND, Backend
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import SequentialGraph
from qewton.graphs.nodes import NodeState, Node


class ChannelMLP(GraphNode, Generic[TensorType]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        n_layers: int = 2,
        n_dim: int = 2,
        non_linearity: type[BackendNode] = GELU,
        dropout_p: float = 0.0,
        name: str = "channel_mlp",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self.backend = backend
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self._graph = self._build_network()
        super().__init__(
            name=name,
            graph=self._graph,
            input_ports=self._graph.sorted_nodes[0].input_ports,
            output_ports=self._graph.sorted_nodes[-1].output_ports,
            backend=backend,
        )
        self._graph.setup()
        self._state = NodeState.UNINITIALIZED

    def setup(self):
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=new_graph.sorted_nodes[0].input_ports,
            output_ports=new_graph.sorted_nodes[-1].output_ports,
        )

    def _build_network(self):
        nodes: list[Node] = []
        layers = self.n_layers + 1
        for i in range(layers):
            nodes.append(
                Conv1d(
                    in_channels=(self.in_channels if i == 0 else self.hidden_channels),
                    out_channels=(
                        self.hidden_channels if i < layers - 1 else self.out_channels
                    ),
                    kernel_size=1,
                    backend=self.backend,
                )
            )
            # No activation after the last layer
            if i < layers - 1:
                nodes.append(self.non_linearity(backend=self.backend))
        return SequentialGraph(*nodes)

    def forward(
        self, x: Annotated[TensorType, DataConfiguration.empty()]
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value
