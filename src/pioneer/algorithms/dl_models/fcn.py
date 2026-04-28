from typing import Any, Annotated

from ..building_blocks.linear import Linear
from ..building_blocks.activation_functions import ReLU
from ..backend import DEFAULT_DL_BACKEND
from ...config import DataConfiguration
from ...graphs.graphs import SequentialGraph
from ...graphs.nodes import Node, NodeState
from ...graphs.control_nodes.graph_node import GraphNode
from ...optim.parameters.hyperparameter_base import HyperParameter


class FCN(GraphNode):
    """Fully Connected Network (FCN) implementation."""

    def __init__(
        self,
        in_neurons: int | HyperParameter,
        hidden_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter,
        n_hidden_layers: int | HyperParameter,
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        name="fcn",
        backend=DEFAULT_DL_BACKEND,
    ):
        self.in_neurons = HyperParameter.from_value(in_neurons, "FCN Input Neurons")
        self.hidden_neurons = HyperParameter.from_value(
            hidden_neurons, "FCN Hidden Neurons"
        )
        self.out_neurons = HyperParameter.from_value(out_neurons, "FCN Output Neurons")
        self.n_hidden_layers = HyperParameter.from_value(
            n_hidden_layers, "FCN Hidden Layers"
        )
        self.bias = HyperParameter.from_value(bias, "FCN Bias")
        self.activation = HyperParameter.from_value(activation, "FCN Activations")
        self.backend = backend

        self._graph = self._build_network()
        self._graph.setup()
        super().__init__(
            name=name,
            graph=self._graph,
            input_ports=self._graph.sorted_nodes[0].input_ports,
            output_ports=self._graph.sorted_nodes[-1].output_ports,
        )
        self._state = NodeState.UNINITIALIZED

    def _build_network(self):
        nodes: list[Node] = []
        layers = self.n_hidden_layers.value + 1
        for i in range(layers):
            nodes.append(
                Linear(
                    in_neurons=(
                        self.in_neurons.value if i == 0 else self.hidden_neurons.value
                    ),
                    out_neurons=(
                        self.hidden_neurons.value
                        if i < layers - 1
                        else self.out_neurons.value
                    ),
                    bias=self.bias.value,
                    backend=self.backend,
                )
            )
            # No activation after the last layer
            if i < layers - 1:
                nodes.append(self.activation.value(backend=self.backend))
        return SequentialGraph(*nodes)

    def setup(self):
        self._graph = self._build_network()
        self._graph.setup()
        self.update_inner_input_ports(self._graph.sorted_nodes[0].input_ports)
        self.update_inner_output_ports(self._graph.sorted_nodes[-1].output_ports)

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return [
            self.in_neurons,
            self.hidden_neurons,
            self.out_neurons,
            self.n_hidden_layers,
            self.bias,
            self.activation,
        ]

    def forward(
        self, x: Annotated[Any, DataConfiguration.empty()]
    ) -> Annotated[Any, DataConfiguration.empty()]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value
