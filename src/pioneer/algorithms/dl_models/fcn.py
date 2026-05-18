from typing import Annotated, Generic

from ..building_blocks.linear import Linear
from ..building_blocks.activation_functions import ReLU
from ...config.backend import DEFAULT_DL_BACKEND, Backend, TensorType
from ...config.data_configurations import DataConfiguration
from ...config.variables import Variable
from ...config.axes import FeatureAxes, EllipsisAxes
from ...graphs.graphs import SequentialGraph
from ...graphs.nodes import Node, NodeState
from ...graphs.control_nodes.graph_node import GraphNode
from ...optim.parameters.hyperparameter_base import HyperParameter


class FCN(GraphNode, Generic[TensorType]):
    """Fully Connected Network (FCN) implementation."""

    def __init__(
        self,
        in_neurons: int | HyperParameter | Variable,
        hidden_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter | Variable,
        n_hidden_layers: int | HyperParameter,
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        name: str = "fcn",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
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

        self._graph = self._build_network(backend)
        self.ellipsis_axes: EllipsisAxes
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
                    backend=backend,
                )
            )
            # No activation after the last layer
            if i < layers - 1:
                nodes.append(self.activation.value(backend=backend))
        return SequentialGraph(*nodes)

    def setup(self):
        new_graph = self._build_network(self.backend)
        self.setup_graph(
            new_graph,
            input_ports=new_graph.sorted_nodes[0].input_ports,
            output_ports=new_graph.sorted_nodes[-1].output_ports,
        )

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

    def x_data_config(self):
        self.ellipsis_axes = EllipsisAxes()
        if isinstance(self.in_neurons.value, Variable):
            return DataConfiguration(
                self.ellipsis_axes,
                FeatureAxes(variable=self.in_neurons.value),
                dtype=self.backend.standard_datatype(),
            )
        return DataConfiguration(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.in_neurons.value,)),
            dtype=self.backend.standard_datatype(),
        )

    def out_data_config(self):
        b_end = self.backend.standard_datatype()
        if isinstance(self.out_neurons.value, Variable):
            return DataConfiguration(
                self.ellipsis_axes,
                FeatureAxes(variable=self.out_neurons.value),
                dtype=b_end,
            )
        return DataConfiguration(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.out_neurons.value,)),
            dtype=b_end,
        )

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore
