from ..building_blocks.linear import Linear
from ..building_blocks.activation_functions import ActivationFunction, ReLU
from ..base import GraphNode
from ..implementation import DEFAULT_DL_IMPLEMENTATION
from ...pipelines.base import SequentialGraph
from ...nodes.base import Node
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
        activation: type[ActivationFunction] | HyperParameter = ReLU,
        name="fcn",
        backend=DEFAULT_DL_IMPLEMENTATION,
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
        self.setup()
        super().__init__(
            name=name,
            graph=self.graph,
            input_ports=self.graph.sorted_nodes[0].input_ports,
            output_ports=self.graph.sorted_nodes[-1].output_ports,
        )

    def setup(self):
        nodes: list[Node] = []
        for i in range(self.n_hidden_layers.value):
            nodes.append(
                Linear(
                    in_neurons=(
                        self.in_neurons.value if i == 0 else self.hidden_neurons.value
                    ),
                    out_neurons=(
                        self.hidden_neurons.value
                        if i < self.n_hidden_layers.value - 1
                        else self.out_neurons.value
                    ),
                    bias=self.bias.value,
                    backend=self.backend,
                )
            )
            if i < self.n_hidden_layers.value - 1:  # No activation after the last layer
                nodes.append(self.activation.value(backend=self.backend))
        self.graph = SequentialGraph(*nodes)

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
