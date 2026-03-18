from ..building_blocks.linear import Linear
from ..building_blocks.activation_functions import ActivationFunction, ReLU
from ...pipelines.base import SequentialGraph
from ..base import GraphNode
from ..implementation import DEFAULT_DL_IMPLEMENTATION


class FCN(GraphNode):
    """Fully Connected Network (FCN) implementation."""

    def __init__(
        self,
        in_neurons,
        hidden_neurons,
        out_neurons,
        n_hidden_layers,
        bias=True,
        activation: type[ActivationFunction] = ReLU,
        name="fcn",
        backend=DEFAULT_DL_IMPLEMENTATION,
    ):
        self.in_neurons = in_neurons  # TODO: make hyperparameter
        self.hidden_neurons = hidden_neurons
        self.out_neurons = out_neurons
        self.n_hidden_layers = n_hidden_layers
        self.bias = bias
        self.activation = activation
        self.backend = backend
        self.setup()
        super(FCN, self).__init__(
            name=name,
            graph=self.graph,
            input_ports=self.graph.sorted_nodes[0].input_ports,
            output_ports=self.graph.sorted_nodes[-1].output_ports,
        )

    def setup(self):
        nodes = []
        for i in range(self.n_hidden_layers):
            nodes.append(
                Linear(
                    in_neurons=self.in_neurons if i == 0 else self.hidden_neurons,
                    out_neurons=(
                        self.hidden_neurons
                        if i < self.n_hidden_layers - 1
                        else self.out_neurons
                    ),
                    bias=self.bias,
                    backend=self.backend,
                )
            )
            if i < self.n_hidden_layers - 1:  # No activation after the last layer
                nodes.append(self.activation(backend=self.backend))
        self.graph = SequentialGraph(nodes)
