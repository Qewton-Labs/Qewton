from typing import Annotated, Generic

from qewton.algorithms.building_blocks.math import Power, Add
from qewton.algorithms.building_blocks.linear import Linear
from qewton.algorithms.building_blocks.activation_functions import ReLU
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.config.axes import FeatureAxes, EllipsisAxes
from qewton.graphs.graphs import SequentialGraph, Graph
from qewton.graphs.nodes import Node, NodeState, NodeConfig
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class FCN(GraphNode, Generic[TensorType]):
    """Fully Connected Network (FCN) implementation.

    Args:
        in_neurons (int | HyperParameter | Variable): The number of input neurons
            or a variable representing it.
        hidden_neurons (int | HyperParameter): The number of neurons in
            each hidden layer.
        out_neurons (int | HyperParameter | Variable): The number of output
            neurons or a variable representing it.
        n_hidden_layers (int | HyperParameter): Number of hidden layers.
        bias (bool | HyperParameter, optional): If a bias should be included.
            Defaults to True.
        activation (type[Node] | HyperParameter, optional): The activation
            function in each layer. Defaults to ReLU.
        name (str, optional): Name of the model. Defaults to "fcn".
        backend (type[Backend[TensorType]], optional): What backend this
            model should use for the computations. Defaults to DEFAULT_DL_BACKEND.
    """

    _type_identifier = "FCNNode"

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
        if isinstance(in_neurons, Variable):
            self.input_var = in_neurons
            in_neurons = in_neurons.dim
        else:
            self.input_var = None
        if isinstance(out_neurons, Variable):
            self.output_var = out_neurons
            out_neurons = out_neurons.dim
        else:
            self.output_var = None

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
            self.in_neurons,
            self.hidden_neurons,
            self.out_neurons,
            self.n_hidden_layers,
            self.bias,
            self.activation,
        ]

    def x_data_config(self):
        """Build the expected input data configuration based on the
        input variable or number of input neurons.

        Returns:
            DataConfiguration: The input data configuration for the FCN.
        """
        self.ellipsis_axes = EllipsisAxes()
        if self.input_var is not None:
            return DataConfiguration(
                self.ellipsis_axes,
                FeatureAxes(variable=self.input_var),
                dtype=self.backend.default_dtype,
            )
        return DataConfiguration(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.in_neurons.value,)),
            dtype=self.backend.default_dtype,
        )

    def out_data_config(self):
        """Build the expected output data configuration based on the
        output variable or number of output neurons.

        Returns:
            DataConfiguration: The output data configuration for the FCN.
        """
        b_end = self.backend.default_dtype
        if self.output_var is not None:
            return DataConfiguration(
                self.ellipsis_axes,
                FeatureAxes(variable=self.output_var),
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


class DeepRitzNet(FCN[TensorType]):
    """Implementation of the architecture used in the Deep Ritz paper [1]_.
    Consists of fully connected layers and residual connections.

    Args:
        in_neurons (int | HyperParameter | Variable): The number of input neurons
            or a variable representing it.
        out_neurons (int | HyperParameter | Variable): The number of output
            neurons or a variable representing it.
        width (int | HyperParameter): The width of the used hidden
            fully connected layers.
        depth (int | HyperParameter): The amount of subsequent residual blocks.
        bias (bool | HyperParameter, optional): If a bias should be included.
            Defaults to True.
        name (str, optional): Name of the model. Defaults to "fcn".
        backend (type[Backend[TensorType]], optional): What backend this
            model should use for the computations. Defaults to DEFAULT_DL_BACKEND.

    Notes:
        [1] Weinan E and Bing Yu, "The Deep Ritz method: A deep learning-based numerical
        algorithm for solving variational problems", 2017

    TODO: Improve initialization of the network-weights, as the current one can
    blow-up for deeper networks.
    """

    _type_identifier = "DeepRitzNetNode"

    def __init__(
        self,
        in_neurons: int | HyperParameter | Variable,
        out_neurons: int | HyperParameter | Variable,
        width: int | HyperParameter,
        depth: int | HyperParameter,
        bias: bool | HyperParameter = True,
        name: str = "DeepRitzNet",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        super().__init__(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            hidden_neurons=width,
            n_hidden_layers=depth,
            bias=bias,
            activation=ReLU,
            name=name,
            backend=backend,
        )

    def _build_network(self, backend):
        graph = Graph()
        linear_in = Linear(
            in_neurons=self.in_neurons.value,
            out_neurons=self.hidden_neurons.value,
            bias=self.bias.value,
            backend=backend,
        )
        last_node = linear_in
        for _ in range(self.n_hidden_layers.value):
            linear_1 = Linear(
                in_neurons=self.hidden_neurons.value,
                out_neurons=self.hidden_neurons.value,
                bias=self.bias.value,
                backend=backend,
            )
            linear_2 = Linear(
                in_neurons=self.hidden_neurons.value,
                out_neurons=self.hidden_neurons.value,
                bias=self.bias.value,
                backend=backend,
            )
            power_node_1 = Power(power=3.0)
            power_node_2 = Power(power=3.0)
            relu_node_1 = ReLU()
            relu_node_2 = ReLU()
            add_node = Add()
            graph.connect(last_node, linear_1)
            graph.connect(linear_1, power_node_1.input_ports[0])
            graph.connect(power_node_1, relu_node_1)
            graph.connect(relu_node_1, linear_2)
            graph.connect(linear_2, power_node_2.input_ports[0])
            graph.connect(power_node_2, relu_node_2)
            graph.connect(last_node, add_node.input_ports[0])
            graph.connect(relu_node_2, add_node.input_ports[1])
            last_node = add_node

        linear_out = Linear(
            in_neurons=self.hidden_neurons.value,
            out_neurons=self.out_neurons.value,
            bias=self.bias.value,
            backend=backend,
        )
        graph.connect(last_node, linear_out)
        graph.sort()
        return graph

    def config_dict(self) -> NodeConfig:
        other_args = {"name": self.name, "backend": self.backend}
        hyperparameters = {
            "in_neurons": self.in_neurons,
            "out_neurons": self.out_neurons,
            "width": self.hidden_neurons,
            "depth": self.n_hidden_layers,
            "bias": self.bias,
        }
        return NodeConfig(
            node_identifier=self._type_identifier,
            node_id=self.node_id,
            mode=self.mode,
            hyperparameters=hyperparameters,
            other_args=other_args,
            state=self.state,
            nested_graphs={"graph": self._graph},
        )
