"""Feedforward network models: :class:`FFN`, :class:`FCN`, :class:`QRES`
and :class:`DeepRitzNet`."""

from typing import Annotated, Generic

from qewton.algorithms.building_blocks.math import Power, Add
from qewton.algorithms.building_blocks.linear import Linear, Quadratic
from qewton.algorithms.building_blocks.activation_functions import ReLU, Tanh
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.data_configurations import DataConfiguration
from qewton.config.variables import Variable
from qewton.config.axes import FeatureAxes, EllipsisAxes
from qewton.graphs.graphs import SequentialGraph, Graph
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class FFN(GraphNode, Generic[TensorType]):
    """Feedforward neural network built from layer blocks and activations.

    The network is a sequence of ``n_hidden_layers + 1`` layer blocks with
    an activation after every block except the last. It is rebuilt in
    :meth:`setup` from the current hyperparameter values.

    Args:
        layer_blocks: The node class computing one layer, e.g.
            :class:`~qewton.algorithms.building_blocks.linear.Linear`. It is
            called with ``in_neurons``, ``out_neurons``, ``bias`` and
            ``backend``.
        in_neurons: The number of input neurons, or a variable whose dimension
            gives it and which is then used as the input feature variable.
        hidden_neurons: The number of neurons in each hidden layer.
        out_neurons: The number of output neurons, or a variable whose
            dimension gives it and which is then used as the output feature
            variable.
        n_hidden_layers: The number of hidden layers.
        bias: Whether the layers include a bias.
        activation: The activation node class applied after each hidden layer.
        name: The name of the node.
        backend: The backend used for computations.

    Attributes:
        input_var (Variable or None): The input feature variable, if given.
        output_var (Variable or None): The output feature variable, if given.
        layer_blocks (type[Node]): The node class computing one layer.
        in_neurons (HyperParameter): The number of input neurons.
        hidden_neurons (HyperParameter): The number of hidden neurons.
        out_neurons (HyperParameter): The number of output neurons.
        n_hidden_layers (HyperParameter): Architecture hyperparameter: the
            number of hidden layers.
        bias (HyperParameter): Architecture hyperparameter: whether the
            layers include a bias.
        activation (HyperParameter): Architecture hyperparameter: the
            activation node class.
        hyperparameters (list of HyperParameter): All architecture
            hyperparameters.

    Notes:
        Subclasses may override ``_build_network`` to build a different graph
        from the hyperparameters.
    """

    def __init__(
        self,
        layer_blocks: type[Node[TensorType]],
        in_neurons: int | HyperParameter | Variable,
        hidden_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter | Variable,
        n_hidden_layers: int | HyperParameter,
        bias: bool | HyperParameter = True,
        activation: type[Node] | HyperParameter = ReLU,
        name: str = "ffn",
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

        self.layer_blocks = layer_blocks
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
        self.set_state(NodeState.UNINITIALIZED)

    def reset(self):
        """Mark the network as uninitialized and reset the node.

        Returns:
            Any: The result of the parent's ``reset``.
        """
        self.set_state(NodeState.UNINITIALIZED)
        return super().reset()

    def _build_network(self, backend):
        nodes: list[Node] = []
        layers = self.n_hidden_layers.value + 1
        for i in range(layers):
            nodes.append(
                self.layer_blocks(
                    in_neurons=(self.in_neurons if i == 0 else self.hidden_neurons),
                    out_neurons=(
                        self.hidden_neurons if i < layers - 1 else self.out_neurons
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
        """Build the network for the current hyperparameter values.

        Only acts if the node is uninitialized. Sets the input and output
        data configurations and marks the node as initialized.
        """
        if self.state == NodeState.UNINITIALIZED:
            new_graph = self._build_network(self.backend)
            self.input_ports[0].update_static_data_configuration(self.x_data_config())
            self.output_ports[0].update_static_data_configuration(self.out_data_config())
            self.setup_graph(
                new_graph,
                input_ports=new_graph.sorted_nodes[0].input_ports,
                output_ports=new_graph.sorted_nodes[-1].output_ports,
            )
            self.set_state(NodeState.INITIALIZED)

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
        """Build the expected input data configuration.

        Returns:
            DataConfiguration: Arbitrary leading axes and one feature axis,
                given by the input variable or the number of input neurons.
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
        """Build the expected output data configuration.

        Returns:
            DataConfiguration: The leading axes of the input and one feature
                axis, given by the output variable or the number of output
                neurons.
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
        """Evaluate the network.

        Args:
            x: The input, see :meth:`x_data_config`.

        Returns:
            TensorType: The output, see :meth:`out_data_config`.
        """
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore


class FCN(FFN[TensorType]):
    """Fully connected network: an :class:`FFN` of linear layers.

    Args:
        in_neurons: The number of input neurons, or a variable whose dimension
            gives it and which is then used as the input feature variable.
        hidden_neurons: The number of neurons in each hidden layer.
        out_neurons: The number of output neurons, or a variable whose
            dimension gives it and which is then used as the output feature
            variable.
        n_hidden_layers: The number of hidden layers.
        bias: Whether the layers include a bias.
        activation: The activation node class applied after each hidden layer.
        name: The name of the node.
        backend: The backend used for computations.
    """

    def __init__(
        self,
        in_neurons: int | HyperParameter | Variable,
        hidden_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter | Variable,
        n_hidden_layers: int | HyperParameter,
        bias: bool | HyperParameter = True,
        activation: type[Node[TensorType]] | HyperParameter = Tanh,
        name: str = "fcn",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            layer_blocks=Linear,
            in_neurons=in_neurons,
            hidden_neurons=hidden_neurons,
            out_neurons=out_neurons,
            n_hidden_layers=n_hidden_layers,
            bias=bias,
            activation=activation,
            name=name,
            backend=backend,
        )


class QRES(FFN[TensorType]):
    """Quadratic residual network: an :class:`FFN` of quadratic layers.

    Args:
        in_neurons: The number of input neurons, or a variable whose dimension
            gives it and which is then used as the input feature variable.
        hidden_neurons: The number of neurons in each hidden layer.
        out_neurons: The number of output neurons, or a variable whose
            dimension gives it and which is then used as the output feature
            variable.
        n_hidden_layers: The number of hidden layers.
        bias: Whether the layers include a bias.
        activation: The activation node class applied after each hidden layer.
        name: The name of the node.
        backend: The backend used for computations.

    Notes:
        The layers are
        :class:`~qewton.algorithms.building_blocks.linear.Quadratic` blocks, as
        in Jie Bu and Anuj Karpatne, "Quadratic Residual Networks: A New Class
        of Neural Networks for Solving Forward and Inverse Problems in Physics
        Involving PDEs", 2021.
    """

    def __init__(
        self,
        in_neurons: int | HyperParameter | Variable,
        hidden_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter | Variable,
        n_hidden_layers: int | HyperParameter,
        bias: bool | HyperParameter = True,
        activation: type[Node[TensorType]] | HyperParameter = ReLU,
        name: str = "fcn",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        super().__init__(
            layer_blocks=Quadratic,
            in_neurons=in_neurons,
            hidden_neurons=hidden_neurons,
            out_neurons=out_neurons,
            n_hidden_layers=n_hidden_layers,
            bias=bias,
            activation=activation,
            name=name,
            backend=backend,
        )


class DeepRitzNet(FCN[TensorType]):
    """Residual network used in the Deep Ritz method.

    A linear input layer, `depth` residual blocks and a linear output
    layer. Each residual block applies twice a linear layer followed by
    ``ReLU(x**3)`` and adds the block input to the result.

    Args:
        in_neurons: The number of input neurons, or a variable whose dimension
            gives it and which is then used as the input feature variable.
        out_neurons: The number of output neurons, or a variable whose
            dimension gives it and which is then used as the output feature
            variable.
        width: The width of the hidden linear layers.
        depth: The number of residual blocks.
        bias: Whether the layers include a bias.
        name: The name of the node.
        backend: The backend used for computations.

    Notes:
        Architecture from Weinan E and Bing Yu, "The Deep Ritz method: A deep
        learning-based numerical algorithm for solving variational problems",
        2017.
    """

    # TODO: Improve the weight initialization, the current one can blow up
    # for deeper networks.

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
