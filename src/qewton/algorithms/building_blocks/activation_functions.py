from qewton.config.dtypes import Number
from qewton.config.data_configurations import DataConfiguration
from qewton.config.axes import EllipsisAxes
from qewton.graphs.nodes import Node, NodeState
from qewton.graphs.graphs import Graph
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.backends import DEFAULT_DL_BACKEND, TensorType, DeepLearningBackend
from qewton.algorithms.building_blocks.math import Multiply
from qewton.algorithms.building_blocks.parameters import ParameterNode


class AdaptiveActivation(GraphNode[TensorType]):
    """An adaptive activation function node that applies a specified
    activation function with learnable slope and shift parameters.

    Args:
        activation_function (Node[TensorType]): The activation function to
            be applied.
        initial_slope (float or TensorType): Initial value for the slope parameter.
            Defaults to 1.0.
        name (str): Name of the node. Defaults to "AdaptiveActivation".
        backend (type[DeepLearningBackend[TensorType]]): The deep learning backend
            to use. Defaults to DEFAULT_DL_BACKEND.
    """

    ellipsis_axes = EllipsisAxes()

    def __init__(
        self,
        activation_function: Node[TensorType],
        initial_slope=1.0,
        name: str = "AdaptiveActivation",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        if isinstance(initial_slope, (int, float)):
            initial_slope = backend.build_tensor(initial_slope)
        self.initial_slope = initial_slope
        self.activation_function = activation_function

        graph, in_port, out_port = self._build_graph(backend)
        super().__init__(
            name=name,
            graph=graph,
            input_ports=[in_port],
            output_ports=[out_port],
            backend=backend,
        )
        self._graph.setup()
        self.set_state(NodeState.UNINITIALIZED)

    def _build_graph(self, backend: type[DeepLearningBackend[TensorType]]):
        graph = Graph()

        a_fn = self.activation_function(backend=backend)
        slope = ParameterNode(
            shape=self.initial_slope.shape,
            initial_value=self.initial_slope,
            backend=backend,
        )

        multiply = Multiply(backend=backend)
        with graph.tracker(1) as x:
            x_mul = multiply(x, slope())
            a_fn(x_mul)
        return graph, multiply.input_ports[0], a_fn.output_ports[0]

    def setup(self):
        if self.state == NodeState.UNINITIALIZED:
            graph, in_port, out_port = self._build_graph(self.backend)  # type: ignore
            self.setup_graph(graph, input_ports=[in_port], output_ports=[out_port])
            self.set_state(NodeState.INITIALIZED)

    def forward(
        self, x: Number[TensorType, DataConfiguration(ellipsis_axes)]
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore


class ReLU(Node[TensorType]):
    """Rectified Linear Unit (ReLU) activation function.

    Applies the ReLU activation function element-wise to the input tensor.
    Returns max(0, x) for each element, effectively zeroing out negative values.
    """

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of ReLU activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with same shape and dtype as input.
        """
        return self.backend.nn.relu(x)


class Tanh(Node[TensorType]):
    """Hyperbolic Tangent (Tanh) activation function.

    Applies the tanh activation function element-wise to the input tensor.
    Maps input values to the range [-1, 1], providing a smooth, differentiable activation.
    """

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of Tanh activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with values in range [-1, 1],
                same shape and dtype as input.
        """
        return self.backend.nn.tanh(x)


class Sigmoid(Node[TensorType]):
    """Sigmoid activation function.

    Applies the sigmoid activation function element-wise to the input tensor.
    Maps input values to the range (0, 1), providing a smooth probability-like output.

    Attributes:
        ellipsis_axes (EllipsisAxes): Configuration for tensor axes handling.

    Examples:
        >>> sigmoid = Sigmoid(backend)
        >>> x = backend.library.array([[-2, 0], [2, 4]])
        >>> output = sigmoid.forward(x)
        >>> # output: approximately [[0.12, 0.5], [0.88, 0.98]]
    """

    ellipsis_axes = EllipsisAxes()

    def forward(
        self,
        x: Number[TensorType, DataConfiguration(ellipsis_axes)],
    ) -> Number[TensorType, DataConfiguration(ellipsis_axes)]:
        """Forward pass of Sigmoid activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            TensorType: Output tensor with values in range (0, 1),
                same shape and dtype as input.
        """
        return self.backend.nn.sigmoid(x)
