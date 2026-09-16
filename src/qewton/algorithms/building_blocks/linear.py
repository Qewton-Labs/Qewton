from typing import Annotated, Generic

from qewton.algorithms.building_blocks.math import MatMul, Add, Multiply
from qewton.algorithms.building_blocks.creation import Identity
from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, DeepLearningBackend, TensorType
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.axes import EllipsisAxes, FeatureAxes, AxesDim
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.graphs.graphs import Graph
from qewton.graphs.control_nodes.graph_node import GraphNode


class FunctionalLinear(GraphNode, Generic[TensorType]):
    ell_ax = EllipsisAxes()
    dim_1 = AxesDim(None)
    dim_2 = AxesDim(None)

    def __init__(
        self,
        name="functional_linear",
        bias=True,
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.matmul_node = MatMul(backend=backend)
        self.add_node = Add(backend=backend)
        graph = Graph()
        in_ports = [self.matmul_node.input_ports[0], self.matmul_node.input_ports[1]]
        if bias:
            graph.connect(self.matmul_node.output_ports[0], self.add_node.input_ports[0])
            output_port = self.add_node.output_ports[0]
            in_ports.append(self.add_node.input_ports[1])
        else:
            graph.add_node(self.matmul_node)
            output_port = self.matmul_node.output_ports[0]

        super().__init__(
            graph=graph,
            input_ports=in_ports,
            output_ports=[output_port],
            name=name,
            backend=backend,
        )
        self.input = self.input_ports[0]
        self.weight = self.input_ports[1]
        if bias:
            self.bias = self.input_ports[2]
        self.output = self.output_ports[0]

    def forward(
        self,
        x: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1,)))],
        weight: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_1, dim_2)))],
        bias: Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_2,)))] = None,
    ) -> Annotated[TensorType, DC(ell_ax, FeatureAxes(shape=(dim_2,)))]:
        self.input.set_value(x)
        self.weight.set_value(weight)
        self.bias.set_value(bias)
        self.run()
        return self.output.value  # type: ignore


class Linear(GraphNode, Generic[TensorType]):
    """A node representing an linear deep learning layer

    Args:
        in_neurons (int | HyperParameter): The number of input neurons.
        out_neurons (int | HyperParameter): The number of output neurons.
        bias (bool): Whether to include a bias term. Defaults to True.
        name (str): Name of the node. Defaults to "linear".
        backend (type[DeepLearningBackend[TensorType]]): The deep learning backend
            to use. Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter,
        bias=True,
        name="linear",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.ellipsis_axes: (
            EllipsisAxes  # defined in x_data_config to allow usage in multiple graphs
        )
        self.weight = ParameterNode(
            (in_neurons, out_neurons), name="weight", backend=backend
        )
        if bias:
            self.bias = ParameterNode((out_neurons,), name="bias", backend=backend)
        self.functional_linear_node = FunctionalLinear(bias=bias, backend=backend)

        graph, input, output = self._build_graph(bias=bias, backend=backend)

        super().__init__(
            graph=graph,
            input_ports=[input],
            output_ports=[output],
            name=name,
            backend=backend,
        )
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def _build_graph(self, bias: bool, backend: type[DeepLearningBackend[TensorType]]):
        graph = Graph()
        graph.connect(self.weight, self.functional_linear_node.weight)
        if bias:
            graph.connect(self.bias, self.functional_linear_node.bias)
        return (
            graph,
            self.functional_linear_node.input,
            self.functional_linear_node.output,
        )

    def x_data_config(self):
        self.ellipsis_axes = EllipsisAxes()
        return DC(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.weight.shape[0].value,)),
            dtype=self.weight.backend.default_dtype,
        )

    def out_data_config(self):
        return DC(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.weight.shape[1].value,)),
            dtype=self.weight.backend.default_dtype,
        )

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        self.input.set_value(x)
        self.run()
        return self.output.value  # type: ignore


class Quadratic(Linear[TensorType]):
    """A node representing an quadratic deep learning layer of the form
    W_1*x (*) W_2*x + W_1*x + b. Here (*) means the hadamard product of
    two vectors (element-wise multiplication). W_1, W_2 are weight matrices
    and b is a bias vector.
    """

    def __init__(
        self,
        in_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter,
        bias=True,
        name="quadratic",
        backend: type[DeepLearningBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.weight_2 = ParameterNode(
            (in_neurons, out_neurons), name="weight", backend=backend
        )
        super().__init__(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            bias=bias,
            name=name,
            backend=backend,
        )

    def _build_graph(self, bias: bool, backend: type[DeepLearningBackend[TensorType]]):
        graph = Graph()
        input_node = Identity(backend=backend)
        matmul_node = MatMul(backend=backend)
        matmul_node_2 = MatMul(backend=backend)
        add_node = Add(backend=backend)
        add_node_2 = Add(backend=backend)
        multiply_node = Multiply(backend=backend)

        with graph.tracker(1) as x:
            x = input_node(x)
            x_1 = matmul_node(x, self.weight())
            x_2 = matmul_node_2(x, self.weight_2())
            x_3 = multiply_node(x_1, x_2)
            x_4 = add_node(x_3, x_1)
            if bias:
                add_node_2(x_4, self.bias())

        if bias:
            output = add_node_2.output_ports[0]
        else:
            output = add_node.output_ports[0]

        return graph, input_node.input_ports[0], output
