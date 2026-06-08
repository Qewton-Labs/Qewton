from typing import Annotated, Generic

from qewton.algorithms.building_blocks.math import MatMul, Add
from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
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
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.matmul_node = MatMul(backend=backend)
        self.add_node = Add(backend=backend)
        graph = Graph()
        if bias:
            graph.connect(self.matmul_node.output_ports[0], self.add_node.input_ports[0])
            output_port = self.add_node.output_ports[0]
        else:
            graph.add_node(self.matmul_node)
            output_port = self.matmul_node.output_ports[0]

        super().__init__(
            graph=graph,
            input_ports=[
                self.matmul_node.input_ports[0],
                self.matmul_node.input_ports[1],
                self.add_node.input_ports[1],
            ],
            output_ports=[output_port],
            name=name,
            backend=backend,
        )
        self.input = self.input_ports[0]
        self.weight = self.input_ports[1]
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
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """

    def __init__(
        self,
        in_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter,
        bias=True,
        name="linear",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
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

        graph = Graph()
        graph.connect(self.weight, self.functional_linear_node.weight)
        if bias:
            graph.connect(self.bias, self.functional_linear_node.bias)

        super().__init__(
            graph=graph,
            input_ports=[self.functional_linear_node.input],
            output_ports=[self.functional_linear_node.output],
            name=name,
            backend=backend,
        )
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def x_data_config(self):
        self.ellipsis_axes = EllipsisAxes()
        return DC(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.weight.shape[0].value,)),
            dtype=self.weight.backend.standard_datatype(),
        )

    def out_data_config(self):
        return DC(
            self.ellipsis_axes,
            FeatureAxes(shape=(self.weight.shape[1].value,)),
            dtype=self.weight.backend.standard_datatype(),
        )

    def forward(
        self, x: Annotated[TensorType, x_data_config]
    ) -> Annotated[TensorType, out_data_config]:
        self.input.set_value(x)
        self.run()
        return self.output.value  # type: ignore
