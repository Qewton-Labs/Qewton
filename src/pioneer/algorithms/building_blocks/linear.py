from typing import Annotated, Generic

from .math import MatMul, Add
from .parameters import ParameterNode
from ..backend import DEFAULT_DL_BACKEND, Backend, TensorType
from ...config.configuration_base import DataConfiguration
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...graphs.graphs import Graph
from ...graphs.control_nodes.graph_node import GraphNode


class FunctionalLinear(GraphNode, Generic[TensorType]):

    def __init__(
        self,
        name="functional_linear",
        bias=True,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ):
        self.backend = backend

        self.matmul_node = MatMul(backend=self.backend)
        self.add_node = Add(backend=self.backend)
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
        )
        self.input = self.input_ports[0]
        self.weight = self.input_ports[1]
        self.bias = self.input_ports[2]
        self.output = self.output_ports[0]

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration.empty()],
        weight: Annotated[TensorType, DataConfiguration.empty()],
        bias: Annotated[TensorType, DataConfiguration.empty()] = None,
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
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
        )
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    def forward(
        self,
        x: Annotated[TensorType, DataConfiguration.empty()],
    ) -> Annotated[TensorType, DataConfiguration.empty()]:
        self.input.set_value(x)
        self.run()
        return self.output.value  # type: ignore
