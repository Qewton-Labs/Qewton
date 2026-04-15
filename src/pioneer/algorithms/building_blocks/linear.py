from .math import MatMul, Add
from .trainable_parameters import TrainableParameterNode
from ..implementation import DEFAULT_DL_IMPLEMENTATION
from ...config.configuration_base import DataConfiguration
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...graphs.graphs import Graph
from ...graphs.nodes import InputPort, OutputPort
from ...graphs.control_nodes.graph_node import GraphNode


class FunctionalLinear(GraphNode):

    def __init__(
        self, name="functional_linear", bias=True, backend=DEFAULT_DL_IMPLEMENTATION
    ):
        self.input = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="input",
        )
        self.weight = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="weight",
        )
        self.bias = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="bias",
            default=None,
        )
        self.output = OutputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="output",
        )
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
            input_ports={
                self.matmul_node.input_ports[1]: self.weight,
                self.matmul_node.input_ports[0]: self.input,
                self.add_node.input_ports[1]: self.bias,
            },
            output_ports={output_port: self.output},
            name=name,
        )


class Linear(GraphNode):
    """A node representing an activation function, which is a special type of
    algorithm that is applied element-wise to the input data.
    """

    def __init__(
        self,
        in_neurons: int | HyperParameter,
        out_neurons: int | HyperParameter,
        bias=True,
        name="linear",
        backend=DEFAULT_DL_IMPLEMENTATION,
    ):
        self.input = InputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="input",
        )
        self.output = OutputPort(
            data_configuration=DataConfiguration(),
            node=self,
            name="output",
        )

        self.weight = TrainableParameterNode(
            (in_neurons, out_neurons), name="weight", backend=backend
        )
        if bias:
            self.bias = TrainableParameterNode(
                (out_neurons,), name="bias", backend=backend
            )
        self.functional_linear_node = FunctionalLinear(bias=bias, backend=backend)

        graph = Graph()
        graph.connect(self.weight, self.functional_linear_node.weight)
        if bias:
            graph.connect(self.bias, self.functional_linear_node.bias)

        super().__init__(
            graph=graph,
            input_ports={self.functional_linear_node.input: self.input},
            output_ports={self.functional_linear_node.output: self.output},
            name=name,
        )
        if self._input_ports is not None and self._output_ports is not None:
            self._input_ports[0].data_configuration.specify_dtype(backend)
            self._output_ports[0].data_configuration.specify_dtype(backend)
