from qewton.algorithms.building_blocks.math import (
    Flatten,
    Transpose,
    Subtract,
    Divide,
    SVD,
    Add,
    MatMul,
    Multiply,
)
from qewton.algorithms.building_blocks.normalizations import (
    StdNormalizationNode,
    InverseStdNormalization,
)
from qewton.algorithms.building_blocks.array_operations import Narrow
from qewton.backends import DEFAULT_DL_BACKEND, Backend
from qewton.config.data_configurations import DataConfiguration
from qewton.data.datasets.base import DataSet
from qewton.graphs.graphs import Graph
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.nodes import Node, InputPort, OutputPort
from qewton.optim.parameters.hyperparameter_base import HyperParameter


class PCA(GraphNode):
    # TODO Maybe get batch axis via Dataconfig and make more complex splitting
    # possible
    def __init__(
        self,
        dataset_node: DataSet,
        batch_axis: int = 0,
        principal_components: int = 10,
        name: str = "PCANode",
        backend: Backend = DEFAULT_DL_BACKEND,
        divide_eps=1.0e-5,
    ) -> None:
        self.divide_eps = divide_eps
        self.dataset_node = dataset_node
        self.batch_axis = batch_axis

        self.principal_components = HyperParameter.from_value(
            principal_components, "PCA Principal Components"
        )
        self.computed_pca = False

        input_ports = []
        self.mean = OutputPort(
            data_configuration=DataConfiguration([]), node=self, name="mean"
        )
        self.std = OutputPort(
            data_configuration=DataConfiguration([]), node=self, name="std"
        )
        self.U = OutputPort(data_configuration=DataConfiguration([]), node=self, name="U")
        self.S = OutputPort(data_configuration=DataConfiguration([]), node=self, name="S")
        self.V = OutputPort(data_configuration=DataConfiguration([]), node=self, name="V")

        self.graph, output_port_mapping = self._build_graph(self.backend)

        super().__init__(
            self.graph, input_ports, output_port_mapping, name, backend=backend
        )

    def reset(self):
        self.computed_pca = False
        super().reset()

    def setup(self):
        self.graph, output_port_mapping = self._build_graph(self.backend)
        self.graph.setup()
        self.computed_pca = False
        self.update_inner_output_ports(list(output_port_mapping.keys()))

    def _build_graph(self, backend):
        graph = Graph()

        # Create all needed nodes

        # Recompute the SVD of the dataset:
        svd_node = SVD(backend=backend)
        normalization_node = StdNormalizationNode(
            dataset_node=self.dataset_node,
            normalization_axis=self.batch_axis,
            divide_eps=self.divide_eps,
            backend=self.backend,
        )

        # Build the new graph:
        normalization_node = StdNormalizationNode(
            dataset_node=self.dataset_node,
            normalization_axis=self.batch_axis,
            divide_eps=self.divide_eps,
            backend=self.backend,
        )

        if self.batch_axis != 0:
            transpose_node = Transpose(perm=[0, self.batch_axis], backend=backend)

        flatten_node = Flatten(start_dim=1, backend=backend)
        svd_node = SVD(backend=backend)

        narrow_node_1 = Narrow(
            dim=0, start=0, length=self.principal_components.value, backend=backend
        )
        narrow_node_2 = Narrow(
            dim=0, start=0, length=self.principal_components.value, backend=backend
        )
        narrow_node_3 = Narrow(
            dim=0, start=0, length=self.principal_components.value, backend=backend
        )

        output_port_mapping = {
            narrow_node_1.output_ports[0]: self.U,
            narrow_node_2.output_ports[0]: self.S,
            narrow_node_3.output_ports[0]: self.V,
        }
        # minus, Std and mean get input directly from the dataset
        std_add_eps_node.input_ports[1].default = self.divide_eps
        graph.connect(self.mean_node.output_ports[0], self.minus_node.input_ports[1])
        graph.connect(self.std_node.output_ports[0], std_add_eps_node.input_ports[0])
        graph.connect(self.minus_node.output_ports[0], divide_node.input_ports[0])
        graph.connect(std_add_eps_node.output_ports[0], divide_node.input_ports[1])
        if self.batch_axis != 0:
            graph.connect(divide_node, transpose_node.input_ports[0])  # type: ignore
            graph.connect(transpose_node, flatten_node.input_ports[0])  # type: ignore
        else:
            graph.connect(divide_node, flatten_node.input_ports[0])

        graph.connect(flatten_node.output_ports[0], svd_node.input_ports[0])
        graph.connect(svd_node.output_ports[0], narrow_node_1.input_ports[0])
        graph.connect(svd_node.output_ports[1], narrow_node_2.input_ports[0])
        graph.connect(svd_node.output_ports[2], narrow_node_3.input_ports[0])

        return graph, output_port_mapping

    def run(self):
        if not self.computed_pca:
            for node in [self.minus_node, self.mean_node, self.std_node]:
                node.input_ports[0].input_received_from_outside_graph = True
                node.input_ports[0].set_value(self.dataset_node.data)
            self.graph.run(self.mode)

            # Write the inner information into the own output ports
            for i, out_port in enumerate(self._inner_output_ports):
                self._output_ports[i].set_value(out_port.value)  # type: ignore
            self.computed_pca = True


class PCANN(GraphNode):
    def __init__(
        self,
        input_pca: PCA,
        neural_network: Node,
        output_pca: PCA,
        name: str = "FunctionalPCANNArchitecture",
    ) -> None:
        self.input_pca = input_pca
        self.neural_network = neural_network
        self.output_pca = output_pca
        assert (
            self.input_pca.backend == self.output_pca.backend
        ), "Using different backends can lead to unexpected behavior."
        assert (
            len(neural_network.input_ports) == 1 and len(neural_network.output_ports) == 1
        ), "Can not handle a network with multiple inputs and outputs."
        graph = self._build_graph()
        self.input = InputPort(DataConfiguration([]), node=self)
        self.output = OutputPort(DataConfiguration([]), node=self)
        super().__init__(
            graph,
            {self.in_subtract_node.input_ports[0]: self.input},
            {self.out_add_node.output_ports[0]: self.output},
            name,
        )

    def setup(self):
        # self.graph, output_port_mapping = self._build_graph(self.backend)
        self.graph.setup()
        # self.update_inner_output_ports(list(output_port_mapping.keys()))

    def _build_graph(self):
        backend = self.input_pca.backend
        # Nodes that transform the input and apply PCA
        self.in_subtract_node = Subtract(backend=backend)
        in_divide_node = Divide(backend=backend)
        if self.input_pca.batch_axis != 0:
            in_batch_transpose_node = Transpose(
                perm=[0, self.input_pca.batch_axis], backend=backend
            )
        in_flatten_node = Flatten(start_dim=1, backend=backend)
        in_transpose_node = Transpose(backend=backend)
        in_matmul_node = MatMul(backend=backend)
        in_divide_s_node = Divide(backend=backend)
        # Nodes that transform the output of the network back to its original shape
        out_multiply_s_node = Multiply(backend=backend)
        out_matmul_node = MatMul(backend=backend)
        # TODO: reshape node to std shape...
        if self.output_pca.batch_axis != 0:
            out_batch_transpose_node = Transpose(
                perm=[0, self.output_pca.batch_axis], backend=backend
            )
        out_multiply_node = Multiply(backend=backend)
        self.out_add_node = Add(backend=backend)
        ### Build the graph:
        graph = Graph()
        # Apply input PCA
        graph.connect(self.input_pca.mean, self.in_subtract_node.input_ports[1])
        graph.connect(self.in_subtract_node, in_divide_node.input_ports[0])
        graph.connect(self.input_pca.std, in_divide_node.input_ports[1])
        if self.input_pca.batch_axis != 0:
            graph.connect(in_divide_node, in_batch_transpose_node.input_ports[0])  # type: ignore
            graph.connect(in_batch_transpose_node, in_flatten_node.input_ports[0])  # type: ignore
        else:
            graph.connect(in_divide_node, in_flatten_node.input_ports[0])
        graph.connect(self.input_pca.V, in_transpose_node.input_ports[0])
        graph.connect(in_flatten_node, in_matmul_node.input_ports[0])
        graph.connect(in_transpose_node, in_matmul_node.input_ports[1])
        graph.connect(in_matmul_node, in_divide_s_node.input_ports[0])
        graph.connect(self.input_pca.S, in_divide_s_node.input_ports[1])
        # Apply network in between
        graph.connect(in_divide_s_node, self.neural_network.input_ports[0])
        # Inverse PCA for output
        graph.connect(
            self.neural_network.output_ports[0], out_multiply_s_node.input_ports[0]
        )
        graph.connect(self.output_pca.S, out_multiply_s_node.input_ports[1])
        graph.connect(out_multiply_s_node, out_matmul_node.input_ports[0])
        graph.connect(self.output_pca.V, out_matmul_node.input_ports[1])
        # TODO: Reshape to original shape
        if self.output_pca.batch_axis != 0:
            graph.connect(..., out_batch_transpose_node.input_ports[0])  # type: ignore
            graph.connect(out_batch_transpose_node, out_multiply_node.input_ports[0])  # type: ignore
        else:
            graph.connect(..., out_multiply_node.input_ports[0])
        graph.connect(self.output_pca.std, out_multiply_node.input_ports[1])
        graph.connect(out_multiply_node, self.out_add_node.input_ports[0])
        graph.connect(self.output_pca.mean, self.out_add_node.input_ports[1])
        return graph
