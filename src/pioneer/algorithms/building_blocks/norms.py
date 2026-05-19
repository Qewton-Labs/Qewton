from ...graphs.control_nodes.graph_node import GraphNode
from ...config.backend import DEFAULT_DL_BACKEND, Backend, TensorType
from ...graphs.graphs import SequentialGraph

from .math import Mean, Square, Sum


class MSN(GraphNode):
    def __init__(self, backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND):
        self.mean_node = Mean(backend=backend)
        self.sum_node = Sum(backend=backend, axis=-1)
        self.square_node = Square(backend=backend)
        graph = SequentialGraph(
            self.square_node,
            self.sum_node,
            self.mean_node,
        )
        super().__init__(
            graph=graph,
            input_ports=self.square_node.input_ports,
            output_ports=self.mean_node.output_ports,
            backend=backend,
            name="MSN",
        )
