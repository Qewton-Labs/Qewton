from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.graphs.graphs import SequentialGraph

from qewton.algorithms.building_blocks.math import Mean, Square, Sum


class MSN(GraphNode):
    _type_identifier = "MSNNode"

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
