from qewton.backends import DEFAULT_DL_BACKEND, ComputingBackend, TensorType
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node
from qewton.graphs.control_nodes.graph_node import GraphNode


class DeepONet(GraphNode[TensorType]):
    """A general deep operator network (DeepONet) architecture that consists
    of a branch network, a trunk network, and a merge node.
    Internally this will create a graph that connects the branch and trunk
    networks to the merge node. The branch and trunk net should both only
    have one input and one output port. The merge node should have two input
    ports and one output port.
    The first input port of the DeepONet will be connected to the branch network,
    and the second input port of the DeepONet to the trunk network.

    Args:
        branch_net (Node[TensorType]): The branch network of the DeepONet.
            Should have one input and one output port. It handles the
            "parameters" of the operator.
        trunk_net (Node[TensorType]): The trunk network of the DeepONet.
            Should have one input and one output port. It handles the
            "coordinates" of the operator.
        merge_node (Node[TensorType]): The merge node of the DeepONet.
            Should have two input ports and one output port. It handles
            the connection/merging of the branch and trunk networks.
        name (str, optional): Defaults to "DeepONet".
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        branch_net: Node[TensorType],
        trunk_net: Node[TensorType],
        merge_node: Node[TensorType],
        name="DeepONet",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ):
        self._check_nodes(branch_net, trunk_net, merge_node, backend)

        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.merge_node = merge_node

        graph = Graph()
        graph.connect(self.branch_net, self.merge_node.input_ports[0])
        graph.connect(self.trunk_net, self.merge_node.input_ports[1])
        super().__init__(
            graph=graph,
            name=name,
            input_ports=self.branch_net.input_ports + self.trunk_net.input_ports,
            output_ports=self.merge_node.output_ports,
            backend=backend,
            **kwargs,
        )
        self.setup()

    def _check_nodes(self, branch_net, trunk_net, merge_node, backend):
        assert (
            branch_net.backend == backend
        ), "Branch network backend must match DeepONet backend."
        assert (
            trunk_net.backend == backend
        ), "Trunk network backend must match DeepONet backend."
        assert (
            merge_node.backend == backend
        ), "Merge node backend must match DeepONet backend."
        assert (
            len(merge_node.input_ports) == 2
        ), "Merge node must have exactly two input ports."
        assert (
            len(branch_net.output_ports) == 1
        ), "Branch network must have exactly one output port."
        assert (
            len(trunk_net.output_ports) == 1
        ), "Trunk network must have exactly one output port."

    def setup(self) -> None:
        self._graph.setup()
        self.branch_net.setup()
        self.trunk_net.setup()
        self.merge_node.setup()

    @property
    def hyperparameters(self) -> list:
        return (
            self.branch_net.hyperparameters
            + self.trunk_net.hyperparameters
            + self.merge_node.hyperparameters
        )
