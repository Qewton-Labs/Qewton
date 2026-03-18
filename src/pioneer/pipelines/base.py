from __future__ import annotations
import warnings
from collections import deque

from ..nodes.base import InputPort, Node, EvaluationPhase, OutputPort
from ..constraints.base import Constraint
from ..optim.trainer.trainable_parameters import TrainableParametersCollection


class Graph:
    _graph_id_counter = 0

    def __init__(self):
        self.nodes: set[Node] = set[Node]()
        self.id = Graph._graph_id_counter
        Graph._graph_id_counter += 1
        self.sorted_nodes: list[Node] = []
        self.mode = EvaluationPhase.ALWAYS

    def copy(self) -> Graph:
        """Creates a (not deep) copy of this graph.

        Returns:
            Graph: The copied graph.
        """
        new_graph = Graph()
        new_graph.nodes = self.nodes.copy()
        return new_graph

    def add_node(self, node: Node, check_warning=True) -> None:
        """Adds a node to this graph.

        Args:
            node (Node): The node that is added.
            check_warning (bool, optional): Whether it is checked, that
                a node with the same name already exists in this graph.
                Defaults to True.
        """
        if check_warning:
            for known_node in self.nodes:
                if known_node.name == node.name:
                    warnings.warn(
                        f"Node with name '{node.name}' already in graph!", UserWarning
                    )
        self.nodes.add(node)

    def remove_node(self, node: Node) -> None:
        """Deletes a given node from this graph.

        Args:
            node (Node): The node that should be deleted.
        """
        self.nodes.remove(node)

    def sort(self):
        in_degree = {node: 0 for node in self.nodes}
        outgoing_edges = {node: [] for node in self.nodes}
        for node in self.nodes:
            for in_port in node.input_ports:
                if in_port.connected_ports[self.id] is not None:
                    in_degree[node] += 1
                    outgoing_edges[in_port.connected_ports[self.id].node].append(node)

        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        self.sorted_nodes: list[Node] = []

        while queue:
            node = queue.popleft()
            self.sorted_nodes.append(node)
            for edge in outgoing_edges[node]:
                in_degree[edge] -= 1
                if in_degree[edge] == 0:
                    queue.append(edge)

        # If two nodes depend on each other, they can never be added to the
        # queue, hence we can compare the length to check for cycles:
        if len(self.sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in computation graph!")

    def connect(
        self,
        from_: Node | OutputPort,
        to_: Node | InputPort,
    ) -> None:
        """Connect two nodes (which are automatically added to the pipeline, if they
        are not part of it). When evaluating the pipeline data will be
        exchanged between connected nodes.

        Args:
            from_ (Node, Port): The output port of a node, yielding the data. If the
                node is passed in directly, it should only have one output port. Else
                the desired port should be passed in via node[node.OutputKeys.].
            to_ (Node, Port): The input port of a node, expecting the data. The
                same node logic as for the first input holds.

        Raises:
            ValueError: The ports of both nodes are not compatible.
        """
        ports_to_connect = []
        for inp in from_, to_:
            if isinstance(inp, Node):
                ports_to_connect.append(
                    inp.output_ports if inp is from_ else inp.input_ports
                )
            else:
                ports_to_connect.append([inp])
        from_ports, to_ports = ports_to_connect
        if len(from_ports) != len(to_ports):
            raise ValueError(
                "When connecting two nodes directly, they should have the same number \
                    of output and input ports!"
            )

        from_node = from_ports[0].node
        to_node = to_ports[0].node
        # Nodes must be added to the graph
        self.add_node(from_node, check_warning=False)
        self.add_node(to_node, check_warning=False)

        # Configurations should match
        for from_port, to_port in zip(from_ports, to_ports):
            out_config = from_port.data_configuration
            in_config = to_port.data_configuration

            if not in_config.fits(out_config):
                raise ValueError("Incompatible input and output data configurations!")

            to_port.set_connected_port(from_port, self.id)  # type: ignore

        # TODO: fix circular import
        # if isinstance(from_node, GraphNode):
        #     from_node.copy_connections(self.id)
        # if isinstance(to_node, GraphNode):
        #     to_node.copy_connections(self.id)

    def _check_connect(
        self, user_input: InputPort | OutputPort | Node, check_input: bool = True
    ) -> InputPort | OutputPort:
        if isinstance(user_input, (InputPort, OutputPort)):
            return user_input
        ports = user_input.input_ports if check_input else user_input.output_ports
        if len(ports) != 1:
            raise ValueError(
                f"Node '{user_input.name}' has multiple output ports. "
                "Specify the port explicitly."
            )
        return ports[0]

    def disconnect(self, port: InputPort) -> None:
        """Remove an edge from this pipeline"""
        port.set_connected_port(None, self.id)

    def validate(self) -> None:
        """Validate that all required input ports of each node are connected."""
        throw_err = False
        for node in self.nodes:
            for port in node.input_ports:
                if port.is_required:  # Check if Input is needed
                    in_port = None
                    try:
                        in_port = port.connected_ports[self.id]
                    except IndexError:
                        throw_err = True

                    if throw_err or in_port is None:
                        raise ValueError(
                            f"Node '{node.name}' has required input port '{port.name}' "
                            f"that is not connected!"
                        )

    def setup(self):
        """Setup all nodes for evaluation."""
        for node in self.nodes:
            node.setup()
        self.sort()

    def run(self, mode: EvaluationPhase = EvaluationPhase.ALWAYS):
        """Run the pipeline. The data will be passed through the graph according to
        the connections and the computations of the nodes will be executed.
        """
        for node in self.sorted_nodes:
            node.set_mode(mode)
            node.set_pipeline_id(self.id)
            node.run()

    def collect_trainable_parameters(self):
        """Collect all trainable parameters from the nodes in this pipeline."""
        params_collection = TrainableParametersCollection()
        for node in self.nodes:
            p = node._trainable_parameters
            if not p.empty:
                params_collection.extend(p)
        return params_collection


class SequentialGraph(Graph):
    """
    A graph that is initialized as a sequence of nodes.
    """

    def __init__(self, *nodes: Node):
        super().__init__()
        for i in range(len(nodes) - 1):
            self.connect(nodes[i], nodes[i + 1])
        self.sorted_nodes = list(nodes)


class Pipeline(Graph):
    """A pipeline represents a workflow of data getting transformed
    through multiple computation steps and algorithms. Along this
    workflow one can set different constraints which can be used
    to train/validate/test the algorithm properties.
    """

    def __init__(self, name="pipeline"):
        """
        Args:
            name (str, optional): The internal name of this pipeline.
                Defaults to "pipeline".
        """
        super().__init__()
        self.constrain_nodes: set[Constraint] = set[Constraint]()
        self.name = name

    def add_node(self, node: Node, check_warning=True) -> None:
        """Adds a node to this pipeline.

        Args:
            node (Node): The node that is added.
            check_warning (bool, optional): Whether it is checked, that
                a node with the same name already exists in this pipeline.
                Defaults to True.
        """
        super().add_node(node, check_warning=check_warning)
        if isinstance(node, Constraint):
            self.constrain_nodes.add(node)

    def remove_node(self, node: Node) -> None:
        """Deletes a given node from this pipeline.

        Args:
            node (Node): The node that should be deleted.
        """
        self.nodes.remove(node)
        if isinstance(node, Constraint):
            self.constrain_nodes.remove(node)


class SequentialPipeline(Pipeline):
    """
    A pipeline that is initialized as a sequence of nodes.
    """

    def __init__(self, *nodes: Node, name="sequential_pipeline"):
        super().__init__(name=name)
        for i in range(len(nodes) - 1):
            self.connect(nodes[i], nodes[i + 1])
        self.sorted_nodes = list(nodes)
