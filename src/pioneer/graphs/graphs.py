from __future__ import annotations
from collections import deque
from contextlib import contextmanager
import inspect
from typing import Callable

from .nodes import InputPort, Node, EvaluationPhase, OutputPort, Port
from ..optim.parameters.trainable_parameters import TrainableParametersCollection
from .edges import Edge


class Graph:

    def __init__(self):
        self.nodes: set[Node] = set[Node]()
        self.sorted_nodes: list[Node] = []
        self.mode = EvaluationPhase.ALWAYS

        self.incoming_edges: dict[Node, list[Edge]] = {}
        self.sorted_edges: list[dict[Port, Edge]] = []
        self.last_outgoing_edges: list[Edge] = []

    @classmethod
    def from_function(
        cls, func: Callable
    ) -> tuple[Graph, list[list[InputPort] | InputPort], list[OutputPort]]:
        """Creates a graph from a function.
        method of a node.

        Args:
            func (Callable): The function that is used to create the graph.
        """
        graph = cls()
        sig = inspect.signature(func)
        if len(sig.parameters) > 0:
            with graph.tracker(n_tracking_vars=len(sig.parameters)) as tracking_vars:
                if isinstance(tracking_vars, TrackingObject):
                    tracking_vars = (tracking_vars,)
                out = func(*tracking_vars)  # type: ignore
            input_ports = [
                var.to_ports if len(var.to_ports) > 1 else var.to_ports[0]
                for var in tracking_vars  # type: ignore
            ]
        else:
            with graph.tracker():
                out = func()
            input_ports = []
        if isinstance(out, tuple):
            assert all(
                isinstance(o, TrackingObject) for o in out
            ), "All outputs of the function must be TrackingObjects."
            output_ports = [var.last_output_port for var in out]
        else:
            if out is None:
                output_ports = []
            else:
                assert isinstance(
                    out, TrackingObject
                ), "The output of the function must be a TrackingObject."
                output_ports = [out.last_output_port]

        return graph, input_ports, output_ports

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
                if known_node.node_id == node.node_id:
                    raise ValueError(
                        f"Node {node.name} and ID: {node.node_id} already exists in this graph!"
                    )
        self.nodes.add(node)
        if not node in self.incoming_edges:
            self.incoming_edges[node] = list[Edge]()

    def remove_node(self, node: Node) -> None:
        """Deletes a given node from this graph.

        Args:
            node (Node): The node that should be deleted.
        """
        self.nodes.remove(node)
        self.incoming_edges.pop(node)

    def sort(self):
        in_degree = {node: 0 for node in self.nodes}
        outgoing_connections = {node: [] for node in self.nodes}
        for node in self.nodes:
            for edge in self.incoming_edges[node]:
                outgoing_connections[edge.from_port.node].append(node)
                if not edge.connects_to_outside:  # don't count incoming, they are ready
                    in_degree[node] += 1

        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        self.sorted_nodes: list[Node] = []
        self.sorted_edges: list[dict[Port, Edge]] = []

        while queue:
            node = queue.popleft()
            self.sorted_nodes.append(node)
            input_port_edge_map = {
                edge.to_port: edge for edge in self.incoming_edges[node]
            }
            self.sorted_edges.append(input_port_edge_map)
            for out_node in outgoing_connections[node]:
                in_degree[out_node] -= 1
                if in_degree[out_node] == 0:
                    queue.append(out_node)

        # If two nodes depend on each other, they can never be added to the
        # queue, hence we can compare the length to check for cycles:
        if len(self.sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in computation graph!")

    def connect(
        self,
        from_: Node | OutputPort,
        to_: Node | InputPort,
    ) -> None:
        """Connect two nodes (which are automatically added to the graph, if they
        are not part of it). When evaluating the graph data will be
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

        # before doing anything, check for validity
        for from_port, to_port in zip(from_ports, to_ports):
            self._check_port_is_free(to_node, to_port)

        # Nodes must be added to the graph
        self.add_node(from_node, check_warning=False)
        self.add_node(to_node, check_warning=False)

        # Configurations should match
        for from_port, to_port in zip(from_ports, to_ports):
            out_config = from_port.data_configuration
            in_config = to_port.data_configuration
            unified_config = out_config.unify(in_config)
            edge = Edge(from_port, to_port, unified_config)
            self.incoming_edges[to_node].append(edge)

    def _check_port_is_free(self, to_node: Node, input_port: InputPort) -> None:
        for e in self.incoming_edges[to_node]:
            if e.to_port == input_port:
                raise ValueError(
                    f"Input port '{input_port.name}' of node '{to_node.name}' is already \
                        connected!"
                )

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

    def connect_from_outside_of_graph(self, from_port: Port, to_port: InputPort):
        """Connects to ports where the 'from_port' is not part of this graph itself.
        The node of the 'to_port' will be added to this graph. The created edge
        will pass data between the ports, but will not increase the in-degree
        of the 'to_port' node (meaning the node can be evaluated at the start,
        since data is read from 'outside' the graph)

        Args:
            from_port (Port): The port of a node where data should come from.
            to_port (InputPort): The port of a node where data should go to.
        """
        to_node = to_port.node
        self._check_port_is_free(to_node, to_port)
        self.add_node(to_node, check_warning=False)
        out_config = from_port.data_configuration
        in_config = to_port.data_configuration
        unified_config = out_config.unify(in_config)
        edge = Edge(from_port, to_port, unified_config, connects_to_outside=True)
        self.incoming_edges[to_node].append(edge)

    def connect_to_outside_of_graph(self, from_port: OutputPort, to_port: Port):
        """Connects to ports where the 'to_port' is not part of this graph itself.
        The node of the 'from_port' will be added to this graph. The created edge
        will pass data between the ports, but will not increase the in-degree
        of the 'from_port' node (meaning the node can be evaluated at the start,
        since data is read from 'outside' the graph)

        Args:
            from_port (OutputPort): The port of a node where data should come from.
            to_port (Port): The port of a node where data should go to.
        """
        from_node = from_port.node
        for e in self.last_outgoing_edges:
            if e.to_port == to_port:
                raise ValueError(f"Output port '{to_port.name}' is already connected!")

        self.add_node(from_node, check_warning=False)
        out_config = from_port.data_configuration
        in_config = to_port.data_configuration
        unified_config = out_config.unify(in_config)
        edge = Edge(from_port, to_port, unified_config, connects_to_outside=True)
        self.last_outgoing_edges.append(edge)

    def disconnect(self, port: InputPort) -> None:
        """Remove an edge from this graph"""
        for edge in self.incoming_edges[port.node]:
            if edge.to_port == port:
                self.incoming_edges[port.node].remove(edge)
                return

    def validate(self) -> None:
        """Validate that all required input ports of each node are connected."""
        for node in self.nodes:
            for port in node.input_ports:
                throw_err = True
                if port.is_required:  # Check if Input is needed
                    for edge in self.incoming_edges[node]:
                        if edge.to_port == port:
                            throw_err = False
                            break
                    if throw_err:
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
        """Run the graph. The data will be passed through the graph according to
        the connections and the computations of the nodes will be executed.
        """
        for node, edges in zip(self.sorted_nodes, self.sorted_edges):
            node.set_mode(mode)
            for in_port in node.input_ports:
                if in_port in edges:
                    in_port.set_value(edges[in_port].from_port.value)
                else:
                    in_port.clear_value()
            node.run()
        for edge in self.last_outgoing_edges:
            edge.to_port.set_value(edge.from_port.value)

    def collect_trainable_parameters(self):
        """Collect all trainable parameters from the nodes in this graph."""
        params_collection = TrainableParametersCollection()
        for node in self.nodes:
            p = node._trainable_parameters
            if not p.empty:
                params_collection.extend(p)
        return params_collection

    @contextmanager
    def tracker(self, n_tracking_vars=1):
        if len(self.nodes) > 0:
            raise RuntimeError(
                "Graph tracking can only be used on an empty graph. Please create a new\
                 graph for tracking or clear the current graph before starting tracking."
            )
        prev_tracked_graph = TrackingObject.current_graph_tracked
        TrackingObject.current_graph_tracked = self

        if prev_tracked_graph is None:
            Node.set_tracking(True)

        tracking_objects = [TrackingObject() for _ in range(n_tracking_vars)]
        try:
            if n_tracking_vars > 1:
                yield tracking_objects if n_tracking_vars > 1 else tracking_objects[0]
            else:
                yield
        finally:
            TrackingObject.current_graph_tracked = prev_tracked_graph
            if TrackingObject.current_graph_tracked is None:
                Node.set_tracking(False)


class SequentialGraph(Graph):
    """
    A graph that is initialized as a sequence of nodes.
    """

    def __init__(self, *nodes: Node):
        super().__init__()
        for i in range(len(nodes) - 1):
            self.connect(nodes[i], nodes[i + 1])
        self.sorted_nodes = list(nodes)

        # If previous loop was skipped (only one node)
        if len(self.sorted_nodes) == 1:
            self.nodes.add(self.sorted_nodes[0])
            self.incoming_edges[self.sorted_nodes[0]] = list[Edge]()


############################################################################
# region: Tracking
class TrackingObject:
    current_graph_tracked: Graph | None = None

    def __init__(self, last_output_port: OutputPort | None = None):
        self.last_output_port: OutputPort | None = last_output_port
        self.to_ports = []

    def add_to_port(self, port: InputPort):
        self.to_ports.append(port)

    def __add__(self, other):
        from ..algorithms.building_blocks import Add

        add_node = Add()
        return add_node(self, other)

    def __matmul__(self, other):
        from ..algorithms.building_blocks import MatMul

        matmul_node = MatMul()
        return matmul_node(self, other)

    def __sub__(self, other):
        from ..algorithms.building_blocks import Subtract

        subtract_node = Subtract()
        return subtract_node(self, other)

    def __mul__(self, other):
        from ..algorithms.building_blocks import Multiply

        multiply_node = Multiply()
        return multiply_node(self, other)

    def __pow__(self, other):
        from ..algorithms.building_blocks import Power

        power_node = Power()
        return power_node(self, other)

    def __truediv__(self, other):
        from ..algorithms.building_blocks import Divide

        divide_node = Divide()
        return divide_node(self, other)

    def __abs__(self):
        from ..algorithms.building_blocks import Abs

        abs_node = Abs()
        return abs_node(self)


# endregion
