from __future__ import annotations
from collections import deque
from contextlib import contextmanager
import inspect
from typing import Callable
from warnings import warn

from qewton.config.data_configurations import DataConfiguration
from qewton.config.errors import DataConfigMismatchError

from qewton.graphs.nodes import InputPort, Node, EvaluationPhase, OutputPort, Port
from qewton.optim.parameters.trainable_parameters import TrainableParametersCollection
from qewton.graphs.edges import Edge


class Graph:
    """
    Represents a directed acyclic graph (DAG) of interconnected nodes.
    """

    def __init__(self):
        self.nodes: set[Node] = set[Node]()
        self.sorted_nodes: list[Node] = []
        self.graph_was_sorted = False
        self.mode = EvaluationPhase.ALWAYS

        self.incoming_edges: dict[Node, list[Edge]] = {}
        self.outgoing_edges: dict[Node, list[Edge]] = {}
        self.skip_connections: list[Edge] = []

        self.sorted_incoming_edges: list[dict[Port, Edge]] = []

        self.edges_from_outside: list[Edge] = []
        self.edges_to_outside: list[Edge] = []

        self.dynamic_data_configs: dict[Node, dict[Port, DataConfiguration]] = {}

    @classmethod
    def from_function(
        cls, func: Callable
    ) -> tuple[Graph, list[list[InputPort] | InputPort], list[OutputPort | int]]:
        """
        Creates a graph by tracking the execution of a given function.

        This method allows defining a graph structure programmatically by
        using `TrackingObject`s as inputs and outputs within the function.

        Args:
            func (Callable): The function whose execution will be tracked to build the
                graph.

        Returns:
            tuple[Graph, list[list[InputPort] | InputPort], list[OutputPort | int]]:
                A tuple containing the constructed graph, its input ports, and its output
                ports.
        """
        graph = Graph()
        sig = inspect.signature(func)
        if len(sig.parameters) > 0:
            with graph.tracker(n_tracking_vars=len(sig.parameters)) as tracking_vars:
                if isinstance(tracking_vars, TrackingObject):
                    tracking_vars = (tracking_vars,)
                out = func(*tracking_vars)  # type: ignore
            input_ports = [var.to_ports for var in tracking_vars]  # type: ignore
        else:
            with graph.tracker():
                out = func()
            input_ports = []
            tracking_vars = ()
        tracking_vars_idcs = {
            var: i for i, var in enumerate(tracking_vars)  # type: ignore
        }
        if isinstance(out, tuple):
            assert all(
                isinstance(o, TrackingObject) for o in out
            ), "All outputs of the function must be TrackingObjects."
            output_ports = [
                (
                    var.last_output_port
                    if var.last_output_port is not None
                    else tracking_vars_idcs[var]
                )
                for var in out
            ]
        else:
            if out is None:
                output_ports = []
            else:
                assert isinstance(
                    out, TrackingObject
                ), "The output of the function must be a TrackingObject."
                output_ports = (
                    [out.last_output_port]
                    if out.last_output_port is not None
                    else [tracking_vars_idcs[out]]
                )

        return graph, input_ports, output_ports  # type: ignore

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
                        f"Node {node.name} and ID: {node.node_id} already exists in this\
                            graph!"
                    )
        self.nodes.add(node)
        if not node in self.incoming_edges:
            self.incoming_edges[node] = list[Edge]()
            self.outgoing_edges[node] = list[Edge]()
            self.dynamic_data_configs[node] = node.copy_data_configs()

    def sort(self):
        """
        Sorts the nodes in the graph topologically (Kahn's algorithm) to determine the
        execution order.

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        in_degree = {node: 0 for node in self.nodes}
        outgoing_connections = {node: [] for node in self.nodes}
        for node in self.nodes:
            for edge in self.incoming_edges[node]:
                if not edge.connects_to_outside:  # don't count incoming, they are ready
                    in_degree[node] += 1
                    outgoing_connections[edge.from_port.node].append(node)

        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        self.sorted_nodes: list[Node] = []
        self.sorted_incoming_edges: list[dict[Port, Edge]] = []

        while queue:
            node = queue.popleft()
            self.sorted_nodes.append(node)
            input_port_edge_map = {
                edge.to_port: edge for edge in self.incoming_edges[node]
            }
            self.sorted_incoming_edges.append(input_port_edge_map)
            for out_node in outgoing_connections[node]:
                in_degree[out_node] -= 1
                if in_degree[out_node] == 0:
                    queue.append(out_node)

        # If two nodes depend on each other, they can never be added to the
        # queue, hence we can compare the length to check for cycles:
        if len(self.sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in computation graph!")
        self.graph_was_sorted = True

    def connect(
        self,
        from_: Node | OutputPort,
        to_: Node | InputPort,
    ) -> None:
        """
        Connects two nodes or ports within the graph.

        If nodes are passed directly, it assumes a single output from `from_`
        and a single input to `to_`. It adds nodes to the graph if they are not
        already present and unifies their data configurations.

        Args:
            from_ (Node | OutputPort): The source node or output port.
            to_ (Node | InputPort): The destination node or input port.

        Raises:
            ValueError: If the number of ports do not match or an input port is
                already connected.
            DataConfigMismatchError: If the data configurations of the connected
                ports are incompatible.
        """
        self._check_graph_was_sorted()

        def get_ports(
            inp, return_output_ports=False
        ) -> list[InputPort] | list[OutputPort]:
            if isinstance(inp, Node):
                if return_output_ports:
                    return inp.output_ports
                return inp.input_ports
            return [inp]

        from_ports = get_ports(from_, return_output_ports=True)
        to_ports = get_ports(to_, return_output_ports=False)
        if len(from_ports) != len(to_ports):
            raise ValueError(
                "When connecting two nodes directly, they should have the"
                + "same number of output and input ports!"
            )

        from_node = from_ports[0].node
        to_node = to_ports[0].node

        # before doing anything, check for validity
        for to_port in to_ports:
            self._check_port_is_free(to_node, to_port)  # type: ignore

        # Nodes must be added to the graph
        self.add_node(from_node, check_warning=False)
        self.add_node(to_node, check_warning=False)

        # Configurations should match
        for from_port, to_port in zip(from_ports, to_ports):
            from_config = self.dynamic_data_configs[from_node][from_port]
            to_config = self.dynamic_data_configs[to_node][to_port]
            try:
                unified_config = from_config.unify_with(to_config)
            except DataConfigMismatchError as e:
                raise DataConfigMismatchError(
                    f"Connection of {from_node.name} to {to_node.name} failed, "
                    + f"because {e}"
                ) from e
            edge = Edge(from_port, to_port)
            self.incoming_edges[to_node].append(edge)
            self.outgoing_edges[from_node].append(edge)

            self.update_data_configurations(from_node, from_port, unified_config[0])
            self.update_data_configurations(to_node, to_port, unified_config[1])

    def update_data_configurations(self, node: Node, port: Port, config_dict: dict):
        """Updates the data configurations recursively for the given node, port,
        and neighbors.

        Args:
            node (Node): The node for which the data configurations should be updated.
            port (Port): The port of the node for which the data configuration should
                be updated.
            config_dict (dict): Mapping of configuration keys/values used to update the
                node and neighbor data configurations.
        """
        # visited_nodes = set[Node]({node}) # experimental, check whether this
        # iterates forever
        # visited_edges = set[Edge]({edge})
        updated_ports = node.update_data_configs(
            port, config_dict, self.dynamic_data_configs[node]
        )
        # If the config was made more concrete we have to pass this information to
        # the remaining graph
        if len(updated_ports) > 0 and node in self.nodes:
            for e in self.incoming_edges[node] + self.outgoing_edges[node]:
                if e.to_port not in updated_ports and e.from_port not in updated_ports:
                    continue
                # do not update outside of this graph context
                if e.connects_to_outside:
                    continue
                # if e not in visited_edges:
                first_config = self.dynamic_data_configs[e.from_port.node][e.from_port]
                second_config = self.dynamic_data_configs[e.to_port.node][e.to_port]
                try:
                    new_config_dict = first_config.unify_with(second_config)
                except DataConfigMismatchError as err:
                    raise DataConfigMismatchError(
                        f"Reconnection of {e.from_port.node.name} to "
                        + f"{e.to_port.node.name} failed, because {err}"
                    ) from err

                # visited_edges.add(e)
                # and e.to_port.node not in visited_nodes
                updated_ports |= self.update_data_configurations(
                    e.from_port.node, e.from_port, new_config_dict[0]
                )
                updated_ports |= self.update_data_configurations(
                    e.to_port.node, e.to_port, new_config_dict[1]
                )
        return updated_ports

    def _check_graph_was_sorted(self):
        """
        Checks if the graph has already been sorted.
        If it has, a warning is issued because adding new edges might invalidate the
        current sort order.
        """
        if self.graph_was_sorted:
            warn(
                "The graph was already sorted. Inputting a new edge may change the "
                "evaluation order, and the graph should be resorted."
            )

    def _check_port_is_free(self, to_node: Node, input_port: InputPort) -> None:
        """
        Checks if a given input port of a node is already connected.

        Args:
            to_node (Node): The node whose input port is being checked.
            input_port (InputPort): The specific input port to check.

        Raises:
            ValueError: If the input port is already connected.
        """
        if to_node in self.incoming_edges:
            for e in self.incoming_edges[to_node]:
                if e.to_port == input_port:
                    raise ValueError(
                        f"Input port '{input_port.name}' of node '{to_node.name}' is \
                            already connected!"
                    )

    def _check_connect(
        self, user_input: InputPort | OutputPort | Node, check_input: bool = True
    ) -> InputPort | OutputPort:
        """
        Helper method to extract a single port from a user input (Node or Port).

        Args:
            user_input (InputPort | OutputPort | Node): The input provided by the user.
            check_input (bool, optional): If True, checks input ports; otherwise, checks
                output ports. Defaults to True.

        Returns:
            InputPort | OutputPort: The extracted single port.

        Raises:
            ValueError: If a Node with multiple ports is provided and a specific port i
                not specified.
        """
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
        self._check_graph_was_sorted()
        to_node = to_port.node
        self._check_port_is_free(to_node, to_port)
        self.add_node(to_node, check_warning=False)

        edge = Edge(from_port, to_port, connects_to_outside=True)
        self.edges_from_outside.append(edge)
        self.incoming_edges[to_node].append(edge)

        from_node = from_port.node
        self._check_external_node_in_config_dict(from_port, from_node)

        to_config = self.dynamic_data_configs[to_node][to_port]
        from_config = self.dynamic_data_configs[from_node][from_port]

        try:
            _, unified_to_config = from_config.unify_with(to_config)
        except DataConfigMismatchError as e:
            self.edges_from_outside.remove(edge)
            self.incoming_edges[to_node].remove(edge)
            raise DataConfigMismatchError(
                f"Connection of {from_node.name} to {to_node.name} failed: {e}"
            ) from e

        self.update_data_configurations(to_node, to_port, unified_to_config)
        self.update_data_configurations(from_node, from_port, unified_to_config)

    def connect_to_outside_of_graph(self, from_port: OutputPort, to_port: Port):
        """Connects to ports where the 'to_port' is not part of this graph itself.
        The node of the 'from_port' will be added to this graph. The created edge
        will pass data from an internal output port to an external port.

        Args:
            from_port (OutputPort): The port of a node where data should come from.
            to_port (Port): The port of a node where data should go to.
        """
        self._check_graph_was_sorted()
        from_node = from_port.node
        self.add_node(from_node, check_warning=False)
        edge = Edge(from_port, to_port, connects_to_outside=True)
        self.edges_to_outside.append(edge)
        self.outgoing_edges[from_node].append(edge)

        to_node = to_port.node
        self._check_external_node_in_config_dict(to_port, to_node)

        from_config = self.dynamic_data_configs[from_node][from_port]
        to_config = self.dynamic_data_configs[to_node][to_port]

        try:
            unified_from_config, _ = from_config.unify_with(to_config)
        except DataConfigMismatchError as e:
            self.edges_to_outside.remove(edge)
            self.outgoing_edges[from_node].remove(edge)
            raise DataConfigMismatchError(
                f"Connection of {from_node.name} to {to_node.name} failed: {e}"
            ) from e

        self.update_data_configurations(to_node, to_port, unified_from_config)
        self.update_data_configurations(from_node, from_port, unified_from_config)

    def _check_external_node_in_config_dict(self, port: Port, node: Node):
        """
        Ensures that an external node and its port have an entry in the dynamic
        data configurations.
        This is crucial when connecting to or from nodes that are not formally part
        of `self.nodes`.

        Args:
            port (Port): The port of the external node.
            node (Node): The external node.
        """
        if node not in self.dynamic_data_configs:
            self.dynamic_data_configs[node] = node.copy_data_configs()
        if port not in self.dynamic_data_configs[node]:
            self.dynamic_data_configs[node][port] = node.copy_data_config_of_port(
                port, {}
            )

    def add_skip_connection(self, from_port: Port, to_port: Port):
        """Adds a skip connection between two ports."""
        to_node = to_port.node
        from_node = from_port.node
        self._check_external_node_in_config_dict(to_port, to_node)
        self._check_external_node_in_config_dict(from_port, from_node)

        edge = Edge(from_port, to_port, connects_to_outside=True)
        self.skip_connections.append(edge)

        self.dynamic_data_configs[from_node][from_port] = self.dynamic_data_configs[
            to_node
        ][to_port]

    def validate(self) -> None:
        """
        Validates the graph structure.
        Checks if all required input ports of every node in the graph are connected.

        Raises:
            ValueError: If a required input port is found to be unconnected.
        """
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
        """
        Runs the graph execution by iterating through the topologically sorted nodes.
        """
        for node, edges in zip(self.sorted_nodes, self.sorted_incoming_edges):
            node.set_mode(mode)
            for in_port in node.input_ports:
                if in_port in edges:
                    in_port.set_value(edges[in_port].from_port.value)
                else:
                    in_port.clear_value()
            node.run()
        for edge in self.edges_to_outside + self.skip_connections:
            edge.to_port.set_value(edge.from_port.value)

    def collect_trainable_parameters(self):
        """
        Collects all trainable parameters from all nodes within the graph.

        Returns:
            TrainableParametersCollection: A collection of all trainable parameters.
        """
        params_collection = TrainableParametersCollection()
        for node in self.nodes:
            p = node._trainable_parameters
            if not p.empty:
                params_collection.extend(p)
        return params_collection

    @contextmanager
    def tracker(self, n_tracking_vars=1):
        """
        A context manager for tracking graph construction.

        When active, nodes will record connections and data flow, allowing a graph
        to be built implicitly from function calls.

        Args:
            n_tracking_vars (int, optional): The number of tracking variables to yield.
                Defaults to 1.

        Yields:
            TrackingObject | tuple[TrackingObject, ...]: One or more tracking objects.
        """
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
            if n_tracking_vars >= 1:
                yield tracking_objects if n_tracking_vars > 1 else tracking_objects[0]
            else:
                yield
        finally:
            TrackingObject.current_graph_tracked = prev_tracked_graph
            if TrackingObject.current_graph_tracked is None:
                Node.set_tracking(False)


class SequentialGraph(Graph):
    """
    A graph specifically designed to represent a linear sequence of nodes.

    Nodes are connected in the order they are provided during initialization.
    """

    def __init__(self, *nodes: Node):
        super().__init__()
        for i in range(len(nodes) - 1):
            self.connect(nodes[i], nodes[i + 1])
        self.sorted_nodes = list(nodes)

        # If previous loop was skipped (only one node)
        if len(self.sorted_nodes) == 1:
            self.add_node(self.sorted_nodes[0])


############################################################################
# region: Tracking
class TrackingObject:
    current_graph_tracked: Graph | None = None
    """
    A special object used during graph tracking to represent data flow.

    When nodes are called with `TrackingObject`s as inputs, the graph
    automatically records the connections.
    """

    def __init__(self, last_output_port: OutputPort | None = None):
        self.last_output_port: OutputPort | None = last_output_port
        self.to_ports = []

    def add_to_port(self, port: InputPort):
        self.to_ports.append(port)

    def __add__(self, other):
        from qewton.algorithms.building_blocks.math import Add

        add_node = Add()
        return add_node(self, other)

    def __radd__(self, other):
        from qewton.algorithms.building_blocks.math import Add

        add_node = Add()
        return add_node(other, self)

    def __matmul__(self, other):
        from qewton.algorithms.building_blocks.math import MatMul

        matmul_node = MatMul()
        return matmul_node(self, other)

    def __sub__(self, other):
        from qewton.algorithms.building_blocks.math import Subtract

        subtract_node = Subtract()
        return subtract_node(self, other)

    def __rsub__(self, other):
        from qewton.algorithms.building_blocks.math import Subtract

        subtract_node = Subtract()
        return subtract_node(other, self)

    def __mul__(self, other):
        from qewton.algorithms.building_blocks.math import Multiply

        multiply_node = Multiply()
        return multiply_node(self, other)

    def __rmul__(self, other):
        from qewton.algorithms.building_blocks.math import Multiply

        multiply_node = Multiply()
        return multiply_node(other, self)

    def __pow__(self, other):
        from qewton.algorithms.building_blocks.math import Power

        power_node = Power()
        return power_node(self, other)

    def __rpow__(self, other):
        from qewton.algorithms.building_blocks.math import Power

        power_node = Power()
        return power_node(other, self)

    def __truediv__(self, other):
        from qewton.algorithms.building_blocks.math import Divide

        divide_node = Divide()
        return divide_node(self, other)

    def __rtruediv__(self, other):
        from qewton.algorithms.building_blocks.math import Divide

        divide_node = Divide()
        return divide_node(other, self)

    def __abs__(self):
        from qewton.algorithms.building_blocks.math import Abs

        abs_node = Abs()
        return abs_node(self)

    def __getitem__(self, key):
        from qewton.algorithms.building_blocks.array_operations import Slice

        slice_node = Slice(key)
        return slice_node(self)

    def gradient(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import Gradient

        grad_node = Gradient()
        return grad_node(self, with_respect_to)

    def normal_derivative(self, with_respect_to, normals):
        from qewton.algorithms.building_blocks.derivatives import NormalDerivative

        norm_der_node = NormalDerivative()
        return norm_der_node(self, with_respect_to, normals)

    def laplacian(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import Laplacian

        lap_node = Laplacian()
        return lap_node(self, with_respect_to)

    def jac(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import Jacobian

        jac_node = Jacobian()
        return jac_node(self, with_respect_to)

    def div(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import Divergence

        div_node = Divergence()
        return div_node(self, with_respect_to)

    def hessian(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import Hessian

        hes_node = Hessian()
        return hes_node(self, with_respect_to)

    def sym_grad(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import SymmetricGradient

        sym_grad_node = SymmetricGradient()
        return sym_grad_node(self, with_respect_to)

    def rot(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import Rotation

        rot_node = Rotation()
        return rot_node(self, with_respect_to)

    def matrix_div(self, with_respect_to):
        from qewton.algorithms.building_blocks.derivatives import MatrixDivergence

        matrix_div_node = MatrixDivergence()
        return matrix_div_node(self, with_respect_to)


# endregion
