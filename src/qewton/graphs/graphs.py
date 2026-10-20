"""Computation graphs (DAGs) of nodes and the tracking objects that build
them from plain Python functions."""

from __future__ import annotations
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, TYPE_CHECKING
from warnings import warn
import inspect
import numpy as np

from qewton.config.axes import FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.devices import Device
from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable
from qewton.config.saving.saving import Serializable
from qewton.graphs.nodes import (
    GraphAwareNode,
    InputPort,
    Node,
    EvaluationPhase,
    OutputPort,
    Port,
)
from qewton.graphs.control_nodes.data_processing_node import DataProcessingNode
from qewton.optim.parameters.trainable_parameters import TrainableParametersCollection
from qewton.graphs.edges import Edge

if TYPE_CHECKING:
    # qewton.visualization and qewton.data.dataloaders.sampler.point_sampler
    # transitively import this module, so they are only imported for type
    # checking here and locally inside the methods that need them.
    from qewton.visualization.plots.base import Plot
    from qewton.visualization.plots.graph import GraphPlot
    from qewton.visualization.layout import Layout
    from qewton.visualization.plots.spec import Scale


class Graph(Serializable):
    """Directed acyclic graph (DAG) of interconnected nodes.

    Nodes are connected port to port via :class:`~qewton.graphs.edges.Edge`
    objects. Every connection unifies the data configurations of both
    ports; the unified configurations are stored per graph in
    `dynamic_data_configs`, the static configurations of the ports stay
    untouched.

    Attributes:
        nodes (set[Node]): All nodes of this graph.
        sorted_nodes (list[Node]): The nodes in execution order, filled by
            :meth:`sort`.
        sorted_incoming_edges (list[dict[Port, Edge]]): For each entry of
            `sorted_nodes`, its incoming edges keyed by the input port they end
            in.
        graph_was_sorted (bool): Whether :meth:`sort` was called since the last
            structural change.
        mode (EvaluationPhase): The evaluation phase of this graph.
        incoming_edges (dict[Node, list[Edge]]): Incoming edges of each node,
            including edges from outside of the graph.
        outgoing_edges (dict[Node, list[Edge]]): Outgoing edges of each node,
            including edges to outside of the graph.
        skip_connections (list[Edge]): Edges added by
            :meth:`add_skip_connection`.
        edges_from_outside (list[Edge]): Edges from a port outside of this
            graph into one of its nodes.
        edges_to_outside (list[Edge]): Edges from one of its nodes to a port
            outside of this graph.
        dynamic_data_configs (dict[Node, dict[Port, DataConfiguration]]): The
            unified data configuration of every port known to this graph.
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
        """Build a graph by tracking the execution of `func`.

        `func` is called once with one :class:`TrackingObject` per
        parameter; every node called on them is recorded in the new graph
        (see :meth:`tracker`).

        Args:
            func: The function to track. Its parameters become the graph
                inputs, its returned tracking objects the graph outputs.

        Returns:
            tuple: A tuple ``(graph, input_ports, output_ports)`` of

                - graph (Graph): The constructed graph.
                - input_ports (list[list[InputPort] | InputPort]): For each
                  parameter of `func`, the input ports it was passed to.
                - output_ports (list[OutputPort | int]): For each returned
                  value, the output port producing it, or the index of the
                  parameter if an input is returned unchanged.

        Raises:
            ValueError: If a parameter of `func` is neither used nor returned.
        """
        graph = Graph()
        sig = inspect.signature(func)
        not_used_inputs = []
        input_ports = []
        if len(sig.parameters) > 0:
            with graph.tracker(n_tracking_vars=len(sig.parameters)) as tracking_vars:
                if isinstance(tracking_vars, TrackingObject):
                    tracking_vars = (tracking_vars,)
                out = func(*tracking_vars)  # type: ignore
            if not tracking_vars is None:
                for i, var in enumerate(tracking_vars):
                    if len(var.to_ports) == 0:
                        not_used_inputs.append(i)
                input_ports = [var.to_ports for var in tracking_vars]  # type: ignore
        else:
            with graph.tracker():
                out = func()
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
        for i in not_used_inputs:
            if i not in output_ports:
                sig_value = list(sig.parameters.values())[i]
                raise ValueError(f"""Input {sig_value} of function {func.__name__} was 
                    not used in the function body. Remove it from the function 
                    if this is desired.""")

        return graph, input_ports, output_ports  # type: ignore

    def add_node(self, node: Node, check_warning=True) -> None:
        """Add a node to this graph.

        Adding a node that is already part of the graph keeps its edges
        and data configurations.

        Args:
            node: The node to add.
            check_warning (bool, optional): Whether to check that no node with
                the same `node_id` is already part of this graph. Default is
                True.

        Raises:
            ValueError: If `check_warning` is True and a node with the same
                `node_id` already exists in this graph.
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
        """Sort the nodes topologically to determine the execution order.

        Uses Kahn's algorithm. Edges from outside of the graph do not
        count towards a node's in-degree, since their data is already
        available when the graph starts.

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
        """Connect two nodes or ports within the graph.

        If nodes are passed, all output ports of `from_` are connected to
        the input ports of `to_` in order. Both nodes are added to the
        graph if needed, and the data configurations of every connected
        port pair are unified. Warns if the graph was already sorted.

        Args:
            from_: The source node or output port.
            to_: The destination node or input port.

        Raises:
            ValueError: If the numbers of ports do not match or an input port
                is already connected.
            DataConfigMismatchError: If the data configurations of the
                connected ports are incompatible.
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

        from_node = from_ports[0].node
        to_node = to_ports[0].node

        if len(from_ports) != len(to_ports):
            raise ValueError(
                "When connecting two nodes directly, they should have the "
                + f"same number of output and input ports! Got node {from_node.name}"
                + f" with {len(from_ports)} output ports and node {to_node.name}"
                + f" with {len(to_ports)} input ports!"
            )

        # before doing anything, check for validity
        for to_port in to_ports:
            self._check_port_is_free(to_node, to_port)  # type: ignore

        # Nodes must be added to the graph
        self.add_node(from_node, check_warning=False)
        self.add_node(to_node, check_warning=False)

        # Configurations should match
        for from_port, to_port in zip(from_ports, to_ports):
            unified_config = self._unify_configs(from_node, to_node, to_port, from_port)
            edge = Edge(from_port, to_port)
            self.incoming_edges[to_node].append(edge)
            self.outgoing_edges[from_node].append(edge)

            self.update_data_configurations(from_node, from_port, unified_config[0])
            self.update_data_configurations(to_node, to_port, unified_config[1])

    def _unify_configs(self, from_node, to_node, to_port, from_port):
        from_config = self.dynamic_data_configs[from_node][from_port]
        to_config = self.dynamic_data_configs[to_node][to_port]
        try:
            unified_config = from_config.unify_with(to_config)
        except DataConfigMismatchError as e:
            raise DataConfigMismatchError(
                f"Connection of {from_node.name} to {to_node.name} failed, "
                + f"because {e}"
            ) from e

        return unified_config

    def remove(self, obj: Node | Edge):
        """Remove a node or edge from the graph.

        Removing a node also removes all edges connected to it. All data
        configurations are recalculated afterwards and the graph is marked
        as unsorted.

        Args:
            obj: The node or edge to remove.
        """
        if isinstance(obj, Node):
            if obj in self.nodes:
                self.nodes.remove(obj)
                self.incoming_edges.pop(obj, None)
                self.outgoing_edges.pop(obj, None)
                self.dynamic_data_configs.pop(obj, None)
                # Remove edges connected to this node
                self._remove_node_connections(self.edges_from_outside, obj)
                self._remove_node_connections(self.edges_to_outside, obj)
                self._remove_node_connections(self.skip_connections, obj)
                # Remove also all edges that other nodes have to this node
                for node in self.nodes:
                    self._remove_node_connections(self.incoming_edges[node], obj)
                    self._remove_node_connections(self.outgoing_edges[node], obj)
        else:
            # Remove the edge from the graph
            if obj in self.edges_from_outside:
                self.edges_from_outside.remove(obj)
                self.incoming_edges[obj.to_port.node].remove(obj)
            elif obj in self.edges_to_outside:
                self.edges_to_outside.remove(obj)
                self.outgoing_edges[obj.from_port.node].remove(obj)
            elif obj in self.skip_connections:
                self.skip_connections.remove(obj)
            else:
                # Remove from incoming and outgoing edges of nodes
                if obj.to_port.node in self.incoming_edges:
                    if obj in self.incoming_edges[obj.to_port.node]:
                        self.incoming_edges[obj.to_port.node].remove(obj)
                if obj.from_port.node in self.outgoing_edges:
                    if obj in self.outgoing_edges[obj.from_port.node]:
                        self.outgoing_edges[obj.from_port.node].remove(obj)
        # Unification cannot be undone, so the configurations are rebuilt
        # from the static ones.
        self._recalculate_data_configurations()
        # A graph is never sorted after a node or edge was removed, since the
        # order might have changed.
        self.graph_was_sorted = False
        self.sorted_nodes = []
        self.sorted_incoming_edges = []

    def duplicate(self) -> Graph:
        """Create a copy of this graph's structure.

        The copy shares the node objects but has its own edges and data
        configurations. It is sorted if this graph is sorted.

        Returns:
            Graph: A new graph with the same nodes and connections.
        """

        def edge_copy(edge: Edge, edge_mapping: dict[Edge, Edge]) -> Edge:
            if not edge in edge_mapping:
                new_edge = Edge(
                    edge.from_port,
                    edge.to_port,
                    connects_to_outside=edge.connects_to_outside,
                )
                edge_mapping[edge] = new_edge
            return edge_mapping[edge]

        new_graph = Graph()
        for node in self.nodes:
            new_graph.add_node(node, check_warning=False)
        # Copy edges and connections
        edge_mapping = {}
        for node in self.nodes:
            for edge in self.incoming_edges[node]:
                new_graph.incoming_edges[node].append(edge_copy(edge, edge_mapping))
            for edge in self.outgoing_edges[node]:
                new_graph.outgoing_edges[node].append(edge_copy(edge, edge_mapping))
            for edge in self.skip_connections:
                new_graph.skip_connections.append(edge_copy(edge, edge_mapping))
        for edge in self.edges_from_outside:
            new_graph.edges_from_outside.append(edge_copy(edge, edge_mapping))
        for edge in self.edges_to_outside:
            new_graph.edges_to_outside.append(edge_copy(edge, edge_mapping))
        # Copy dynamic data configurations
        config_dict = {}
        for node, port_configs in self.dynamic_data_configs.items():
            new_graph.dynamic_data_configs[node] = {}
            for port, config in port_configs.items():
                new_graph.dynamic_data_configs[node][port] = deepcopy(
                    config, memo=config_dict
                )
        # Sorting
        if self.graph_was_sorted:
            new_graph.sort()
        return new_graph

    def replace_node(
        self,
        old_node: Node,
        new_node: Node,
        input_port_mapping: dict[Port, Port] | None = None,
        output_port_mapping: dict[Port, Port] | None = None,
    ):
        """Replace a node of the graph with a new node.

        All edges and skip connections of `old_node` are transferred to
        `new_node`, the data configurations are recalculated and the graph
        is marked as unsorted.

        Args:
            old_node: The node to replace.
            new_node: The node that replaces `old_node`.
            input_port_mapping: Maps input ports of `old_node` to input ports
                of `new_node`. None (default) matches them by order, which
                requires both nodes to have the same number of input ports.
            output_port_mapping: Maps output ports of `old_node` to output
                ports of `new_node`. None (default) matches them by order,
                which requires both nodes to have the same number of output
                ports.

        Raises:
            ValueError: If `old_node` is not part of the graph, `new_node`
                already is, or the numbers of ports do not match without a
                mapping.
        """
        if old_node not in self.nodes:
            raise ValueError(f"Node {old_node.name} is not part of this graph.")
        if new_node in self.nodes:
            raise ValueError(f"Node {new_node.name} is already part of this graph.")

        # Build the mapping between ports
        if input_port_mapping is None:
            if len(old_node.input_ports) != len(new_node.input_ports):
                raise ValueError(
                    f"Old node {old_node.name} and new node {new_node.name} must have "
                    "the same number of input ports."
                )
            input_port_mapping = {
                old_port: new_port
                for old_port, new_port in zip(old_node.input_ports, new_node.input_ports)
            }
        if output_port_mapping is None:
            if len(old_node.output_ports) != len(new_node.output_ports):
                raise ValueError(
                    f"Old node {old_node.name} and new node {new_node.name} must have "
                    "the same number of output ports."
                )
            output_port_mapping = {
                old_port: new_port
                for old_port, new_port in zip(
                    old_node.output_ports, new_node.output_ports
                )
            }

        # Add the new node to the graph
        self.add_node(new_node, check_warning=False)

        # Transfer incoming edges from old_node to new_node
        for edge in self.incoming_edges[old_node]:
            edge.to_port.node = new_node
            edge.to_port = input_port_mapping[edge.to_port]
            self.incoming_edges[new_node].append(edge)
        self.incoming_edges.pop(old_node)

        # Transfer outgoing edges from old_node to new_node
        for edge in self.outgoing_edges[old_node]:
            edge.from_port.node = new_node
            edge.from_port = output_port_mapping[edge.from_port]
            self.outgoing_edges[new_node].append(edge)
        self.outgoing_edges.pop(old_node)

        # Transfer skip connections involving old_node to new_node
        for edge in list(self.skip_connections):
            if edge.from_port.node == old_node:
                edge.from_port.node = new_node
                edge.from_port = output_port_mapping[edge.from_port]
            if edge.to_port.node == old_node:
                edge.to_port.node = new_node
                edge.to_port = input_port_mapping[edge.to_port]

        # Update dynamic data configurations
        self.dynamic_data_configs.pop(old_node)
        self.dynamic_data_configs[new_node] = new_node.copy_data_configs()

        # Remove the old node from the graph
        self.nodes.remove(old_node)

        # Recalculate data configurations after replacement
        self._recalculate_data_configurations()

        # Mark the graph as unsorted since the structure has changed
        self.graph_was_sorted = False

    def _remove_node_connections(self, edge_container: list[Edge], node: Node):
        """Remove all edges touching `node` from `edge_container` in place."""
        edge_list = list(edge_container)  # Create a copy
        for edge in edge_list:
            if edge.from_port.node == node or edge.to_port.node == node:
                edge_container.remove(edge)

    def _recalculate_data_configurations(self):
        # Reset all data configurations
        for node in self.dynamic_data_configs.keys():
            self.dynamic_data_configs[node] = node.copy_data_configs()
        # Pass information over the edges
        for edges in self.incoming_edges.values():
            for edge in edges:
                from_node, from_port, to_node, to_port = edge.endpoints()
                unified_config = self._unify_configs(
                    from_node, to_node, to_port, from_port
                )
                self.update_data_configurations(from_node, from_port, unified_config[0])
                self.update_data_configurations(to_node, to_port, unified_config[1])
        for edge in self.skip_connections:
            from_node, from_port, to_node, to_port = edge.endpoints()
            self.dynamic_data_configs[from_node][from_port] = self.dynamic_data_configs[
                to_node
            ][to_port]
        for edge in self.edges_from_outside:
            self._propagate_outside_edge(edge, keep="to")
        for edge in self.edges_to_outside:
            self._propagate_outside_edge(edge, keep="from")

    def _propagate_outside_edge(self, edge: Edge, keep="to"):
        from_node, from_port, to_node, to_port = edge.endpoints()
        from_config = self.dynamic_data_configs[from_node][from_port]
        to_config = self.dynamic_data_configs[to_node][to_port]
        from_result, to_result = from_config.unify_with(to_config)
        unified = to_result if keep == "to" else from_result
        self.update_data_configurations(from_node, from_port, unified)
        self.update_data_configurations(to_node, to_port, unified)

    def update_data_configurations(self, node: Node, port: Port, config_dict: dict):
        """Update the data configuration of `port` and propagate the change.

        Every port of `node` whose configuration became more concrete is
        unified with the ports it is connected to inside this graph, and
        the update continues recursively from there. Edges to or from
        outside of the graph are not followed.

        Args:
            node: The node owning `port`.
            port: The port whose data configuration is updated.
            config_dict: Maps axes to their replacements, as returned by
                :meth:`~qewton.config.data_configurations.base.DataConfiguration.unify_with`.

        Returns:
            set[Port]: All ports whose configuration changed.

        Raises:
            DataConfigMismatchError: If a propagated configuration is
                incompatible with a connected port.
        """
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
        """Warn if the graph is sorted, since a new edge may invalidate the
        order."""
        if self.graph_was_sorted:
            warn(
                "The graph was already sorted. Inputting a new edge may change the "
                "evaluation order, and the graph should be resorted."
            )

    def _check_port_is_free(self, to_node: Node, input_port: InputPort) -> None:
        """Raise a ValueError if `input_port` already has an incoming edge."""
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
        """Return `user_input` if it is a port, else the single input
        (`check_input`) or output port of the node; raise a ValueError if
        the node has more than one."""
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
        """Connect a port outside of this graph to an input port inside it.

        The node of `to_port` is added to this graph. The created edge
        passes data between the ports but does not increase the in-degree
        of the node of `to_port`, so that node can be evaluated at the
        start of a run.

        Args:
            from_port: The port outside of this graph that provides the data.
            to_port: The input port that receives the data.

        Raises:
            ValueError: If `to_port` is already connected.
            DataConfigMismatchError: If the data configurations of the ports
                are incompatible. The edge is not added in this case.
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
        """Connect an output port inside this graph to a port outside of it.

        The node of `from_port` is added to this graph. The value of
        `from_port` is written to `to_port` at the end of :meth:`run`.

        Args:
            from_port: The output port inside this graph that provides the
                data.
            to_port: The port outside of this graph that receives the data.

        Raises:
            DataConfigMismatchError: If the data configurations of the ports
                are incompatible. The edge is not added in this case.
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
        """Make sure `node` and `port` have an entry in
        `dynamic_data_configs`, also if `node` is not part of `nodes`."""
        if node not in self.dynamic_data_configs:
            self.dynamic_data_configs[node] = node.copy_data_configs()
        if port not in self.dynamic_data_configs[node]:
            self.dynamic_data_configs[node][port] = node.copy_data_config_of_port(
                port, {}
            )

    def add_skip_connection(self, from_port: Port, to_port: Port):
        """Add a skip connection between two ports.

        The edge is not part of the topological order; the value of
        `from_port` is written to `to_port` at the end of :meth:`run`. The
        data configuration of `from_port` in this graph is set to the one
        of `to_port`.

        Args:
            from_port: The port providing the data.
            to_port: The port receiving the data.
        """
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
        """Check that every required input port of every node is connected.

        Raises:
            ValueError: If a required input port is not connected.
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
        """Sort the graph and set up all nodes for evaluation.

        A :class:`~qewton.graphs.nodes.GraphAwareNode` receives this graph
        in its `setup`. A
        :class:`~qewton.graphs.control_nodes.data_processing_node.DataProcessingNode`
        also receives this graph, but is only set up if its data source
        node is part of this graph.
        """
        self.sort()
        for node in self.sorted_nodes:
            if isinstance(node, DataProcessingNode):
                if node.data_source_node in self.nodes:
                    node.setup(self)
            elif isinstance(node, GraphAwareNode):
                node.setup(self)
            else:
                node.setup()

    def run(self, mode: EvaluationPhase = EvaluationPhase.ALWAYS):
        """Run all nodes in topological order.

        Before a node runs, each of its input ports receives the value of
        its incoming edge, or its default if it has none. After all nodes
        ran, values are passed along edges to outside of the graph and
        skip connections. Requires :meth:`sort` (or :meth:`setup`) first.

        Args:
            mode: The evaluation phase every node is set to.
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

    def _build_path_to_node(self, node: Node) -> set[Node]:
        """Unordered set of all nodes `node` depends on inside this graph,
        excluding `node` itself."""
        nodes_to_run = set[Node]()
        nodes_to_check = deque[Node]([node])
        while nodes_to_check:
            n = nodes_to_check.popleft()
            if n in nodes_to_run:
                continue
            nodes_to_run.add(n)
            for edge in self.incoming_edges[n]:
                if not edge.connects_to_outside:
                    nodes_to_check.append(edge.from_port.node)
        # Remove the last node, since we want to return all nodes
        # needed to reach it
        nodes_to_run.remove(node)
        return nodes_to_run

    def run_to(
        self, last_node: Node, mode: EvaluationPhase = EvaluationPhase.ALWAYS
    ) -> dict[Port, Edge]:
        """Run only the nodes `last_node` depends on, in topological order.

        `last_node` itself is not run.

        Args:
            last_node: The node to run the graph up to.
            mode: The evaluation phase every run node is set to.

        Returns:
            dict[Port, Edge]: The incoming edges of `last_node`, keyed by its
                input ports. Empty if `last_node` is not among the sorted
                nodes.
        """
        nodes_to_run = self._build_path_to_node(last_node)
        for node, edges in zip(self.sorted_nodes, self.sorted_incoming_edges):
            if node == last_node:
                return edges
            if node not in nodes_to_run:
                continue
            node.set_mode(mode)
            for in_port in node.input_ports:
                if in_port in edges:
                    in_port.set_value(edges[in_port].from_port.value)
                else:
                    in_port.clear_value()
            node.run()
        return {}

    def _nodes_needed_for(self, target_nodes: set[Node]) -> set[Node]:
        """All nodes needed to produce a value for `target_nodes`,
        including the target nodes themselves."""
        nodes_to_run = set(target_nodes)
        for node in target_nodes:
            nodes_to_run |= self._build_path_to_node(node)
        return nodes_to_run

    def _samplers_on_path_to(self, ports: tuple[Port, ...]) -> list:
        """All PointSamplers needed to produce a value for any of `ports`,
        including their owning nodes."""
        from qewton.data.dataloaders.sampler.point_sampler import PointSampler

        nodes_to_run = self._nodes_needed_for({p.node for p in ports})
        return [n for n in nodes_to_run if isinstance(n, PointSampler)]

    def _run_nodes_needed_for(
        self, ports: tuple[Port, ...], mode: EvaluationPhase
    ) -> None:
        """Run every node needed to produce a value for each of `ports` in
        topological order, including the owning nodes of `ports`. Requires
        a sorted graph."""
        nodes_to_run = self._nodes_needed_for({p.node for p in ports})
        for node, edges in zip(self.sorted_nodes, self.sorted_incoming_edges):
            if node not in nodes_to_run:
                continue
            node.set_mode(mode)
            for in_port in node.input_ports:
                if in_port in edges:
                    in_port.set_value(edges[in_port].from_port.value)
                else:
                    in_port.clear_value()
            node.run()

    def diagram(self,
                depth: int = 1,
                show_port_names: bool = False,
                show_edge_labels: bool = True) -> GraphPlot:
        """Build a diagram of this graph's structure.

        Args:
            depth: How many levels of composite nodes (subclasses of
                :class:`~qewton.graphs.control_nodes.graph_node.GraphNode`) are
                expanded inline instead of drawn as a single box.
            show_port_names: Whether nodes with more than one input or output
                port also get a name label at each port.
            show_edge_labels: Whether edges are labeled with their data
                configuration.

        Returns:
            GraphPlot: The node-link diagram of this graph.
        """
        from qewton.visualization.plots.graph import GraphPlot

        return GraphPlot(self,
                         depth=depth,
                         show_port_names=show_port_names,
                         show_edge_labels=show_edge_labels)

    def visualize(
        self,
        port: Port | list[Port],
        reference: "Port | Callable | Any" = None,
        error: str | None = "signed",
        plot_type: type["Plot"] | None = None,
        max_vertex_distance: float | dict[Variable, float] = 0.05,
        device: Device | str | None = None,
        share_scale: bool = True,
        scale: "Scale | None" = None,
        controls=None,
        variables: list[Variable] | None = None,
        prediction_config: DataConfiguration | None = None,
        reference_config: DataConfiguration | None = None,
        mode: EvaluationPhase = EvaluationPhase.VALIDATION,
        **plot_kwargs,
    ) -> "Layout":
        """Evaluate the graph for `port` and build plots of the result.

        Only the nodes needed to produce a value for `port` are run. The
        plots are built with :func:`~qewton.visualization.auto.auto_plot`.

        Without `reference`, every
        :class:`~qewton.data.dataloaders.sampler.point_sampler.PointSampler`
        on the path to `port` is switched to mesh mode for this run (see
        :func:`~qewton.data.dataloaders.sampler.point_sampler.discretization_mode`),
        so the result is drawn on a deterministic discretization of its
        geometry.

        With `reference`, the result is compared to reference data in a
        Reference / Prediction / Error layout. The model is evaluated at
        the points of the reference, so no interpolation is needed.

        Args:
            port: One port, or a list of ports, to visualize.
            reference: Reference data for a single `port`. One of

                - plain data (a numpy array or a backend tensor on any
                  device); requires `reference_config`. Its Variable must
                  be the exact same instance as the one of `port`'s value.
                  The model is evaluated at the points of its geometry; a
                  list of GeometryAxes, one per factor of a product domain,
                  is evaluated at all combinations of the factors' points.
                - another port of this graph (e.g. the true-output port of
                  a data loader), run together with `port`. Its Variable
                  must equal the one of `port` by name, dimension and
                  structure.
                - a callable, evaluated at the mesh-mode points of `port`.

                None (default) draws `port` alone.
            error: With `reference`, which error panel to add: ``"signed"``
                (prediction - reference, diverging colormap, symmetric scale),
                ``"absolute"`` (absolute difference, scale from 0),
                ``"relative"`` (difference divided by the reference, NaN where
                the reference is close to 0, diverging, symmetric), or None for
                no error panel. Ignored without `reference`.
            plot_type: Plot type passed to
                :func:`~qewton.visualization.auto.auto_plot` for every plot.
                None (default) selects it automatically.
            max_vertex_distance: Maximal vertex distance of the mesh used in
                mesh mode. A dict sets it per geometry Variable, e.g. per
                factor of a space-time domain; geometries missing from it use
                None (the coarsest mesh). Unused with plain reference data.
            device: If given, every node needed for `port` is moved to this
                device before running, and mesh-mode points are created there.
            share_scale: With `reference`, whether the reference and prediction
                panels share one color scale. The error panel always has its
                own scale.
            scale: A fixed color scale for every slider state and animation
                frame. With `reference`, it is shared by the reference and
                prediction panels instead of the scale from `share_scale`; the
                error panel keeps its own scale. None (default) leaves the
                range to the plot.
            controls (ControlSpec class or instance, dict, list, or None):
                Controls for the axes not used by a plot role. A class or
                instance applies to all such axes, a dict maps axes to a class
                or instance, a list of ControlSpec instances is used as-is.
                None (default) uses sliders. With `reference`, each control
                instance is shared by all panels. On a product geometry, a
                control may be bound to one of its 1-dimensional factor
                Variables to step through that factor.
            variables: Narrows `port` (and `reference`, unless it already has
                exactly this combination) to the slice of these Variables of a
                composed FeatureAxes before plotting. With more than one
                Variable, all panels share one dropdown to switch between them,
                which requires them to have the same dimension. None (default)
                uses the Variable of `port` as declared.
            prediction_config: Used instead of the data configuration of `port`
                in this graph to build the prediction plot and to check
                `reference` against.
            reference_config: Used instead of the data configuration of a port
                `reference` in this graph. Required for plain reference data.
            mode: The evaluation phase every run node is set to; selects which
                split a data loader provides.
            **plot_kwargs: Passed to
                :func:`~qewton.visualization.auto.auto_plot` for every plot.

        Returns:
            Layout: Without `reference`, an ``Overlay`` of the plot for a
                single port, or a ``Row`` of plots for a list of ports. With
                `reference`, an ``Overlay`` of the reference, prediction and
                error plots for curve-like plots, or a ``Row`` of them
                otherwise. Show it with ``Figure(layout).show()``.

        Raises:
            ValueError: If `reference` is given for a list of ports, plain
                reference data is given without `reference_config`, the
                Variables or shapes of prediction and reference do not match,
                `error` is unknown, `scale` is given for plots without color,
                or `controls` names an axis that takes no control.
        """
        from qewton.visualization.layout import Overlay, Row

        ports = port if isinstance(port, list) else [port]

        if reference is not None:
            if len(ports) != 1:
                raise ValueError(
                    "reference is only supported for a single port, not a list."
                )
            if (
                not isinstance(reference, Port)
                and not callable(reference)
                and reference_config is None
            ):
                raise ValueError(
                    "reference_config is required when reference is plain "
                    "data, not a Port or a callable."
                )
            return self._visualize_with_reference(
                ports[0],
                reference,
                error,
                plot_type,
                max_vertex_distance,
                device,
                share_scale,
                scale,
                controls,
                variables,
                prediction_config,
                reference_config,
                mode,
                **plot_kwargs,
            )

        from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
        from qewton.visualization.auto import auto_plot
        from qewton.visualization.plots.spec import SelectorSpec

        nodes_to_run = self._nodes_needed_for({p.node for p in ports})
        if device is not None:
            for node in nodes_to_run:
                node.to(device)

        samplers = self._samplers_on_path_to(ports)
        with discretization_mode(samplers, max_vertex_distance, device):
            self._run_nodes_needed_for(ports, mode=mode)

        # Move right before plotting, not before running - the sampler(s)
        # and model can stay on the set device
        for sampler in samplers:
            for sampled_geometry in sampler.sampled_geometries:
                sampled_geometry.to_numpy()

        combined_variable = Variable.compose(variables) if variables else None
        controls_kwarg = {} if controls is None else {"controls": controls}
        plots = []
        for p in ports:
            data = p.node.backend.to_numpy(p.value)
            config = p.get_data_configuration(self)  # type: ignore
            if combined_variable is not None:
                data, config = self._narrow_to_variable(data, config, combined_variable)
            scale_kwarg = {} if scale is None else {"scale": scale}
            plots.append(
                auto_plot(
                    data,
                    config,
                    plot_type,
                    **controls_kwarg,
                    **scale_kwarg,
                    **plot_kwargs,
                )
            )

        if variables and len(variables) > 1:
            shared_selector_spec = SelectorSpec(variables)
            for plot in plots:
                self._redirect_to_shared_variable(plot, variables, shared_selector_spec)

        return Overlay(plots[0]) if len(plots) == 1 else Row(*plots)

    @staticmethod
    def _product_reference_geometry(geometry_axes: list, samplers) -> "Geometry":
        """The points to evaluate the model at for reference data given with
        one GeometryAxes per factor of a product domain: all combinations of
        the factors' points, with the coordinates in the order of the
        samplers' own domain variable - so the reference may list its
        factors in any order, e.g. angle first for a sampler on space x
        angle."""
        from qewton.geometries.discrete.product_mesh_geometry import (
            ProductDiscreteGeometry,
        )

        factors = [axes.geometry for axes in geometry_axes]
        variable = samplers[0].geometry.variable if samplers else None
        try:
            return ProductDiscreteGeometry(factors, variable=variable)
        except ValueError as e:
            names = [f.variable.name for f in factors]
            domain = variable.name if variable is not None else "the sampler's domain"
            raise ValueError(
                f"reference_config has one GeometryAxes per factor {names}, but "
                f"they do not make up {domain}: {e} Build each factor from the "
                "same Variable instance as the sampler's domain."
            ) from e

    @staticmethod
    def _reference_to_numpy(reference, port: Port) -> np.ndarray:
        """Reference values as a numpy array - also from a backend tensor on
        any device (e.g. a CUDA tensor), which np.asarray can't convert."""
        if not isinstance(reference, np.ndarray):
            try:
                reference = port.node.backend.to_numpy(reference)
            except (AttributeError, TypeError):
                pass
        return np.asarray(reference)

    def _visualize_with_reference(
        self,
        port: Port,
        reference: "Port | Callable | Any",
        error: str | None,
        plot_type,
        max_vertex_distance: float | dict[Variable, float],
        device,
        share_scale: bool,
        scale,
        controls,
        variables: list[Variable] | None,
        prediction_config: DataConfiguration | None,
        reference_config: DataConfiguration | None,
        mode: EvaluationPhase,
        **plot_kwargs,
    ) -> "Layout":
        """The `reference=` case of visualize(): builds a Reference/
        Prediction[/Error] comparison instead of a single plot.

        A callable reference is evaluated at the model's own points (mesh
        mode, like the no-reference path). A Port reference is evaluated
        together with `port` in one run. Plain reference data is evaluated
        against by substituting its own geometry for the model's, so no
        interpolation is needed.
        """
        from qewton.data.dataloaders.sampler.point_sampler import (
            active_discretization_mode,
            discretization_mode,
        )
        from qewton.visualization.auto import auto_plot, is_curve_like
        from qewton.visualization.layout import Overlay, Row
        from qewton.visualization.plots.spec import ColorSpec, Scale, SelectorSpec
        from qewton.visualization.themes import cmaps

        combined_variable = Variable.compose(variables) if variables else None
        pred_config = prediction_config or port.get_data_configuration(self)
        if pred_config is None:
            raise ValueError(
                f"{port} has no DataConfiguration - a callable reference "
                "needs the model's own evaluation points to call itself at."
            )
        pred_variable = (
            combined_variable
            if combined_variable is not None
            else (
                pred_config.feature_axes.variables if pred_config.feature_axes else None
            )
        )

        if callable(reference) and not isinstance(reference, Port):
            nodes_to_run = self._nodes_needed_for({port.node})
            if device is not None:
                for node in nodes_to_run:
                    node.to(device)

            samplers = self._samplers_on_path_to((port,))
            with discretization_mode(samplers, max_vertex_distance, device):
                self._run_nodes_needed_for((port,), mode=mode)

            for sampler in samplers:
                for sampled_geometry in sampler.sampled_geometries:
                    sampled_geometry.to_numpy()

            pred_data = port.node.backend.to_numpy(port.value)
            geometry_axes = pred_config.geometry_axes
            if geometry_axes is None:
                raise ValueError(
                    f"{port}'s own value has no GeometryAxes - a callable "
                    "reference needs the model's own evaluation points to "
                    "call itself at."
                )
            points = geometry_axes.geometry.discretization_points
            points = (
                points
                if isinstance(points, np.ndarray)
                else np.asarray(geometry_axes.geometry.backend.to_numpy(points))
            )
            ref_data = self._reference_to_numpy(reference(points), port)
            ref_config = pred_config
        elif isinstance(reference, Port):
            ref_port = reference
            ref_config = reference_config or ref_port.get_data_configuration(self)
            ref_variable = (
                ref_config.feature_axes.variables if ref_config.feature_axes else None
            )
            self._check_reference_variable(
                port, pred_variable, ref_variable, by_identity=False
            )

            nodes_to_run = self._nodes_needed_for({port.node, ref_port.node})
            if device is not None:
                for node in nodes_to_run:
                    node.to(device)

            self._run_nodes_needed_for((port, ref_port), mode=mode)

            pred_data = port.node.backend.to_numpy(port.value)
            pred_config = prediction_config or port.get_data_configuration(self)
            ref_data = ref_port.node.backend.to_numpy(ref_port.value)
            ref_config = reference_config or ref_port.get_data_configuration(self)
        else:
            assert reference_config is not None
            ref_config = reference_config
            ref_geometry_axes = ref_config.geometry_axes
            if ref_geometry_axes is None:
                raise ValueError("reference_config has no GeometryAxes.")
            samplers = self._samplers_on_path_to((port,))
            if isinstance(ref_geometry_axes, list):
                # One GeometryAxes per factor of a product domain, e.g. the
                # parameter values and a grid of space points.
                ref_geometry = self._product_reference_geometry(
                    ref_geometry_axes, samplers
                )
            else:
                ref_geometry = ref_geometry_axes.geometry

            ref_variable = (
                ref_config.feature_axes.variables if ref_config.feature_axes else None
            )
            self._check_reference_variable(port, pred_variable, ref_variable)

            nodes_to_run = self._nodes_needed_for({port.node})
            if device is not None:
                for node in nodes_to_run:
                    node.to(device)

            with active_discretization_mode(samplers, ref_geometry):
                self._run_nodes_needed_for((port,), mode=mode)

            for sampler in samplers:
                for sampled_geometry in sampler.sampled_geometries:
                    sampled_geometry.to_numpy()

            pred_data = port.node.backend.to_numpy(port.value)
            if combined_variable is not None:
                # Narrow using the port's own real config - pred_config is
                # about to be replaced with ref_config below, which would
                # otherwise already look narrowed (matching ref_variable)
                # while pred_data itself is still shaped for the full,
                # un-narrowed Variable.
                pred_data, _ = self._narrow_to_variable(
                    pred_data,
                    prediction_config or port.get_data_configuration(self),
                    combined_variable,
                )
            # The samplers evaluated ref_geometry's points as one flat batch -
            # give the prediction the reference's structure back.
            structure = tuple(ref_geometry.shape)
            if pred_data.ndim == 2 and pred_data.shape[0] == int(np.prod(structure)):
                pred_data = pred_data.reshape(*structure, pred_data.shape[-1])
            # auto_plot() dispatches on a DataConfiguration's structural
            # type (e.g. GridGeometry vs. a plain DiscreteGeometry) -
            # port.get_data_configuration(self) still reflects the
            # sampler's own geometry type even after active_discretization_
            # mode gave it ref_geometry's points, so prediction reuses
            # ref_config directly instead.
            pred_config = ref_config
            ref_data = self._reference_to_numpy(reference, port)

        if combined_variable is not None:
            pred_data, pred_config = self._narrow_to_variable(
                pred_data, pred_config, combined_variable
            )
            ref_data, ref_config = self._narrow_to_variable(
                ref_data, ref_config, combined_variable
            )

        if pred_data.shape != ref_data.shape:
            raise ValueError(
                f"prediction shape {pred_data.shape} does not match "
                f"reference shape {ref_data.shape} - both must be evaluated "
                "at the same points."
            )

        # A curve-like family uses "label" (an Overlay legend entry) to
        # distinguish panels; every other family uses "title" (a Row panel
        # heading) - probe once with auto_plot() to find out which. The probe
        # gets the same controls as the panels, since they decide e.g. which
        # factor of a space-time domain is drawn.
        probe_kwargs = dict(plot_kwargs)
        if controls is not None:
            probe_kwargs["controls"] = controls
        probe = auto_plot(ref_data, ref_config, plot_type, **probe_kwargs)
        is_curve = is_curve_like(probe)
        name_kwarg = "label" if is_curve else "title"

        def _named(name: str, shared_controls=None) -> dict:
            kwargs = dict(plot_kwargs, **{name_kwarg: name})
            if shared_controls is not None:
                kwargs["controls"] = shared_controls
            elif controls is not None:
                kwargs["controls"] = controls
            return kwargs

        reference_plot = auto_plot(ref_data, ref_config, plot_type, **_named("Reference"))
        # One resolved ControlSpec instance per surplus axis, shared across
        # every panel from here on.
        shared_controls = reference_plot.controls
        prediction_plot = auto_plot(
            pred_data, pred_config, plot_type, **_named("Prediction", shared_controls)
        )
        plots = [reference_plot, prediction_plot]

        shared_selector_spec = (
            SelectorSpec(variables) if variables and len(variables) > 1 else None
        )
        if shared_selector_spec is not None:
            for plot in plots:
                self._redirect_to_shared_variable(plot, variables, shared_selector_spec)

        ref_color = getattr(reference_plot, "color", None)
        pred_color = getattr(prediction_plot, "color", None)
        has_colors = isinstance(ref_color, ColorSpec) and isinstance(
            pred_color, ColorSpec
        )
        if scale is not None and not has_colors:
            raise ValueError(
                "scale= is not used, since the reference and prediction plots "
                "have no color."
            )
        if has_colors and (share_scale or scale is not None):
            shared_scale = scale if scale is not None else Scale()
            ref_color.scale = shared_scale
            pred_color.scale = shared_scale

        if error is not None:
            if error == "signed":
                error_data = pred_data - ref_data
                error_scale = Scale(symmetric=True)
                diverging = True
            elif error == "absolute":
                error_data = np.abs(pred_data - ref_data)
                error_scale = Scale(vmin=0.0)
                diverging = False
            elif error == "relative":
                diff = pred_data - ref_data
                threshold = 1e-8 * np.max(np.abs(ref_data))
                with np.errstate(divide="ignore", invalid="ignore"):
                    error_data = diff / ref_data
                error_data = np.where(np.abs(ref_data) < threshold, np.nan, error_data)
                error_scale = Scale(symmetric=True)
                diverging = True
            else:
                raise ValueError(
                    'error must be "signed", "absolute", "relative", or '
                    f"None, got {error!r}."
                )

            error_plot = auto_plot(
                error_data, ref_config, plot_type, **_named("Error", shared_controls)
            )
            if shared_selector_spec is not None:
                self._redirect_to_shared_variable(
                    error_plot, variables, shared_selector_spec
                )
            error_color = getattr(error_plot, "color", None)
            if isinstance(error_color, ColorSpec):
                error_color.scale = error_scale
                if error_color.cmap is None and diverging:
                    error_color.cmap = cmaps.diverging
            plots.append(error_plot)

        return Overlay(*plots) if is_curve else Row(*plots)

    @staticmethod
    def _narrow_to_variable(data, config: DataConfiguration, variable: Variable):
        """(data, config) narrowed to just `variable`'s own slice of a
        composed FeatureAxes - e.g. picking a model's velocity output out
        of a port declared as a combined velocity+pressure Variable. A
        no-op if `config` already names exactly `variable`."""
        current = config.feature_axes.variables if config.feature_axes else None
        if current == variable:
            return data, config
        slc = config.get_variable_slice(variable)
        narrowed_data = np.asarray(data)[slc]
        new_axes = tuple(
            FeatureAxes(variable) if isinstance(axes, FeatureAxes) else axes
            for axes in config.axes
        )
        return narrowed_data, DataConfiguration(*new_axes, dtype=config.dtype)

    @staticmethod
    def _check_reference_variable(
        port: Port, pred_variable, ref_variable, by_identity: bool = True
    ) -> None:
        """Checks that `reference`'s Variable matches `port`'s own.

        `by_identity=True` (the default, for plain reference data) requires
        the exact same Variable instance. `by_identity=False` (the Port-
        reference case) falls back to structural equality
        (Variable.__eq__: name, dim, and children)."""
        if by_identity:
            matches = ref_variable is pred_variable
            reason = "the exact same Variable instance as"
        else:
            matches = ref_variable is not None and ref_variable == pred_variable
            reason = "the same Variable (by name/dim/structure) as"
        if not matches:
            raise ValueError(
                f"reference's DataConfiguration must be built from {reason} "
                f"{port}'s own value ({pred_variable!r}) - found "
                f"{ref_variable!r} instead. Reuse the same Variable "
                "instance building both configs; never match by name."
            )

    @staticmethod
    def _redirect_to_shared_variable(
        plot: "Plot", variables: list[Variable], shared_spec
    ) -> None:
        """Points every PlotSpec attribute on `plot` that currently names
        one of `variables` (e.g. its `color`/`vector`/`y`) at `shared_spec`
        instead, so switching the shared SelectorSpec's state moves every
        panel together."""
        from qewton.visualization.plots.spec import PlotSpec

        for value in vars(plot).values():
            if isinstance(value, PlotSpec) and value.variable_or_axes in variables:
                value.variable_or_axes = shared_spec

    def collect_trainable_parameters(self):
        """Collect the trainable parameters of all nodes in this graph.

        Nodes in state ``NodeState.FIXED`` contribute no parameters.

        Returns:
            TrainableParametersCollection: The trainable parameters of all
                nodes.
        """
        params_collection = TrainableParametersCollection()
        for node in self.nodes:
            p = node._trainable_parameters  # pylint: disable=W0212
            if not p.empty:
                params_collection.extend(p)
        return params_collection

    @contextmanager
    def tracker(self, n_tracking_vars=1):
        """Context manager that builds this graph from node calls.

        While active, calling a node with :class:`TrackingObject` arguments
        does not compute anything but connects the node in this graph and
        returns tracking objects for its outputs. Non-tracking arguments
        become defaults of the corresponding input ports. Trackers can be
        nested; the innermost graph receives the connections.

        Args:
            n_tracking_vars (int, optional): The number of tracking objects to
                yield. Default is 1.

        Yields:
            TrackingObject or list[TrackingObject] or None: A single tracking
                object if `n_tracking_vars` is 1, a list of them if it is
                larger, None if it is 0.

        Raises:
            RuntimeError: If the graph already contains nodes.
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
    """Graph of nodes connected in a linear sequence.

    Each node is connected to the next one with :meth:`Graph.connect`, so
    the output ports of a node must match the input ports of its
    successor. The given order is used as execution order.

    Args:
        *nodes: The nodes in execution order.
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
    """Placeholder for a value flowing through a graph during tracking.

    Inside :meth:`Graph.tracker`, calling a node with tracking objects
    connects the node in the tracked graph and returns new tracking objects
    for its output ports. Arithmetic operators (``+``, ``-``, ``*``,
    ``/``, ``**``, ``@``, unary ``-``, ``abs``), indexing, item assignment
    and the methods of this class add the corresponding nodes from
    :mod:`qewton.algorithms.building_blocks` in the same way and return
    the tracking object of the result.

    Args:
        last_output_port: The output port that produces this value. None for a
            graph input that is not produced by any node.

    Attributes:
        current_graph_tracked (Graph or None): Class attribute: the graph
            currently being tracked, None outside of :meth:`Graph.tracker`.
        last_output_port (OutputPort or None): The output port that produces
            this value.
        to_ports (list[InputPort]): For a graph input, the input ports it was
            passed to.
    """

    current_graph_tracked: Graph | None = None

    def __init__(self, last_output_port: OutputPort | None = None):
        self.last_output_port: OutputPort | None = last_output_port
        self.to_ports = []

    def add_to_port(self, port: InputPort):
        """Record that this graph input is passed to `port`."""
        self.to_ports.append(port)

    def __add__(self, other):
        """Track ``self + other``."""
        from qewton.algorithms.building_blocks.math import Add

        add_node = Add()
        return add_node(self, other)

    def __radd__(self, other):
        """Track ``other + self``."""
        from qewton.algorithms.building_blocks.math import Add

        add_node = Add()
        return add_node(other, self)

    def __matmul__(self, other):
        """Track the matrix product ``self @ other``."""
        from qewton.algorithms.building_blocks.math import MatMul

        matmul_node = MatMul()
        return matmul_node(self, other)

    def __sub__(self, other):
        """Track ``self - other``."""
        from qewton.algorithms.building_blocks.math import Subtract

        subtract_node = Subtract()
        return subtract_node(self, other)

    def __neg__(self):
        """Track the negation ``-self``."""
        from qewton.algorithms.building_blocks.math import Negative

        neg_node = Negative()
        return neg_node(self)

    def __rsub__(self, other):
        """Track ``other - self``."""
        from qewton.algorithms.building_blocks.math import Subtract

        subtract_node = Subtract()
        return subtract_node(other, self)

    def __mul__(self, other):
        """Track ``self * other``."""
        from qewton.algorithms.building_blocks.math import Multiply

        multiply_node = Multiply()
        return multiply_node(self, other)

    def __rmul__(self, other):
        """Track ``other * self``."""
        from qewton.algorithms.building_blocks.math import Multiply

        multiply_node = Multiply()
        return multiply_node(other, self)

    def __pow__(self, other):
        """Track ``self ** other``."""
        from qewton.algorithms.building_blocks.math import Power

        power_node = Power()
        return power_node(self, other)

    def __rpow__(self, other):
        """Track ``other ** self``."""
        from qewton.algorithms.building_blocks.math import Power

        power_node = Power()
        return power_node(other, self)

    def __truediv__(self, other):
        """Track ``self / other``."""
        from qewton.algorithms.building_blocks.math import Divide

        divide_node = Divide()
        return divide_node(self, other)

    def __rtruediv__(self, other):
        """Track ``other / self``."""
        from qewton.algorithms.building_blocks.math import Divide

        divide_node = Divide()
        return divide_node(other, self)

    def __abs__(self):
        """Track the element-wise absolute value ``abs(self)``."""
        from qewton.algorithms.building_blocks.math import Abs

        abs_node = Abs()
        return abs_node(self)

    def norm(self, ord: float | str = 2):
        """Track the norm along the last axis.

        Args:
            ord: The order of the norm.
        """
        from qewton.algorithms.building_blocks.math import Norm

        norm_node = Norm(ord=ord)
        return norm_node(self)

    def sqrt(self):
        """Track the element-wise square root."""
        from qewton.algorithms.building_blocks.math import Sqrt

        sqrt_node = Sqrt()
        return sqrt_node(self)

    def __getitem__(self, key):
        """Track the slice ``self[key]``."""
        from qewton.algorithms.building_blocks.array_operations import Slice

        slice_node = Slice(key)
        return slice_node(self)

    def __setitem__(self, key, value):
        """Track the item assignment ``self[key] = value``."""
        from qewton.algorithms.building_blocks.array_operations import SetItem

        set_item_node = SetItem()
        return set_item_node(self, key, value)

    def dot(self, other):
        """Track the dot product with `other`.

        Args:
            other (TrackingObject): The second factor.
        """
        from qewton.algorithms.building_blocks.math import Dot

        dot_node = Dot()
        return dot_node(self, other)

    def squeeze(self, dim):
        """Track removing the dimension `dim` of size 1.

        Args:
            dim (int): The dimension to remove.
        """
        from qewton.algorithms.building_blocks.array_operations import Squeeze

        squeeze_node = Squeeze(dim)
        return squeeze_node(self)

    def unsqueeze(self, dim):
        """Track adding a dimension of size 1 at position `dim`.

        Args:
            dim (int): The position of the new dimension.
        """
        from qewton.algorithms.building_blocks.array_operations import Unsqueeze

        unsqueeze_node = Unsqueeze(dim)
        return unsqueeze_node(self)

    def track_gradient(self):
        """Track enabling gradient computation with respect to this value."""
        from qewton.algorithms.building_blocks.derivatives import GradientTracking

        track_node = GradientTracking()
        return track_node(self)

    def gradient(self, with_respect_to):
        """Track the gradient of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import Gradient

        grad_node = Gradient()
        return grad_node(self, with_respect_to)

    def normal_derivative(self, normals, with_respect_to):
        """Track the normal derivative of this value.

        Args:
            normals (TrackingObject): The normal vectors.
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import NormalDerivative

        norm_der_node = NormalDerivative()
        return norm_der_node(self, normals, with_respect_to)

    def laplacian(self, with_respect_to):
        """Track the Laplacian of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import Laplacian

        lap_node = Laplacian()
        return lap_node(self, with_respect_to)

    def jac(self, with_respect_to):
        """Track the Jacobian of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import Jacobian

        jac_node = Jacobian()
        return jac_node(self, with_respect_to)

    def div(self, with_respect_to):
        """Track the divergence of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import Divergence

        div_node = Divergence()
        return div_node(self, with_respect_to)

    def hessian(self, with_respect_to):
        """Track the Hessian of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import Hessian

        hes_node = Hessian()
        return hes_node(self, with_respect_to)

    def sym_grad(self, with_respect_to):
        """Track the symmetric gradient of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import SymmetricGradient

        sym_grad_node = SymmetricGradient()
        return sym_grad_node(self, with_respect_to)

    def rot(self, with_respect_to):
        """Track the rotation (curl) of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import Rotation

        rot_node = Rotation()
        return rot_node(self, with_respect_to)

    def matrix_div(self, with_respect_to):
        """Track the row-wise divergence of this value.

        Args:
            with_respect_to (TrackingObject): The input to differentiate with
                respect to.
        """
        from qewton.algorithms.building_blocks.derivatives import MatrixDivergence

        matrix_div_node = MatrixDivergence()
        return matrix_div_node(self, with_respect_to)


# endregion
