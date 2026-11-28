"""Computation graphs (DAGs) of nodes and the tracking objects that build
them from plain Python functions."""

from __future__ import annotations
from collections import deque
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, TYPE_CHECKING
from warnings import warn
import inspect
import numpy as np

from qewton.config.axes import FeatureAxes
from qewton.config.data_configurations import DataConfiguration, refining
from qewton.config.devices import Device
from qewton.config.errors import DataConfigMismatchError, StructureUnknown
from qewton.config.variables import Variable
from qewton.config.saving.saving import Serializable
from qewton.graphs.nodes import (
    InputPort,
    Node,
    EvaluationPhase,
    NodeState,
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
    from qewton.visualization.axes import ControlAxis, Selector
    from qewton.visualization.layout import Layout
    from qewton.visualization.scales import ColorScale


class Graph(Serializable):
    """Directed acyclic graph (DAG) of interconnected nodes.

    Nodes are connected port to port via :class:`~qewton.graphs.edges.Edge`
    objects. Every connection merges the data configurations of both ports
    (see :meth:`~qewton.config.data_configurations.DataConfiguration.merge`):
    axes and dims that correspond become one object. A graph keeps the
    matched configuration and the value of every port it contains, also of
    the ports inside its GraphNodes, per *place*: a port together with the
    path of GraphNodes leading to it, so one node may occur at several places
    (see :meth:`config_of` and :meth:`value_of`). The static configurations
    of the ports stay untouched.

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

        # Per place of a node, (path of GraphNodes, node): the configurations
        # of its ports. The places whose `relate` waits for more structure.
        # Per place of a port, (path, port): its value in the last run. The
        # evaluation steps in order; None until planned.
        self._configs: dict[tuple, dict[Port, DataConfiguration]] = {}
        self._waiting: list[tuple] = []
        self._values: dict[tuple, Any] = {}
        self._steps: list | None = None

    #: Rebuilt from the structure when needed, never saved.
    _TRANSIENT = ("_configs", "_waiting", "_values", "_steps")

    def _collect_serializable_attributes(self) -> tuple[list, list]:
        keys = [k for k in self.__dict__ if k not in self._TRANSIENT]
        return keys, [self.__dict__[k] for k in keys]

    def load(self, serializer, data_config: dict) -> None:
        super().load(serializer, data_config)
        self._configs, self._waiting, self._values, self._steps = {}, [], {}, None

    def _invalidate(self) -> None:
        self._steps = None

    # region: Places

    def _structure(self, path: tuple) -> Graph:
        """The graph whose nodes are at `path`: this graph, or the inner
        graph of the last GraphNode of `path`."""
        return path[-1].inner_graph if path else self

    def _paths_to(self, node: Node, path: tuple = ()) -> list[tuple]:
        """All paths of GraphNodes leading to `node` in this graph."""
        from qewton.graphs.control_nodes.graph_node import GraphNode

        structure = self._structure(path)
        found = [path] if node in structure.nodes else []
        for inner in structure.nodes:
            if isinstance(inner, GraphNode):
                found += self._paths_to(node, path + (inner,))
        return found

    def _path_of(self, node: Node, within) -> tuple:
        """The path leading to `node`: `within`, or the only one there is."""
        if within is not None:
            return tuple(within)
        if node in self.nodes or ((), node) in self._configs:
            return ()
        paths = self._paths_to(node)
        if len(paths) == 1:
            return paths[0]
        if not paths:
            raise ValueError(f"{node.name} is not part of this graph.")
        options = ", ".join("(" + ", ".join(n.name for n in p) + ")" for p in paths)
        raise ValueError(
            f"{node.name} occurs at several places of this graph; pass `within` "
            f"with the GraphNodes leading to one of them: {options}."
        )

    def _source(self, path: tuple, port: Port) -> tuple:
        """The place whose value the port at (path, port) carries: following
        incoming edges and the boundaries of GraphNodes, to the output that
        computes it, or to an unconnected input (its default)."""
        from qewton.graphs.control_nodes.graph_node import GraphNode

        structure = self._structure(path)
        node = port.node
        if isinstance(port, InputPort):
            for edge in structure.incoming_edges.get(node, []):
                if edge.to_port is port:
                    return self._source(path, edge.from_port)
            if path and node is path[-1]:
                return self._source(path[:-1], port)
            return (path, port)
        if node in structure.nodes and isinstance(node, GraphNode):
            return self._source(path + (node,), port)
        for edge in structure.edges_to_outside + structure.skip_connections:
            if edge.to_port is port:
                return self._source(path, edge.from_port)
        return (path, port)

    def config_of(self, port: Port, within=None) -> DataConfiguration | None:
        """The matched data configuration of `port` in this graph.

        Args:
            port: A port of a node in this graph or in the inner graph of
                one of its GraphNodes, at any depth.
            within (tuple[GraphNode, ...], optional): The GraphNodes, from
                the outside in, leading to the node of `port`; needed only if
                the node occurs at several places of this graph.

        Returns:
            DataConfiguration or None: The configuration, or None if the node
                is not part of this graph.
        """
        try:
            path = self._path_of(port.node, within)
        except ValueError as error:
            if "several places" in str(error):
                raise
            return None
        if (path, port.node) not in self._configs:
            self._ensure_plan()
            if (path, port.node) not in self._configs:
                return None
        return self._config(path, port).resolve()

    def value_of(self, port: Port, within=None) -> Any:
        """The value of `port` in the last run of this graph.

        Args:
            port: A port of a node in this graph or in the inner graph of
                one of its GraphNodes, at any depth.
            within (tuple[GraphNode, ...], optional): The GraphNodes, from
                the outside in, leading to the node of `port`; needed only if
                the node occurs at several places of this graph, e.g. a node
                whose weights are shared by two copies (:meth:`Node.copy`).

        Returns:
            Any: The value; None if it was not computed in the last run.

        Raises:
            ValueError: If the node of `port` is not part of this graph, or
                occurs at several places and `within` is not given.
        """
        place = self._source(self._path_of(port.node, within), port)
        if place in self._values:
            return self._values[place]
        return getattr(place[1], "default", None)

    # endregion
    # region: Matching

    def _instantiate(self, path: tuple, node: Node, inside: bool = True) -> None:
        """Give `node` its configurations at `path`: a copy of its declared
        ones, matched inside a built GraphNode with its inner graph, related
        by its `relate`. With `inside` False, only the copy (a node outside of this
        graph that an edge leads to)."""
        from qewton.graphs.control_nodes.graph_node import GraphNode

        if (path, node) in self._configs:
            return
        configs = node.copy_data_configs()
        for port, config in configs.items():
            config.mark_origin(port)
        self._configs[(path, node)] = configs
        if not inside:
            return
        # The inner graph of an unbuilt GraphNode is not valid yet; until its
        # setup builds it, only its declared ports count.
        if isinstance(node, GraphNode) and node.state != NodeState.UNINITIALIZED:
            inner_path = path + (node,)
            for inner in node.inner_graph.nodes:
                self._instantiate(inner_path, inner)
            for edge in node.inner_graph.all_edges():
                self._merge_edge(inner_path, edge)
        self._relate(path, node, refine=False)

    def _config(self, path: tuple, port: Port) -> DataConfiguration:
        """The configuration of `port` at `path`; a port of the GraphNode
        whose inner graph is at `path` is the one of its own place."""
        node = port.node
        if path and node is path[-1]:
            return self._config(path[:-1], port)
        if (path, node) not in self._configs:
            self._instantiate(path, node, inside=False)
        configs = self._configs[(path, node)]
        if port not in configs:
            configs[port] = node.copy_data_config_of_port(port, {})
            configs[port].mark_origin(port)
        return configs[port]

    def _merge_edge(self, path: tuple, edge: Edge) -> None:
        """Merge the configurations of both ends of `edge` at `path`."""
        try:
            self._config(path, edge.from_port).merge(self._config(path, edge.to_port))
        except DataConfigMismatchError as error:
            raise DataConfigMismatchError(
                f"Connection of {edge.from_port.node.name} to "
                f"{edge.to_port.node.name} failed: {error}"
            ) from error

    def _relate(self, path: tuple, node: Node, refine: bool) -> bool:
        """Merge the outputs of `node` at `path` with what its `relate`
        returns; False if it waits for more structure."""
        configs = self._configs[(path, node)]
        try:
            with refining(refine):
                related = node.relate(*(configs[port] for port in node.input_ports))
        except StructureUnknown:
            if (path, node) not in self._waiting:
                self._waiting.append((path, node))
            return False
        if (path, node) in self._waiting:
            self._waiting.remove((path, node))
        if related is None:
            return True
        outputs = related if isinstance(related, tuple) else (related,)
        for port, config in zip(node.output_ports, outputs):
            try:
                configs[port].merge(config)
            except DataConfigMismatchError as error:
                raise DataConfigMismatchError(
                    f"The output {port.name} of {node.name} contradicts its "
                    f"declaration: {error}"
                ) from error
        return True

    def _relate_waiting(self, refine: bool = False) -> None:
        """Run the waiting `relate`s again until none makes progress."""
        progress = True
        while progress:
            progress = False
            for path, node in list(self._waiting):
                progress |= self._relate(path, node, refine)

    def _merge_defaults(self) -> None:
        """Merge every unconnected input that has a default with the
        configuration of the default."""
        for (path, node), configs in list(self._configs.items()):
            structure = self._structure(path)
            if node not in structure.nodes:
                continue
            connected = {edge.to_port for edge in structure.incoming_edges[node]}
            for port in node.input_ports:
                if port in connected or port.is_required:
                    continue
                default = DataConfiguration.of_value(port.default)
                if default is not None:
                    configs[port].merge(default)

    @contextmanager
    def _unchanged_on_mismatch(self):
        """Leave the nodes, edges and configurations as they were if the
        body raises a DataConfigMismatchError. Merging can not be undone, so
        the configurations are matched again from the declarations."""
        nodes = set(self.nodes)
        incoming = {node: list(edges) for node, edges in self.incoming_edges.items()}
        outgoing = {node: list(edges) for node, edges in self.outgoing_edges.items()}
        from_outside = list(self.edges_from_outside)
        to_outside = list(self.edges_to_outside)
        skip_connections = list(self.skip_connections)
        try:
            yield
        except DataConfigMismatchError:
            self.nodes.clear()
            self.nodes.update(nodes)
            self.incoming_edges, self.outgoing_edges = incoming, outgoing
            self.edges_from_outside = from_outside
            self.edges_to_outside = to_outside
            self.skip_connections = skip_connections
            self._rematch()
            raise

    def _rematch(self) -> None:
        """Match again after a part of the graph is gone (an edge or node
        removed or replaced, a connection that failed). Sizes inferred from
        the connections may have lost their source, so the inferred
        HyperParameters of the nodes not built yet are opened and inferred
        anew from the remaining connections; a built node keeps them, since
        its weights depend on them."""
        for node in self.nodes:
            if node.state == NodeState.UNINITIALIZED:
                for parameter in node.hyperparameters:
                    parameter.reopen()
        self._rebuild(refine=False)

    def _rebuild(self, refine: bool) -> None:
        """Match all configurations from the declarations. With `refine`,
        as at the end of a setup, also defaults of unconnected inputs, and
        `relate`s still waiting refine their unknown inputs."""
        self._configs, self._waiting = {}, []
        for node in self.nodes:
            self._instantiate((), node)
        for edge in self.all_edges():
            self._merge_edge((), edge)
        self._relate_waiting()
        if refine:
            self._merge_defaults()
            self._relate_waiting()
            self._relate_waiting(refine=True)

    def all_edges(self) -> list[Edge]:
        """Every edge of this graph, also those from and to outside and the
        skip connections, once each."""
        edges = []
        for edge in [e for es in self.incoming_edges.values() for e in es] + (
            self.edges_to_outside + self.skip_connections
        ):
            if edge not in edges:
                edges.append(edge)
        return edges

    # endregion

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
        self._invalidate()
        if not node in self.incoming_edges:
            self.incoming_edges[node] = list[Edge]()
            self.outgoing_edges[node] = list[Edge]()
        self._instantiate((), node)

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
                connected ports are incompatible. The graph is left as it
                was: no node or edge is added.
        """
        self._check_graph_was_sorted()
        self._invalidate()

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

        with self._unchanged_on_mismatch():
            # Nodes must be added to the graph
            self.add_node(from_node, check_warning=False)
            self.add_node(to_node, check_warning=False)

            # Configurations are merged
            for from_port, to_port in zip(from_ports, to_ports):
                edge = Edge(from_port, to_port)
                self._merge_edge((), edge)
                self.incoming_edges[to_node].append(edge)
                self.outgoing_edges[from_node].append(edge)
            self._relate_waiting()

    def remove(self, obj: Node | Edge):
        """Remove a node or edge from the graph.

        Removing a node also removes all edges connected to it. All data
        configurations are matched again afterwards and the graph is marked
        as unsorted.

        Args:
            obj: The node or edge to remove.
        """
        if isinstance(obj, Node):
            if obj in self.nodes:
                self.nodes.remove(obj)
                self.incoming_edges.pop(obj, None)
                self.outgoing_edges.pop(obj, None)
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
        # Merging cannot be undone, so the configurations are matched again
        # from the declarations.
        self._rematch()
        self._invalidate()
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
        new_graph._rebuild(refine=False)
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
            edge.to_port = input_port_mapping[edge.to_port]
            self.incoming_edges[new_node].append(edge)
        self.incoming_edges.pop(old_node)

        # Transfer outgoing edges from old_node to new_node
        for edge in self.outgoing_edges[old_node]:
            edge.from_port = output_port_mapping[edge.from_port]
            self.outgoing_edges[new_node].append(edge)
        self.outgoing_edges.pop(old_node)

        # Transfer skip connections involving old_node to new_node
        for edge in list(self.skip_connections):
            if edge.from_port.node == old_node:
                edge.from_port = output_port_mapping[edge.from_port]
            if edge.to_port.node == old_node:
                edge.to_port = input_port_mapping[edge.to_port]

        # Remove the old node from the graph
        self.nodes.remove(old_node)

        # Match the configurations again after replacement
        self._rematch()
        self._invalidate()

        # Mark the graph as unsorted since the structure has changed
        self.graph_was_sorted = False

    def _remove_node_connections(self, edge_container: list[Edge], node: Node):
        """Remove all edges touching `node` from `edge_container` in place."""
        edge_list = list(edge_container)  # Create a copy
        for edge in edge_list:
            if edge.from_port.node == node or edge.to_port.node == node:
                edge_container.remove(edge)

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
                are incompatible. The graph is left as it was: no node or
                edge is added.
        """
        self._check_graph_was_sorted()
        to_node = to_port.node
        self._check_port_is_free(to_node, to_port)
        with self._unchanged_on_mismatch():
            self.add_node(to_node, check_warning=False)

            edge = Edge(from_port, to_port, connects_to_outside=True)
            self._merge_edge((), edge)
            self.edges_from_outside.append(edge)
            self.incoming_edges[to_node].append(edge)
            self._relate_waiting()

    def connect_to_outside_of_graph(self, from_port: OutputPort, to_port: Port):
        """Connect an output port inside this graph to a port outside of it.

        The node of `from_port` is added to this graph; `to_port` carries the
        value of `from_port` (see :meth:`value_of`).

        Args:
            from_port: The output port inside this graph that provides the
                data.
            to_port: The port outside of this graph that receives the data.

        Raises:
            DataConfigMismatchError: If the data configurations of the ports
                are incompatible. The graph is left as it was: no node or
                edge is added.
        """
        self._check_graph_was_sorted()
        from_node = from_port.node
        with self._unchanged_on_mismatch():
            self.add_node(from_node, check_warning=False)
            edge = Edge(from_port, to_port, connects_to_outside=True)
            self._merge_edge((), edge)
            self.edges_to_outside.append(edge)
            self.outgoing_edges[from_node].append(edge)
            self._relate_waiting()

    def add_skip_connection(self, from_port: Port, to_port: Port):
        """Add a skip connection between two ports.

        The edge is not part of the topological order; `to_port` carries
        the value of `from_port` (see :meth:`value_of`), and their data
        configurations are merged.

        Args:
            from_port: The port providing the data.
            to_port: The port receiving the data.

        Raises:
            DataConfigMismatchError: If the data configurations of the ports
                are incompatible. The graph is left as it was: no edge is
                added.
        """
        with self._unchanged_on_mismatch():
            edge = Edge(from_port, to_port, connects_to_outside=True)
            self._merge_edge((), edge)
            self.skip_connections.append(edge)
        self._invalidate()

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

    def setup(self, plan: bool = True):
        """Sort the graph, set up all nodes for evaluation, in order, and
        match their configurations.

        Every node receives this graph in its `setup` (see
        :meth:`~qewton.graphs.nodes.Node.setup`); a
        :class:`~qewton.graphs.nodes.GraphAwareNode` builds with it. A
        :class:`~qewton.graphs.control_nodes.data_processing_node.DataProcessingNode`
        is only set up if its data source node is part of this graph.

        Args:
            plan: Whether to match the configurations and plan the run now.
                False for the inner graph of a GraphNode: a graph containing
                the GraphNode matches it in place, and it plans itself when
                it is run on its own.
        """
        self.sort()
        if any(node.open_hyperparameters for node in self.nodes):
            # Sizes still open are inferred before the nodes are built.
            self._rebuild(refine=True)
        for node in self.sorted_nodes:
            if isinstance(node, DataProcessingNode) and node.data_source_node not in self.nodes:
                continue
            node.setup(self)
        # A plan made during the setups (e.g. by run_to) may hold inner graphs
        # that nodes have rebuilt since.
        if plan:
            self._plan()
        else:
            self._steps = None

    # region: Planning and running

    def _ensure_plan(self) -> None:
        if self._steps is None:
            self._plan()

    def _plan(self) -> None:
        """Match all configurations from the declarations and list the
        evaluation steps.

        The inner graph of a GraphNode is not run as a whole: its steps are
        inserted at the place of the GraphNode, in the order of the inner
        graph. Inputs read the place of the output that computes their
        value (see :meth:`_source`).
        """
        self._rebuild(refine=True)
        self._steps = []
        self._list_steps(())

    def _list_steps(self, path: tuple) -> None:
        from qewton.graphs.control_nodes.graph_node import GraphNode

        structure = self._structure(path)
        if not structure.graph_was_sorted:
            structure.sort()
        owner = path[0] if path else None
        for node in structure.sorted_nodes:
            if isinstance(node, GraphNode):
                self._list_steps(path + (node,))
                continue
            sources = tuple(self._source(path, port) for port in node.input_ports)
            outputs = tuple((path, port) for port in node.output_ports)
            self._steps.append((node, sources, outputs, owner or node))

    def _execute(self, mode, owners=None, on_node_done=None) -> None:
        """Evaluate the listed steps, only those of `owners` if given; call
        `on_node_done` with each owning node after its last step."""
        values = self._values
        steps = [step for step in self._steps if owners is None or step[3] in owners]
        last_step = {id(step[3]): index for index, step in enumerate(steps)}
        for index, (node, sources, outputs, owner) in enumerate(steps):
            node.set_mode(mode)
            result = node(
                *[
                    values[s] if s in values else getattr(s[1], "default", None)
                    for s in sources
                ]
            )
            if len(outputs) == 1:
                values[outputs[0]] = result
            elif outputs:
                values.update(zip(outputs, result))
            if on_node_done is not None and last_step[id(owner)] == index:
                on_node_done(owner)

    def evaluate(
        self,
        inputs: dict[Port, Any],
        outputs: list[Port],
        mode: EvaluationPhase = EvaluationPhase.ALWAYS,
    ) -> list[Any]:
        """Run this graph with the given values at ports outside of it, e.g.
        the input ports of the GraphNode it is the inner graph of.

        Args:
            inputs: The value of each port outside of this graph that edges
                lead from.
            outputs: Ports whose values to return.
            mode: The evaluation phase every node is set to.

        Returns:
            list: The value of each of `outputs` after the run.
        """
        self._ensure_plan()
        for port, value in inputs.items():
            self._values[((), port)] = value
        self._execute(mode)
        return [self._values.get(self._source((), port)) for port in outputs]

    # endregion

    def run(
        self,
        mode: EvaluationPhase = EvaluationPhase.ALWAYS,
        discretization: "float | dict[Variable, float] | None" = None,
        on_node_done: "Callable[[Node], None] | None" = None,
    ):
        """Run all nodes in execution order.

        Each input receives the value of its incoming edge, or its default
        if it has none. The values are kept in this graph, see
        :meth:`value_of`.

        Args:
            mode: The evaluation phase every node is set to.
            discretization: If given, every
                :class:`~qewton.data.dataloaders.sampler.point_sampler.PointSampler`
                of the graph is switched to mesh mode for this run (see
                :func:`~qewton.data.dataloaders.sampler.point_sampler.discretization_mode`):
                it outputs the vertices of a mesh of its geometry with this
                maximal vertex distance (one value, or one per geometry
                Variable) instead of sampled points. The values can then be
                drawn with ``visualize(port, run=False)`` at the same
                discretization. None (default) samples as usual.
            on_node_done: Called with each node of this graph once it has
                run, in execution order, e.g. to report progress.
        """
        self._ensure_plan()
        if discretization is None:
            self._execute(mode, on_node_done=on_node_done)
            return
        from qewton.data.dataloaders.sampler.point_sampler import (
            PointSampler,
            discretization_mode,
        )

        samplers = [node for node in self.sorted_nodes if isinstance(node, PointSampler)]
        with discretization_mode(samplers, discretization):
            self._execute(mode, on_node_done=on_node_done)

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
        self._ensure_plan()
        self._execute(mode, owners=nodes_to_run)
        for node, edges in zip(self.sorted_nodes, self.sorted_incoming_edges):
            if node == last_node:
                return edges
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
        self._ensure_plan()
        self._execute(mode, owners=nodes_to_run)

    def diagram(self,
                depth: int = 1,
                show_port_names: bool = False,
                show_configs: bool = False) -> GraphPlot:
        """Build a diagram of this graph's structure.

        Args:
            depth: How many levels of composite nodes (subclasses of
                :class:`~qewton.graphs.control_nodes.graph_node.GraphNode`) are
                expanded inline instead of drawn as a single box.
            show_port_names: Whether nodes with more than one input or output
                port also get a name label at each port.
            show_configs: Whether every edge is labeled with the axes of
                its matched data configuration; otherwise they are shown on
                hover.

        Returns:
            GraphPlot: The node-link diagram of this graph.
        """
        from qewton.visualization.plots.graph import GraphPlot

        return GraphPlot(self,
                         depth=depth,
                         show_port_names=show_port_names,
                         show_configs=show_configs)

    def visualize(
        self,
        port: Port | list[Port],
        reference: "Port | Callable | Any" = None,
        error: str | None = "signed",
        mask: Port | None = None,
        plot_type: type["Plot"] | None = None,
        max_vertex_distance: float | dict[Variable, float] = 0.05,
        device: Device | str | None = None,
        share_scale: bool = True,
        color_scale: "ColorScale | None" = None,
        controls: "list[ControlAxis]" = (),
        default_control: "type[ControlAxis] | None" = None,
        variables: list[Variable] | None = None,
        prediction_config: DataConfiguration | None = None,
        reference_config: DataConfiguration | None = None,
        mode: EvaluationPhase = EvaluationPhase.VALIDATION,
        run: bool = True,
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
            mask: A port of this graph, run together with `port`, whose
                value has one component per point of the plotted data,
                e.g. the domain mask of a data loader. Points where it is 0
                or NaN are not drawn in any panel and do not train the
                scales. None (default) draws every point.
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
            color_scale: The color scale of every plot, e.g.
                ``ColorScale(vmin=0.0, vmax=1.0)`` for one fixed range. With
                `reference`, it is shared by the reference and prediction
                panels instead of the scale from `share_scale`; the error
                panel keeps its own scale. None (default) gives each plot its
                own.
            controls: Control axes for chosen dimensions, e.g.
                ``[TimeAxis(t)]``; a control on a 1D factor of a product
                geometry steps through that factor.
            default_control: Control axis class for every other dimension
                not drawn; None for auto_plot's default (a slider). Every
                control is shared by all plots whose data has its axes or
                Variable (see :class:`~qewton.visualization.auto.AutoPlots`).
            variables: Narrows `port` (and `reference`, unless it already has
                exactly this combination) to the slice of these Variables of a
                composed FeatureAxes before plotting. With more than one
                Variable, all plots share one Selector to switch between them,
                which requires them to be scalars. None (default) uses the
                Variable of `port` as declared.
            prediction_config: Used instead of the data configuration of `port`
                in this graph to build the prediction plot and to check
                `reference` against.
            reference_config: Used instead of the data configuration of a port
                `reference` in this graph. Required for plain reference data.
            mode: The evaluation phase every run node is set to; selects which
                split a data loader provides.
            run: Whether to run the nodes needed for `port` first. False
                draws the values of the last run instead, which must have
                been ``run(discretization=max_vertex_distance)`` at the same
                `max_vertex_distance` (the configurations are read in mesh
                mode, so data and configuration agree). Not supported with
                `reference`.
            **plot_kwargs: Passed to
                :func:`~qewton.visualization.auto.auto_plot` for every plot.

        Returns:
            Layout: The plots (with `reference`: reference, prediction and
                error) arranged by
                :func:`~qewton.visualization.layout.auto_layout`; panels in one
                Overlay are named by their legend entries, panels side by side
                by their titles. Show it with ``Figure(layout).show()``.

        Raises:
            ValueError: If `reference` is given for a list of ports, plain
                reference data is given without `reference_config`, the
                Variables or shapes of prediction and reference do not match,
                `mask` has not one value per point of the data, `error` is
                unknown, or auto_plot raises.
        """
        if default_control is not None:
            plot_kwargs["default_control"] = default_control

        ports = port if isinstance(port, list) else [port]

        if reference is not None:
            if not run:
                raise ValueError(
                    "reference needs a run of its own; visualize with run=True."
                )
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
                mask,
                plot_type,
                max_vertex_distance,
                device,
                share_scale,
                color_scale,
                controls,
                variables,
                prediction_config,
                reference_config,
                mode,
                **plot_kwargs,
            )

        from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
        from qewton.visualization.auto import AutoPlots
        from qewton.visualization.layout import auto_layout

        run_ports = (*ports, mask) if mask is not None else tuple(ports)
        nodes_to_run = self._nodes_needed_for({p.node for p in run_ports})
        if device is not None:
            for node in nodes_to_run:
                node.to(device)

        samplers = self._samplers_on_path_to(ports)
        combined_variable = Variable.compose(variables) if variables else None
        value = self._shared_value(variables)
        plots = AutoPlots(controls)
        # The configurations are read in mesh mode: the dims bound to the
        # number of points of a sampler have the number of mesh points.
        with discretization_mode(samplers, max_vertex_distance, device):
            # Without running, the values are those of an earlier run in mesh
            # mode; switching to mesh mode sized the configurations for them.
            if run:
                self._run_nodes_needed_for(run_ports, mode=mode)
            mask_values = self._mask_values(mask)

            # Move right before plotting, not before running - the sampler(s)
            # and model can stay on the set device
            for sampler in samplers:
                for sampled_geometry in sampler.sampled_geometries:
                    sampled_geometry.to_numpy()

            for p in ports:
                data = p.node.backend.to_numpy(self.value_of(p))
                config = p.get_data_configuration(self)
                if combined_variable is not None:
                    data, config = self._narrow_to_variable(
                        data, config, combined_variable
                    )
                data, config, panel_kwargs = self._with_mask(data, config, value, mask_values)
                plots.add(
                    data, config, plot_type, color_scale=color_scale, **panel_kwargs,
                    **plot_kwargs,
                )
        plots.check_controls()
        return auto_layout(plots.plots)

    def _mask_values(self, mask: Port | None) -> "tuple[np.ndarray, Variable] | None":
        """The value of the run port `mask` as numpy, with its Variable;
        None without a mask."""
        if mask is None:
            return None
        config = mask.get_data_configuration(self)
        variable = config.feature_axes.variables if config.feature_axes else None
        if variable is None:
            raise ValueError(f"mask {mask} has no feature Variable.")
        return np.asarray(mask.node.backend.to_numpy(self.value_of(mask))), variable

    @staticmethod
    def _with_mask(
        data: np.ndarray,
        config: DataConfiguration,
        value: "Selector | None",
        mask: "tuple[np.ndarray, Variable] | None",
    ) -> tuple[np.ndarray, DataConfiguration, dict]:
        """`data` and `config` of one panel with the mask as one more
        feature channel, and the auto_plot arguments drawing the panel's
        own value hidden by that channel."""
        if mask is None:
            return data, config, {"value": value}
        mask_data, mask_variable = mask
        if mask_data.shape[:-1] != data.shape[:-1]:
            raise ValueError(
                f"mask has shape {mask_data.shape} for data of shape {data.shape} - "
                "it needs one value per point of the plotted data."
            )
        variable = config.feature_axes.variables
        axes = tuple(
            FeatureAxes(variable * mask_variable) if isinstance(axes, FeatureAxes) else axes
            for axes in config.axes
        )
        data = np.concatenate([data, mask_data.astype(data.dtype)], axis=-1)
        value = value if value is not None else variable
        return data, DataConfiguration(*axes, dtype=config.dtype), {
            "value": value, "mask": mask_variable
        }

    @staticmethod
    def _shared_value(variables: list[Variable] | None) -> "Selector | None":
        """One Selector shared by every plot for several `variables`, so
        choosing one switches every plot; None otherwise."""
        from qewton.visualization.axes import Selector

        return Selector(variables) if variables and len(variables) > 1 else None

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
        mask: Port | None,
        plot_type,
        max_vertex_distance: float | dict[Variable, float],
        device,
        share_scale: bool,
        color_scale,
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
        from qewton.visualization.auto import AutoPlots
        from qewton.visualization.layout import Overlay, auto_layout
        from qewton.visualization.scales import ColorScale
        from qewton.visualization.themes import cmaps

        # A callable reference keeps its samplers in mesh mode until the
        # plots are built: the dims bound to their number of points have
        # the number of mesh points meanwhile.
        with ExitStack() as modes:
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
                run_ports = (port, mask) if mask is not None else (port,)
                nodes_to_run = self._nodes_needed_for({p.node for p in run_ports})
                if device is not None:
                    for node in nodes_to_run:
                        node.to(device)

                samplers = self._samplers_on_path_to((port,))
                modes.enter_context(discretization_mode(samplers, max_vertex_distance, device))
                self._run_nodes_needed_for(run_ports, mode=mode)

                for sampler in samplers:
                    for sampled_geometry in sampler.sampled_geometries:
                        sampled_geometry.to_numpy()

                pred_data = port.node.backend.to_numpy(self.value_of(port))
                pred_config = prediction_config or port.get_data_configuration(self)
                geometry_axes = pred_config.geometry_axes
                if geometry_axes is None:
                    raise ValueError(
                        f"{port}'s own value has no GeometryAxes - a callable "
                        "reference needs the model's own evaluation points to "
                        "call itself at."
                    )
                # One GeometryAxes per factor (a product sampler): every
                # combination of their points, in the order of the data.
                geometry = (
                    self._product_reference_geometry(geometry_axes, samplers)
                    if isinstance(geometry_axes, list)
                    else geometry_axes.geometry
                )
                ref_data = self._reference_to_numpy(reference(geometry.coordinates()), port)
                ref_data = ref_data.reshape(*pred_data.shape[:-1], ref_data.shape[-1])
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

                run_ports = (port, ref_port, mask) if mask is not None else (port, ref_port)
                nodes_to_run = self._nodes_needed_for({p.node for p in run_ports})
                if device is not None:
                    for node in nodes_to_run:
                        node.to(device)

                self._run_nodes_needed_for(run_ports, mode=mode)

                pred_data = port.node.backend.to_numpy(self.value_of(port))
                pred_config = prediction_config or port.get_data_configuration(self)
                ref_data = ref_port.node.backend.to_numpy(self.value_of(ref_port))
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

                run_ports = (port, mask) if mask is not None else (port,)
                nodes_to_run = self._nodes_needed_for({p.node for p in run_ports})
                if device is not None:
                    for node in nodes_to_run:
                        node.to(device)

                with active_discretization_mode(samplers, ref_geometry):
                    self._run_nodes_needed_for(run_ports, mode=mode)

                for sampler in samplers:
                    for sampled_geometry in sampler.sampled_geometries:
                        sampled_geometry.to_numpy()

                pred_data = port.node.backend.to_numpy(self.value_of(port))
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

            value = self._shared_value(variables)
            mask_values = self._mask_values(mask)
            panels = AutoPlots(controls)

            def add_panel(data, config, **kwargs):
                data, config, panel_kwargs = self._with_mask(data, config, value, mask_values)
                return panels.add(data, config, plot_type, **panel_kwargs, **kwargs, **plot_kwargs)

            reference_plot = add_panel(ref_data, ref_config, color_scale=color_scale)
            # The reference's scale (None without color) is shared if asked for.
            if share_scale or color_scale is not None:
                color_scale = reference_plot.color_scale
            add_panel(pred_data, pred_config, color_scale=color_scale)
            names = ["Reference", "Prediction"]

            if error is not None:
                if error == "signed":
                    error_data = pred_data - ref_data
                    error_scale = ColorScale(cmap=cmaps.diverging, symmetric=True)
                elif error == "absolute":
                    error_data = np.abs(pred_data - ref_data)
                    error_scale = ColorScale(vmin=0.0)
                elif error == "relative":
                    diff = pred_data - ref_data
                    threshold = 1e-8 * np.max(np.abs(ref_data))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        error_data = diff / ref_data
                    error_data = np.where(np.abs(ref_data) < threshold, np.nan, error_data)
                    error_scale = ColorScale(cmap=cmaps.diverging, symmetric=True)
                else:
                    raise ValueError(
                        'error must be "signed", "absolute", "relative", or '
                        f"None, got {error!r}."
                    )
                error_kwargs = {} if reference_plot.color_scale is None else {
                    "color_scale": error_scale
                }
                add_panel(error_data, ref_config, **error_kwargs)
                names.append("Error")

            panels.check_controls()
            layout = auto_layout(panels.plots)
            for plot, name in zip(panels.plots, names):
                if isinstance(layout, Overlay):
                    plot.label = name
                else:
                    plot.title = name
            return layout

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
