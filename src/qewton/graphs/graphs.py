from __future__ import annotations
from collections import deque
from contextlib import contextmanager
import inspect
from typing import Any, Callable, TYPE_CHECKING
from warnings import warn

import numpy as np

from qewton.config.axes import FeatureAxes
from qewton.config.data_configurations import DataConfiguration
from qewton.config.devices import Device
from qewton.config.errors import DataConfigMismatchError
from qewton.config.variables import Variable

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
    # Deferred (not just for typing - see visualize()'s own local imports):
    # qewton.visualization / qewton.data.dataloaders.sampler.point_sampler
    # both transitively import qewton.optim, which imports GraphBasedTrainer,
    # which imports Graph itself - a module-level import here would make
    # `import qewton` circular.
    from qewton.visualization.plots.base import Plot
    from qewton.visualization.plots.graph import GraphPlot
    from qewton.visualization.layout import Layout


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

    def _build_path_to_node(self, node: Node) -> set[Node]:
        """
        Builds a path from the root nodes to the specified node.

        Args:
            node (Node): The target node for which the path is to be built.

        Returns:
            set[Node]: A set of nodes that need to be evaluated to reach
                the specified node. Note that the specified node itself is not
                included in the returned set and the nodes are not ordered
                for evaluation.
        """
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
        """Runs the graph execution by iterating through the topologically
        sorted nodes, only up to the provided node.
        Then it returns the last_node and all incoming edges of this node in this
        graph as a dictionary of [Port, Edge].

        Args:
            last_node (Node): The node we want to run the graph up to,
                and return the incoming edges of this node.
            mode (EvaluationPhase, optional): The mode the graph should be run
                in. Defaults to EvaluationPhase.ALWAYS.

        Returns:
            dict[Port, Edge]: The incoming edges of the last_node in this
                graph as a dictionary of [Port, Edge].
        """
        self._check_graph_was_sorted()
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
        """Every node needed to produce a value for `target_nodes`,
        including the target nodes themselves."""
        nodes_to_run = set(target_nodes)
        for node in target_nodes:
            nodes_to_run |= self._build_path_to_node(node)
        return nodes_to_run

    def _samplers_on_path_to(self, ports: tuple[Port, ...]) -> list:
        """Every PointSampler needed to produce a value for any of `ports` -
        on the path to them, or one of their own owning nodes."""
        from qewton.data.dataloaders.sampler.point_sampler import PointSampler

        nodes_to_run = self._nodes_needed_for({p.node for p in ports})
        return [n for n in nodes_to_run if isinstance(n, PointSampler)]

    def _run_nodes_needed_for(
        self, ports: tuple[Port, ...], mode: EvaluationPhase
    ) -> None:
        """Runs every node needed to produce a value for each of `ports`, in
        topological order - including each port's own owning node (unlike
        run_to(), which stops just before its target so a caller can take
        over from there instead). Unlike run_to()/connect(), this doesn't
        warn via _check_graph_was_sorted() - an already-sorted graph is the
        expected, normal precondition here (visualize() is meant to run
        after setup()), not a stale-sort hazard to flag."""
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

    def diagram(self, depth: int = 1) -> GraphPlot:
        from qewton.visualization.plots.graph import GraphPlot

        return GraphPlot(self, depth=depth)

    def visualize(
        self,
        port: Port | list[Port],
        reference: "Port | Callable | Any" = None,
        error: str | None = "signed",
        plot_type: type["Plot"] | None = None,
        max_vertex_distance: float = 0.05,
        device: Device | str | None = None,
        share_scale: bool = True,
        controls=None,
        variables: list[Variable] | None = None,
        prediction_config: DataConfiguration | None = None,
        reference_config: DataConfiguration | None = None,
        mode: EvaluationPhase = EvaluationPhase.VALIDATION,
        **plot_kwargs,
    ) -> "Layout":
        """Runs just enough of the graph to produce a value for `port` (or
        each of several), and builds a Layout of Plots via auto_plot().

        Without `reference`, every PointSampler needed to reach `port` is
        switched into mesh mode for this run (discretization_mode) - a
        deterministic mesh/grid discretization of its geometry, instead of
        whatever batch it would otherwise produce.

        With `reference` set, the model is instead evaluated at the
        reference's own points, so no interpolation is needed to compare
        them.

        Args:
            port: One Port, or a list of Ports, to visualize.
            reference: Already-loaded reference data (requires
                `reference_config`), another Port from this same graph (e.g.
                a DataLoader's "true output" port, for an operator-learning
                comparison), or a callable, evaluated at `port`'s own
                points. Reference data (plain or from a Port) must be built
                from the exact same Variable instance as `port`'s own
                value, unless overridden by `reference_config`/
                `prediction_config` - only supported for a single `port`,
                not a list. None (default) draws `port` alone.
            error: With `reference` set, which difference panel to add:
                "signed" (pred - ref, diverging colormap, scale symmetric
                around 0 - the default), "absolute" (abs(pred - ref),
                sequential colormap, scale from 0), "relative"
                ((pred - ref) / ref, NaN where abs(ref) is ~0, diverging,
                symmetric), or None for no error panel. Ignored without
                `reference`.
            plot_type: Passed through to auto_plot() for every plot - an
                explicit Plot type if auto-selection doesn't apply, or None
                (default) to auto-select.
            max_vertex_distance: Passed to discretization_mode() - caps the
                mesh resolution used while sampling in mesh mode. Unused
                with `reference` set (the reference's own points are used
                instead).
            device: If given, every node needed to reach `port` is moved
                there before running - the same explicit control
                GraphBasedTrainer's own `device=` gives at training time.
                Also controls where mesh-mode's own freshly-generated points
                are built. Left alone (default None) otherwise.
            share_scale: With `reference` set, whether the reference and
                prediction panels share one color Scale, so they read on
                the same range at a glance. The error panel always gets its
                own Scale regardless of this flag.
            controls: A ControlSpec class, instance, or `{axis: class-or-
                instance}` dict, used to resolve a control for any axis left
                over after `port`'s (and, with `reference` set, every
                panel's) own roles - None (default) behaves as SliderSpec.
                With `reference` set, one resolved instance per axis is
                shared across every panel, so a slider dragged in one moves
                them all.
            variables: Narrows `port` (and `reference`, unless it already
                names exactly this combination) down to just these
                Variables' own slice of a composed FeatureAxes before
                anything else runs - e.g. `variables=[U]` to pick out a
                velocity-only quantity from a port declared as a combined
                velocity+pressure Variable, or `variables=[U, P]` to keep
                both but drop some other bundled quantity. A single
                element selects it outright; several same-dim ones also get
                switched between live via one shared dropdown across every
                panel, same as before - the narrowing is new, letting this
                also cover quantities that weren't already jointly
                dispatchable (e.g. dropping a differently-dimensioned P to
                leave just a switchable U); the dropdown itself still
                requires the remaining Variables to share one dim. None
                (default) uses `port`'s (and `reference`'s) own Variable as
                declared.
            prediction_config: A DataConfiguration used in place of
                `port.get_data_configuration(self)` to build/label the
                prediction panel and to check `reference` against. None
                (default) uses `port`'s own config as-is.
            reference_config: Same as `prediction_config`, but for
                `reference` when it's a Port - used in place of
                `reference.get_data_configuration(self)`. Required (and
                used as reference's own config) when `reference` is plain
                data rather than a Port or a callable.
            mode: The EvaluationPhase every DataLoader-like node needed for
                the run is set to - determines which of its splits supplies
                the batch. Defaults to EvaluationPhase.VALIDATION.
            **plot_kwargs: Passed through to auto_plot() for every plot.

        Returns:
            Layout: `Overlay(plot)` for a single port with no reference;
                `Row(*plots)` for a list of ports; with `reference` set, the
                reference/prediction[/error] comparison (`Overlay` for a
                curve family - LinePlot/PathPlot - `Row` otherwise). The
                caller composes: `Figure(graph.visualize(...)).show()`.
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
                controls,
                variables,
                prediction_config,
                reference_config,
                mode,
                **plot_kwargs,
            )

        from qewton.data.dataloaders.sampler.point_sampler import discretization_mode
        from qewton.visualization.auto import auto_plot
        from qewton.visualization.plots.spec import VariableSpec

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
            sampler.sampled_geometry.to_numpy()

        combined_variable = Variable.compose(variables) if variables else None
        controls_kwarg = {} if controls is None else {"controls": controls}
        plots = []
        for p in ports:
            data = p.node.backend.to_numpy(p.value)
            config = p.get_data_configuration(self)  # type: ignore
            if combined_variable is not None:
                data, config = self._narrow_to_variable(data, config, combined_variable)
            plots.append(auto_plot(data, config, plot_type, **controls_kwarg, **plot_kwargs))

        if variables and len(variables) > 1:
            shared_variable_spec = VariableSpec(variables)
            for plot in plots:
                self._redirect_to_shared_variable(plot, variables, shared_variable_spec)

        return Overlay(plots[0]) if len(plots) == 1 else Row(*plots)

    def _visualize_with_reference(
        self,
        port: Port,
        reference: "Port | Callable | Any",
        error: str | None,
        plot_type,
        max_vertex_distance: float,
        device,
        share_scale: bool,
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
        from qewton.visualization.plots.spec import ColorSpec, Scale, VariableSpec

        combined_variable = Variable.compose(variables) if variables else None
        pred_config = prediction_config or port.get_data_configuration(self)
        pred_variable = (
            combined_variable
            if combined_variable is not None
            else (pred_config.feature_axes.variables if pred_config.feature_axes else None)
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
                sampler.sampled_geometry.to_numpy()

            pred_data = port.node.backend.to_numpy(port.value)
            pred_config = prediction_config or port.get_data_configuration(self)
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
            ref_data = np.asarray(reference(points))
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
            ref_config = reference_config
            ref_geometry_axes = ref_config.geometry_axes
            if ref_geometry_axes is None:
                raise ValueError("reference_config has no GeometryAxes.")
            ref_geometry = ref_geometry_axes.geometry

            ref_variable = (
                ref_config.feature_axes.variables if ref_config.feature_axes else None
            )
            self._check_reference_variable(port, pred_variable, ref_variable)

            nodes_to_run = self._nodes_needed_for({port.node})
            if device is not None:
                for node in nodes_to_run:
                    node.to(device)

            samplers = self._samplers_on_path_to((port,))
            with active_discretization_mode(samplers, ref_geometry):
                self._run_nodes_needed_for((port,), mode=mode)

            for sampler in samplers:
                sampler.sampled_geometry.to_numpy()

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
            # auto_plot() dispatches on a DataConfiguration's structural
            # type (e.g. GridGeometry vs. a plain DiscreteGeometry) -
            # port.get_data_configuration(self) still reflects the
            # sampler's own geometry type even after active_discretization_
            # mode gave it ref_geometry's points, so prediction reuses
            # ref_config directly instead.
            pred_config = ref_config
            ref_data = np.asarray(reference)

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
        # heading) - probe once with auto_plot() to find out which.
        probe = auto_plot(ref_data, ref_config, plot_type, **plot_kwargs)
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

        shared_variable_spec = VariableSpec(variables) if variables and len(variables) > 1 else None
        if shared_variable_spec is not None:
            for plot in plots:
                self._redirect_to_shared_variable(plot, variables, shared_variable_spec)

        ref_color = getattr(reference_plot, "color", None)
        pred_color = getattr(prediction_plot, "color", None)
        if (
            share_scale
            and isinstance(ref_color, ColorSpec)
            and isinstance(pred_color, ColorSpec)
        ):
            shared_scale = Scale()
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
            if shared_variable_spec is not None:
                self._redirect_to_shared_variable(
                    error_plot, variables, shared_variable_spec
                )
            error_color = getattr(error_plot, "color", None)
            if isinstance(error_color, ColorSpec):
                error_color.scale = error_scale
                if error_color.cmap is None and diverging:
                    error_color.cmap = "RdBu"
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
        instead, so switching the shared VariableSpec's state moves every
        panel together."""
        from qewton.visualization.plots.spec import PlotSpec

        for value in vars(plot).values():
            if isinstance(value, PlotSpec) and value.variable_or_axes in variables:
                value.variable_or_axes = shared_spec

    def collect_trainable_parameters(self):
        """
        Collects all trainable parameters from all nodes within the graph.

        Returns:
            TrainableParametersCollection: A collection of all trainable parameters.
        """
        params_collection = TrainableParametersCollection()
        for node in self.nodes:
            p = node._trainable_parameters  # pylint: disable=W0212
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

    def __neg__(self):
        from qewton.algorithms.building_blocks.math import Negative

        neg_node = Negative()
        return neg_node(self)

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

    def __setitem__(self, key, value):
        from qewton.algorithms.building_blocks.array_operations import SetItem

        set_item_node = SetItem()
        return set_item_node(self, key, value)

    def dot(self, other):
        from qewton.algorithms.building_blocks.math import Dot

        dot_node = Dot()
        return dot_node(self, other)

    def squeeze(self, dim):
        from qewton.algorithms.building_blocks.array_operations import Squeeze

        squeeze_node = Squeeze(dim)
        return squeeze_node(self)

    def unsqueeze(self, dim):
        from qewton.algorithms.building_blocks.array_operations import Unsqueeze

        unsqueeze_node = Unsqueeze(dim)
        return unsqueeze_node(self)

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
