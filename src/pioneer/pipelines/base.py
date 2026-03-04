from __future__ import annotations
import warnings

import networkx as nx
import matplotlib.pyplot as plt

from ..nodes.base import Node, Port, _NodeRuntime, EvaluationMode
from .edges.base import Edge
from ..algorithms.base import AlgorithmNode
from ..constraints.base import Constraint


# TODO: Branching graphs are supported. What about cycles?
# Cycles may be better included inside a node itself (e.g. a WhileNode has a
# "sub-pipeline" that runs a time-stepping scheme?).


class Pipeline:
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
        self.nodes: set[Node] = set[Node]()
        self.constrain_nodes: set[Constraint] = set[Constraint]()
        self.algorithm_nodes: set[AlgorithmNode] = set[AlgorithmNode]()
        self.edges: list[Edge] = []
        self.name = name
        self.mode = EvaluationMode.ALWAYS

    def copy(self) -> Pipeline:
        """Creates a (not deep) copy of this pipeline.

        Returns:
            Pipeline: The copied pipeline.
        """
        new_pipeline = Pipeline()
        new_pipeline.nodes = self.nodes.copy()
        new_pipeline.edges = self.edges.copy()
        return new_pipeline

    def add_node(self, node: Node, check_warning=True) -> None:
        """Adds a node to this pipeline.

        Args:
            node (Node): The node that is added.
            check_warning (bool, optional): Whether it is checked, that
                a node with the same name already exists in this pipeline.
                Defaults to True.
        """
        if check_warning:
            for known_node in self.nodes:
                if known_node.name == node.name:
                    warnings.warn(
                        f"Node with name '{node.name}' already in graph!", UserWarning
                    )
        self.nodes.add(node)
        if isinstance(node, Constraint):
            self.constrain_nodes.add(node)
        if isinstance(node, AlgorithmNode):
            self.algorithm_nodes.add(node)

    def remove_node(self, node: Node) -> None:
        """Deletes a given node from this pipeline.

        Args:
            node (Node): The node that should be deleted.
        """
        self.nodes.remove(node)
        if isinstance(node, Constraint):
            self.constrain_nodes.remove(node)
        if isinstance(node, AlgorithmNode):
            self.algorithm_nodes.remove(node)
        # Remove all edges connected to this node
        self.edges = [
            edge
            for edge in self.edges
            if edge.from_node is not node and edge.to_node is not node
        ]

    def connect(
        self,
        from_: Node | Port,
        to_: Node | Port,
    ) -> None:
        """Connect two nodes (which are automatically added to the pipeline, if they
        are not part of it) via an edge. When evaluating the pipeline data will be
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
        from_port = self._check_connect(from_, check_input=False)
        to_port = self._check_connect(to_, check_input=True)

        from_node = from_port.node
        to_node = to_port.node
        # Nodes must be added to the graph
        self.add_node(from_node, check_warning=False)
        self.add_node(to_node, check_warning=False)

        # Configurations should match
        out_config = from_port.data_configuration
        in_config = to_port.data_configuration

        if not in_config.fits(out_config):
            raise ValueError("Incompatible input and output data configurations!")

        # Create edge
        from_port_name = next(
            (k for k, v in from_node.output_ports.items() if v == from_port)
        )
        to_port_name = next((k for k, v in to_node.input_ports.items() if v == to_port))
        edge = Edge(from_node, from_port_name, to_node, to_port_name)
        self.edges.append(edge)

    def _check_connect(self, user_input: Port | Node, check_input: bool = True) -> Port:
        if isinstance(user_input, Port):
            return user_input
        ports = user_input.input_ports if check_input else user_input.output_ports
        if len(ports) != 1:
            raise ValueError(
                f"Node '{user_input.name}' has multiple output ports. "
                "Specify the port explicitly."
            )
        return next(iter(ports.values()))

    def disconnect(self, edge: Edge) -> None:
        """Remove an edge from this pipeline"""
        self.edges.remove(edge)

    def outgoing_edges(self, node) -> list[Edge]:
        """Obtain all edges outgoing from a node.

        Args:
            node (_type_): The node we want to check.

        Returns:
            list[Edge]: A list of outgoing edges.
        """
        return [e for e in self.edges if e.from_node is node]

    def incoming_edges(self, node) -> list[Edge]:
        """Obtain all edges incoming to a node.

        Args:
            node (_type_): The node we want to check.

        Returns:
            list[Edge]: A list of incoming edges.
        """
        return [e for e in self.edges if e.to_node is node]

    def validate(self) -> None:
        """Validate that all required input ports of each node are connected."""
        for node in self.nodes:
            for port_name, port_config in node.input_ports.items():
                if port_config.required:  # Check if Input is needed
                    # Check if this port has at least one incoming edge
                    incoming_edges = [
                        e
                        for e in self.edges
                        if e.to_node is node and e.to_port == port_name
                    ]

                    if len(incoming_edges) == 0:
                        raise ValueError(
                            f"Node '{node.name}' has required input port '{port_name}' "
                            f"that is not connected!"
                        )

    def create_runtime(self) -> PipelineRuntime:
        """Creates a runtime object of this pipeline, such that multiple pipelines
        can be evaluated independently.
        """
        return PipelineRuntime(self)

    def set_mode(self, new_mode: EvaluationMode, include_constraints: bool = True):
        """Set the process mode for the given training phase.

        Args:
            new_mode (EvaluationMode): The new evaluation mode.
        """
        self.mode = new_mode
        nodes_to_set = (
            self.nodes if include_constraints else self.nodes - self.constrain_nodes
        )
        for node in nodes_to_set:
            node.set_mode(new_mode)

    def setup(self):
        """Setup all nodes for evaluation."""
        for node in self.algorithm_nodes:
            node.setup()

    def visualize(self):
        # TODO: Just some quick way to visualize a graph, nothing final
        graph_visual = nx.DiGraph()
        nodes_list = [node.name for node in self.nodes]
        edges_list = [(edge.from_node.name, edge.to_node.name) for edge in self.edges]

        graph_visual.add_nodes_from(nodes_list)
        graph_visual.add_edges_from(edges_list)

        # Automatically compute node positions
        pos = nx.planar_layout(graph_visual)  # nice spacing with reproducibility

        # Draw nodes
        node_sizes = [
            len(n) * 200 for n in graph_visual.nodes()
        ]  # scale size with label length
        nx.draw_networkx_nodes(
            graph_visual, pos, node_size=node_sizes, node_color="skyblue", alpha=0.9
        )

        # Draw edges with arrows for direction
        nx.draw_networkx_edges(
            graph_visual,
            pos,
            arrowstyle="->",
            arrowsize=20,
            edge_color="gray",
            width=5,
        )

        # Draw labels
        nx.draw_networkx_labels(graph_visual, pos, font_size=10, font_color="black")

        # Remove axes for cleaner look
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class PipelineRuntime:
    """Again the runtime is split form the definition, for easier
    management of multiple runs?
    """

    def __init__(self, graph: Pipeline):
        self.graph = graph
        self.runtime_nodes: dict[Node, _NodeRuntime] = {}
        for node in graph.nodes:
            if isinstance(node, Constraint):
                if EvaluationMode.ALWAYS == node.mode:
                    pass
                elif node.mode != graph.mode:
                    continue
            self.runtime_nodes[node] = node.create_runtime()

    def run(self):
        ready_nodes: set[_NodeRuntime] = set()  # Nodes that are ready to run
        for runtime_node in self.runtime_nodes.values():
            runtime_node.has_run = False
            # Find all nodes we can start with:
            if runtime_node.is_ready():
                ready_nodes.add(runtime_node)

        # This already allows for branching, since we just run all nodes
        # that can run. All other nodes just wait.
        while len(ready_nodes) > 0:
            next_ready_nodes: set[_NodeRuntime] = set()

            for runtime_node in ready_nodes:
                outputs = runtime_node.run()
                # Next pass data to all connected nodes
                for edge in self.graph.outgoing_edges(runtime_node.node):
                    value = outputs[edge.from_port]
                    target_rt = self.runtime_nodes[edge.to_node]
                    target_rt.receive(edge.to_port, value)
                    # Check if node is now ready to run
                    if target_rt.is_ready():
                        next_ready_nodes.add(target_rt)

            ready_nodes = next_ready_nodes.copy()
