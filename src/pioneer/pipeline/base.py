from __future__ import annotations
import warnings

from .nodes.base import Node, _NodeRuntime
from .edges.base import Edge


# TODO: Branching graphs are supported. What about cycles?
# Cycles may be better included inside a node itself (e.g. a WhileNode has a
# "sub-pipeline" that runs a time-stepping scheme?).


class Pipeline:

    def __init__(self):
        self.nodes: set[Node] = set()
        self.edges: list[Edge] = []

    def copy(self) -> Pipeline:
        new_pipeline = Pipeline()
        new_pipeline.nodes = self.nodes.copy()
        new_pipeline.edges = self.edges.copy()
        return new_pipeline

    def add_node(self, node: Node) -> None:
        for known_node in self.nodes:
            if known_node.name == node.name:
                warnings.warn(
                    f"Node with name '{node.name}' already in graph!", UserWarning
                )
        self.nodes.add(node)

    def remove_node(self, node: Node) -> None:
        self.nodes.remove(node)
        # Remove all edges connected to this node
        self.edges = [
            edge
            for edge in self.edges
            if edge.from_node is not node and edge.to_node is not node
        ]

    def connect(
        self,
        from_node: Node,
        to_node: Node,
        from_port: str | None = None,
        to_port: str | None = None,
    ) -> None:
        # Nodes must be in graph
        self.add_node(from_node)
        self.add_node(to_node)

        # Ports must exist
        if from_port is not None and from_port not in from_node.output_ports:
            raise ValueError(
                f"{from_node} has no output port '{from_port}', \
                there are only the ports {from_node.output_ports.keys()}"
            )

        if to_port is not None and to_port not in to_node.input_ports:
            raise ValueError(
                f"{to_node} has no input port '{to_port}', \
                there are only the ports {to_node.input_ports.keys()}"
            )
        # If no port is given, use the first one as a default
        if from_port is None:
            from_port = next(iter(from_node.output_ports.keys()))
        if to_port is None:
            to_port = next(iter(to_node.input_ports.keys()))

        # Configurations should match
        out_config = from_node.output_ports[from_port]
        in_config = to_node.input_ports[to_port][0]

        if out_config.fits(in_config):
            raise ValueError("Incompatible input and output data configurations!")

        # Create edge
        edge = Edge(from_node, from_port, to_node, to_port)
        self.edges.append(edge)

    def disconnect(self, edge: Edge) -> None:
        self.edges.remove(edge)

    def outgoing_edges(self, node) -> list[Edge]:
        return [e for e in self.edges if e.from_node is node]

    def incoming_edges(self, node) -> list[Edge]:
        return [e for e in self.edges if e.to_node is node]

    def create_runtime(self) -> PipelineRuntime:
        return PipelineRuntime(self)


class PipelineRuntime:
    """Again the runtime is split form the definition, for easier
    management of multiple runs?
    """

    def __init__(self, graph: Pipeline):
        self.graph = graph
        self.node_runtimes: dict[Node, _NodeRuntime] = {
            node: node.create_runtime() for node in graph.nodes
        }

    def run(self):
        ready_nodes: set[_NodeRuntime] = set()  # Nodes that are ready to run
        for runtime_node in self.node_runtimes.values():
            runtime_node.has_run = False
            # Find all nodes we can start with:
            if runtime_node.is_ready():
                ready_nodes.add(runtime_node)

        while len(ready_nodes) > 0:
            next_ready_nodes: set[_NodeRuntime] = set()

            for runtime_node in ready_nodes:
                outputs = runtime_node.run()
                # Next pass data to all connected nodes
                for edge in self.graph.outgoing_edges(runtime_node.node):
                    value = outputs[edge.from_port]
                    target_rt = self.node_runtimes[edge.to_node]
                    target_rt.receive(edge.to_port, value)
                    # Check if node is now ready to run
                    if target_rt.is_ready():
                        next_ready_nodes.add(target_rt)

            ready_nodes = next_ready_nodes.copy()
