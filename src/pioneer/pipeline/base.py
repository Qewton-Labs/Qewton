from __future__ import annotations
import warnings

from .nodes.base import Node, Port, _NodeRuntime
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
        from_port: Port,
        to_port: Port,
    ) -> None:
        # Nodes must be in graph
        from_node = from_port.node
        to_node = to_port.node

        self.nodes.add(from_node)
        self.nodes.add(to_node)

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

    def disconnect(self, edge: Edge) -> None:
        self.edges.remove(edge)

    def outgoing_edges(self, node) -> list[Edge]:
        return [e for e in self.edges if e.from_node is node]

    def incoming_edges(self, node) -> list[Edge]:
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
        return PipelineRuntime(self)


class PipelineRuntime:
    """Again the runtime is split form the definition, for easier
    management of multiple runs?
    """

    def __init__(self, graph: Pipeline):
        self.graph = graph
        self.runtime_nodes: dict[Node, _NodeRuntime] = {
            node: node.create_runtime() for node in graph.nodes
        }

    def run(self):
        ready_nodes: set[_NodeRuntime] = set()  # Nodes that are ready to run
        for runtime_node in self.runtime_nodes.values():
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
                    target_rt = self.runtime_nodes[edge.to_node]
                    target_rt.receive(edge.to_port, value)
                    # Check if node is now ready to run
                    if target_rt.is_ready():
                        next_ready_nodes.add(target_rt)

            ready_nodes = next_ready_nodes.copy()
