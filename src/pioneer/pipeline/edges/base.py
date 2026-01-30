from dataclasses import dataclass

from ..nodes.base import Node


@dataclass(frozen=True)
class Edge:
    """Class representing a connection between two nodes in a pipeline.
    Is frozen, such that edges are immutable after creation.
    """

    from_node: Node
    from_port: str
    to_node: Node
    to_port: str
