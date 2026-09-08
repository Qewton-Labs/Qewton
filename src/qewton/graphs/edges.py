from qewton.graphs.nodes import Port
from qewton.config.saving.saving import Serializable


class Edge(Serializable):
    """Represents a connection of two nodes in the graph.

    Args:
        from_port (Port): The port from which the edge starts.
        to_port (Port): The port to which the edge ends.
        connects_to_outside (bool, optional): If the edge connects to a node
            outside of the graph. Defaults to False.
    """

    def __init__(
        self,
        from_port: Port,
        to_port: Port,
        connects_to_outside: bool = False,
    ):
        self.to_port = to_port
        self.from_port = from_port
        self.connects_to_outside = connects_to_outside
