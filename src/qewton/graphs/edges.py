from qewton.graphs.nodes import Port


class Edge:
    def __init__(
        self,
        from_port: Port,
        to_port: Port,
        connects_to_outside: bool = False,
    ):
        self.to_port = to_port
        self.from_port = from_port
        self.connects_to_outside = connects_to_outside
