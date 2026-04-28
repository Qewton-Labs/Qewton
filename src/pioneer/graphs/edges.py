from .nodes import InputPort, Port
from ..config.configuration_base import DataConfiguration


class Edge:
    def __init__(
        self,
        from_port: Port,
        to_port: InputPort,
        data_config: DataConfiguration,
        connects_to_outside: bool = False,
    ):
        self.to_port = to_port
        self.from_port = from_port
        self.data_config = data_config
        self.connects_to_outside = connects_to_outside
