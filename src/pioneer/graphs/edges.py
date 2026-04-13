from .nodes import InputPort, OutputPort
from ..config.configuration_base import DataConfiguration


class Edge:
    def __init__(
        self,
        from_port: OutputPort,
        to_port: InputPort,
        data_config: DataConfiguration,
    ):
        self.to_port = to_port
        self.from_port = from_port
        self.data_config = data_config
