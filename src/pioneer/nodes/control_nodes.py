from .base import Node, InputPort, OutputPort
from ..config.configuration_base import DataConfiguration


class ControlNode(Node):
    """A node that save the data flowing through this node, that
    data can be used/evaluated later.
    """

    def __init__(self, data_config: DataConfiguration, name: str = "ControlNode") -> None:
        super().__init__(name=name)
        self.data_config = data_config
        self.stored_data = None

        self.in_port = InputPort(self.data_config, self, "in_port")
        self.out_port = OutputPort(self.data_config, self, "out_port")

    def run(self):
        self.stored_data = self.in_port.value
        self.out_port.set_value(self.stored_data)

    def reset(self):
        self.stored_data = None
