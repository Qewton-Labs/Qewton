from typing import Any

from .base import Node, Port, InputPortDictionary, OutputPortDictionary
from ...data.configurations.configuration_base import DataConfiguration


class ControlNode(Node):
    """Base class for control nodes that save the data flowing through this node,
    to use it later in different constraints.
    """

    def __init__(self, data_config: DataConfiguration, name: str = "ControlNode") -> None:
        super().__init__(name=name)
        self.data_config = data_config
        self.stored_data = None

        self._port = Port(self.data_config, self, True)

    @property
    def input_ports(self) -> InputPortDictionary:
        return {"input": self._port}

    @property
    def output_ports(self) -> OutputPortDictionary:
        return {"output": self._port}

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Input can not be None!")
        self.stored_data = inputs["input"]
        return {"output": self.stored_data}
