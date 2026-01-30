from typing import Any

from .base import Node
from ...data.configurations.configuration_base import DataConfiguration


class ControlNode(Node):
    """Base class for control nodes that save the data flowing through this node,
    to use it later in different constraints.
    """

    def __init__(self, data_config: DataConfiguration, name: str = "ControlNode") -> None:
        super().__init__(name=name)
        self.data_config = data_config
        self.stored_data = None

    @property
    def input_ports(self) -> dict[str, tuple[DataConfiguration, bool]]:
        return {"input": (self.data_config, True)}

    @property
    def output_ports(self) -> dict[str, DataConfiguration]:
        return {"output": self.data_config}

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Input can not be None!")
        self.stored_data = inputs["input"]
        return {"output": self.stored_data}
