from typing import Any
import copy

from .base import Node, Port
from ..configurations.variables import Variable
from ..configurations.configuration_base import DataConfiguration


class SliceNode(Node):
    """Slices the input into all variables

    TODO: For now assumes that Featureaxis is last axis, change this!!!!
    TODO: Also just assumes that slicing works
    """

    def __init__(
        self, input_data_config: DataConfiguration, name: str = "SliceNode"
    ) -> None:
        super().__init__(name=name)
        self.input_data_config = input_data_config
        self._in_port = Port(self.input_data_config, self, "port", True)
        if (
            input_data_config.feature_axis is ...
            or input_data_config.feature_axis.variables is None
        ):
            raise ValueError("No variables found in this data configuration!")

        self.variables: Variable = input_data_config.feature_axis.variables
        # Save the output ports and the dimension we have to slice
        self._out_ports: dict[str, Port] = {}
        for key, dim in self.variables.items():
            config_copy = copy.deepcopy(input_data_config)
            config_copy.feature_axis.variables = Variable(key, dim)  # type: ignore
            self._out_ports[key] = Port(config_copy, self, key)

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self._in_port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return self._out_ports

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            raise ValueError("Input can not be None!")
        data = inputs[self.InputKeys.INPUT]
        output_dict = {}
        start_dim = 0
        for key, dim in self.variables.items():
            output_dict[key] = data[..., start_dim : start_dim + dim]
            start_dim += dim
        return output_dict
