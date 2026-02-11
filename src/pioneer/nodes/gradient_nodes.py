import torch

from .base import Node, Port
from ..config.configuration_base import DataConfiguration


class GradientTrackingNode(Node):
    """Enables gradient tracking of the data flowing through this node."""

    def __init__(
        self, data_config: DataConfiguration, name: str = "TrackingNode"
    ) -> None:
        """
        Args:
            data_config (DataConfiguration): The expected data shape.
            name (str, optional): The name of this node. Defaults to "TrackingNode".
        """
        super().__init__(name=name)
        self.data_config = data_config
        self._port = Port(self.data_config, self, "port", True)

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self._port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return {self.OutputKeys.OUTPUT: self._port}

    def run(
        self, inputs: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if inputs is None:
            raise ValueError("Input can not be None!")
        data = inputs[self.InputKeys.INPUT]
        data.requires_grad = True
        return {self.OutputKeys.OUTPUT: data}
