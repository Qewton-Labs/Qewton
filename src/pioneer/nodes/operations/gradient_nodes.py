from ..base import Node, InputPort, OutputPort
from ...config.configuration_base import DataConfiguration


class GradientTrackingNode(Node):
    """Enables gradient tracking of the data flowing through this node.

    TODO: Currently only in PyTorch okay!!!
    """

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
        self.in_port = InputPort(self.data_config, self)
        self.output_port = OutputPort(self.data_config, self)

    def run(self):
        data = self.in_port.value
        data.requires_grad = True
        self.output_port.set_value(data)
