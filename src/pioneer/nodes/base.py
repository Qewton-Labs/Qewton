from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from ..config.configuration_base import DataConfiguration
from ..config.variables import Variable
from ..optim.hyperparameter.base import HyperParameter
from ..optim.base import EvaluationMode

class NO_DEFAULT:
    """Sentinel value to denote that no default value is provided for a parameter."""
    pass

class Port:
    """Denotes the expected data shape of a node."""

    def __init__(
        self,
        data_configuration: DataConfiguration,
        node: Node,
        name: str,
    ) -> None:
        """
        Args:
            data_configuration (DataConfiguration): The configuration denoting the
                expected shape of the data
            owner (Node): The parent node.
            name (str): A name for this port.
        """
        self.data_configuration = data_configuration
        self.node = node
        self.name = name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Port):
            return False
        return (
            self.data_configuration == value.data_configuration
            and self.node == value.node
        )

# Probleme: nodes könnten mehrfach in einer pipeline auftauchen

class InputPort(Port):
    """Denotes an input port of a node."""
    def __init__(self,
                 data_configuration: DataConfiguration,
                 node: Node,
                 name: str,
                 default: Any = NO_DEFAULT()):
        super().__init__(data_configuration, node, name)
        self.default = default
        self.connected_ports = []
    
    @property
    def is_required(self):
        return isinstance(self.default, NO_DEFAULT)
    
    def set_connected_port(self, port: OutputPort, pipeline_id: int):
        if len(self.connected_ports) <= pipeline_id:
            self.connected_ports.extend([None] * (pipeline_id - len(self.connected_ports) + 1))
        self.connected_ports[pipeline_id] = port
    
    @property
    def value(self):
        if self.connected_ports[self.node.pipeline_id] is not None:
            return self.connected_ports[self.node.pipeline_id].value
        if not self.is_required:
            return self.default
        raise ValueError(f"Input port {self.name} is required but no value is provided.")

class OutputPort(Port):
    """Denotes an output port of a node."""
    def __init__(self, data_configuration: DataConfiguration, node: Node, name: str):
        super().__init__(data_configuration, node, name)
        self._value = None
        self._current_data_config = [] # the updated data config for each pipeline
    
    @property
    def current_data_config(self):
        return self._current_data_config[self.node.pipeline_id]
    
    def set_current_data_config(self, data_config: DataConfiguration, pipeline_id: int):
        if len(self._current_data_config) <= pipeline_id:
            self._current_data_config.extend([None] * (pipeline_id - len(self._current_data_config) + 1))
        self._current_data_config[pipeline_id] = data_config
    
    def set_value(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

class Node(ABC):
    """Base class for all nodes to create a pipeline.

    TODO: Do we need a validate method here?
    TODO: How about save and load methods?
    """

    def __init__(self, name: str = "Node") -> None:
        """
        Args:
            name (str, optional): The name of this node. Defaults to "Node".
        """
        super().__init__()
        self.name = name
        self.mode: EvaluationMode = EvaluationMode.ALWAYS
        self.pipeline_id: int | None = None
        
        self._input_ports: list[InputPort] = None
        self._output_ports: list[OutputPort] = None

    def setup(self) -> None:
        """Creates the underlying algorithm instance (e.g. creates the
        neural network)

        This should not happen in the __init__ call, given that in the
        HyperParameter tuning we need to recreated the underlying algorithm
        instance, but dont want to create a new node inside our graph.
        """
        pass

    @property
    def input_ports(self) -> list[InputPort]:
        """Returns all of the input ports of this node.

        Returns:
            list[InputPort]: A list of input ports.
        """
        if self._input_ports is None:
            self._input_ports = [v for v in vars(self).values() if isinstance(v, InputPort)]
        return self._input_ports

    @property
    def output_ports(self) -> list[OutputPort]:
        """Returns all of the output ports of this node.

        Returns:
            list[OutputPort]: A list of output ports.
        """
        if self._output_ports is None:
            self._output_ports = [v for v in vars(self).values() if isinstance(v, OutputPort)]
        return self._output_ports

    def run(self) -> None:
        raise NotImplementedError("The run method must be implemented by subclasses of Node.")

    def __call__(self, **kwargs):
        for port in self.input_ports:
            if port.name not in kwargs and port.is_required:
                raise ValueError(f"Missing required input: {port.name}")
            port.set_value(kwargs[port.name])
        self.run()
        out_dict = {}
        for port in self.output_ports:
            out_dict[port.name] = port.value
        if len(out_dict) == 1:
            return out_dict.values()[0]
        return out_dict

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        """Returns tunable parameters of this node."""
        # This checks all self. objects, which makes it save for any subclass
        # but it is still more efficient to just overwrite this in the subclass
        # and directly define the list of all HyperParameter
        return [v for v in vars(self).values() if isinstance(v, HyperParameter)]

    @property
    def trainable_parameters(self) -> Any:
        """Returns trainable parameters of this node."""

    def to(self, device):
        """Move data stored in this node to a different device (GPU, CPU)"""

    def set_mode(self, new_mode: EvaluationMode):
        """Set the when this node should be evaluated, in the training
        process.

        Args:
            new_mode (EvaluationMode): The new evaluation mode.
        """

    def reset(self):
        """Reset the state of the node."""
