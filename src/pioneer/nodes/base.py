from __future__ import annotations
from abc import ABC
from typing import Any

from ..config.configuration_base import DataConfiguration
from ..optim.hyperparameter.base import HyperParameter
from ..optim.base import EvaluationPhase
from ..optim.trainer.trainable_parameters import (
    _TrainableParameterBase,
    TrainableParameters,
)


class NO_DEFAULT:
    """Sentinel value to denote that no default value is provided for a parameter."""


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


class InputPort(Port):
    """Denotes an input port of a node."""

    def __init__(
        self,
        data_configuration: DataConfiguration,
        node: Node,
        name: str = "Input",
        default: Any = NO_DEFAULT(),
    ):
        super().__init__(data_configuration, node, name)
        self.default = default
        self.connected_ports = []

        # a value that is used only when one uses __call__ instead of pipelines
        # order is: 1) check manual execution value,
        #           2) check connected ports,
        #           3) check default value
        # a bit hacky
        self._manual_execution_value = None

    @property
    def is_required(self):
        return isinstance(self.default, NO_DEFAULT)

    def set_connected_port(self, port: OutputPort | None, pipeline_id: int):
        if len(self.connected_ports) <= pipeline_id:
            self.connected_ports.extend(
                [None] * (pipeline_id - len(self.connected_ports) + 1)
            )
        self.connected_ports[pipeline_id] = port

    def set_manual_value(self, value):
        self._manual_execution_value = value

    def clear_manual_value(self):
        self._manual_execution_value = None

    @property
    def value(self):
        if self._manual_execution_value is not None:
            return self._manual_execution_value
        if self.connected_ports[self.node.pipeline_id] is not None:
            return self.connected_ports[self.node.pipeline_id].value
        if not self.is_required:
            return self.default
        raise ValueError(f"Input port {self.name} is required but no value is provided.")


class OutputPort(Port):
    """Denotes an output port of a node."""

    def __init__(
        self, data_configuration: DataConfiguration, node: Node, name: str = "Output"
    ):
        super().__init__(data_configuration, node, name)
        self._value = None
        self._current_data_config = []  # the updated data config for each pipeline

    @property
    def current_data_config(self):
        return self._current_data_config[self.node.pipeline_id]

    def set_current_data_config(self, data_config: DataConfiguration, pipeline_id: int):
        if len(self._current_data_config) <= pipeline_id:
            self._current_data_config.extend(
                [None] * (pipeline_id - len(self._current_data_config) + 1)
            )
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
        self.mode: EvaluationPhase = EvaluationPhase.ALWAYS
        self.pipeline_id: int = 0

        self._input_ports: list[InputPort] | None = None
        self._output_ports: list[OutputPort] | None = None

    def setup(self) -> None:
        """Creates the underlying algorithm instance (e.g. creates the
        neural network)

        This should not happen in the __init__ call, given that in the
        HyperParameter tuning we need to recreate the underlying algorithm
        instance, but dont want to create a new node inside our graph.
        """

    @property
    def input_ports(self) -> list[InputPort]:
        """Returns all of the input ports of this node.

        Returns:
            list[InputPort]: A list of input ports.
        """
        if self._input_ports is None:
            self._input_ports = [
                v for v in vars(self).values() if isinstance(v, InputPort)
            ]
        return self._input_ports

    @property
    def output_ports(self) -> list[OutputPort]:
        """Returns all of the output ports of this node.

        Returns:
            list[OutputPort]: A list of output ports.
        """
        if self._output_ports is None:
            self._output_ports = [
                v for v in vars(self).values() if isinstance(v, OutputPort)
            ]
        return self._output_ports

    def run(self) -> None:
        raise NotImplementedError(
            "The run method must be implemented by subclasses of Node."
        )

    def __call__(self, **kwargs):
        for port in self.input_ports:
            if port.name not in kwargs and port.is_required:
                raise ValueError(f"Missing required input: {port.name}")
            port.set_manual_value(
                kwargs[port.name],
            )

        self.run()

        for port in self.input_ports:
            port.clear_manual_value()

        out_dict = {}
        for port in self.output_ports:
            out_dict[port.name] = port.value
        if len(out_dict) == 1:
            return next(iter(out_dict.values()))
        return out_dict

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        """Returns tunable parameters of this node."""
        # This checks all self. objects, which makes it save for any subclass
        # but it is still more efficient to just overwrite this in the subclass
        # and directly define the list of all HyperParameter
        return [v for v in vars(self).values() if isinstance(v, HyperParameter)]

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        """Returns trainable parameters of this node."""
        return TrainableParameters.create_empty()

    def to(self, device):
        """Move data stored in this node to a different device (GPU, CPU)"""

    def set_mode(self, new_mode: EvaluationPhase):
        """Set the current phase/mode in the training process. Some
        nodes behave differently depending on the current mode.
        E.g. disabling Dropout when in validation.

        Args:
            new_mode (EvaluationPhase): The new evaluation mode.
        """
        self.mode = new_mode

    def reset(self):
        """Reset the state of the node."""

    def set_pipeline_id(self, pipeline_id: int):
        self.pipeline_id = pipeline_id
