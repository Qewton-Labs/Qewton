from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from ..configurations.configuration_base import DataConfiguration
from ..optimization.hyperparameter.base import HyperParameter
from ..optimization.base import EvaluationMode


class Port:
    """A class denoting the expected data shape to reach a given node."""

    def __init__(
        self,
        data_configuration: DataConfiguration,
        owner: Node,
        name: str,
        is_required: bool = False,
    ) -> None:
        self.data_configuration = data_configuration
        self.node = owner
        self.required = is_required
        self.name = name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Port):
            return False
        return (
            self.data_configuration == value.data_configuration
            and self.node == value.node
            and self.name == value.name
        )


class Node(ABC):
    """Base class for all nodes to create a pipeline.

    TODO: Do we need a reset method or a validate method here?
    TODO: How about save and load methods?
    """

    class InputKeys(str, Enum):
        INPUT = "input"

    class OutputKeys(str, Enum):
        OUTPUT = "output"

    def __init__(self, name: str = "Node") -> None:
        super().__init__()
        self.name = name  # TODO: make name read-only?
        self.mode: EvaluationMode = EvaluationMode.ALWAYS

    @property
    @abstractmethod
    def input_ports(self) -> dict[str, Port]:
        pass

    @property
    @abstractmethod
    def output_ports(self) -> dict[str, Port]:
        pass

    @abstractmethod
    # TODO: Can we make input better than a dictionary?
    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        pass

    def __call__(self, *arg, **kwds):
        inputs = self._bind_inputs(*arg, **kwds)
        values = tuple(self.run(inputs).values())
        return values[0] if len(values) == 1 else values

    def _bind_inputs(self, *args, **kwargs):
        port_names = list(self.input_ports.keys())

        inputs = {}

        # Assume general args are in order of port names
        for name, value in zip(port_names, args):
            inputs[name] = value

        # Afterwards we add the keyword arguments
        inputs.update(kwargs)

        # Check if all required inputs are provided
        for name, port_info in self.input_ports.items():
            if port_info.required and name not in inputs:
                raise ValueError(f"Missing input: {name}")

        return inputs

    def create_runtime(self) -> _NodeRuntime:
        return _NodeRuntime(self)

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return []

    @property
    def trainable_parameters(self):
        return None

    def to(self, device):
        """Move data stored in this node to a different device (GPU, CPU)"""

    def __getitem__(self, port_name: str) -> Port:
        input_ports = self.input_ports
        if port_name in input_ports.keys():
            return input_ports[port_name]
        output_ports = self.output_ports
        if port_name in output_ports.keys():
            return output_ports[port_name]
        raise ValueError(f"Port {port_name} does not exist")

    def set_mode(self, new_mode: EvaluationMode):
        pass


class _NodeRuntime:
    """Class to manage the runtime state of a node in a pipeline.
    This includes tracking received inputs and whether the node has run.

    TODO: Is this really needed or should the pipeline manage this?
    Advantage of having it here is that each node can manage its own state and one could
    maybe start multiple nodes in parallel more easily?
    """

    def __init__(self, node: Node):
        self.node = node
        self.received_inputs = {}
        self.has_run = False  # optional

    def receive(self, port_name: str, value):
        self.received_inputs[port_name] = value

    def is_ready(self) -> bool:
        for name, port_info in self.node.input_ports.items():
            if port_info.required and name not in self.received_inputs:
                return False
        return not self.has_run

    def run(self) -> dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Node is not ready")

        outputs = self.node.run(self.received_inputs)
        self.received_inputs.clear()
        self.has_run = True
        return outputs
