from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypedDict, TypeVar, Generic


from ...data.configurations.configuration_base import DataConfiguration
from ...optimization.hyperparameter.base import HyperParameter


class Port:
    """A class denoting the expected data shape to reach a given node."""

    def __init__(
        self,
        data_configuration: DataConfiguration,
        owner: Node,
        is_required: bool = False,
    ) -> None:
        self.data_configuration = data_configuration
        self.node = owner
        self.required = is_required

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Port):
            return False
        return (
            self.data_configuration == value.data_configuration
            and self.node == value.node
        )


InputPorts = TypeVar("InputPorts", bound=dict[str, Port])
OutputPorts = TypeVar("OutputPorts", bound=dict[str, Port])


class PortDictionary(TypedDict):
    pass


class InputPortDictionary(PortDictionary):
    input: Port


class OutputPortDictionary(PortDictionary):
    output: Port


class Node(ABC, Generic[InputPorts, OutputPorts]):
    """Base class for all nodes to create a pipeline.

    TODO: Do we need a reset method or a validate method here?
    TODO: How about save and load methods?
    """

    def __init__(self, name: str = "Node") -> None:
        super().__init__()
        self.name = name  # TODO: make name read-only?

    @property
    def ports(self) -> None:
        print(f"--- Port Information of Node {self.name}---")
        print("Input ports:")
        for key in self.input_ports.keys():
            print(key)
        print("Output ports:")
        for key in self.output_ports.keys():
            print(key)

    @property
    @abstractmethod
    def input_ports(self) -> InputPorts:
        """Defines the input ports of the node.
        str -> expected data shape + boolean to know if the input is required

        TODO: Do we need a special object for port definitions?
        """

    @property
    @abstractmethod
    def output_ports(self) -> OutputPorts:
        pass

    @abstractmethod
    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        pass

    def __call__(self, *arg, **kwds):
        inputs = self._bind_inputs(*arg, **kwds)
        return self.run(inputs).values()

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

    def to(self, device):
        """Move data stored in this node to a different device (GPU, CPU)"""


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
