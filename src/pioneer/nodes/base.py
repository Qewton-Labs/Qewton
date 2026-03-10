from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from ..config.configuration_base import DataConfiguration
from ..config.variables import Variable
from ..optim.hyperparameter.base import HyperParameter
from ..optim.base import EvaluationMode


class Port:
    """Denotes the expected data shape of a node."""

    def __init__(
        self,
        data_configuration: DataConfiguration,
        owner: Node,
        key: str,
        is_required: bool = False,
    ) -> None:
        """
        Args:
            data_configuration (DataConfiguration): The configuration denoting the
                expected shape of the data
            owner (Node): The parent node.
            name (str): A name for this port.
            is_required (bool, optional): If the evaluation of the parent node
                requires data from this port. Defaults to False.
        """
        self.data_configuration = data_configuration
        self.node = owner
        self.key = key
        self.required = is_required

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Port):
            return False
        return (
            self.data_configuration == value.data_configuration
            and self.node == value.node
        )


class Node(ABC):
    """Base class for all nodes to create a pipeline.

    TODO: Do we need a validate method here?
    TODO: How about save and load methods?
    """

    class InputKeys(str, Enum):
        """Denotes the names for the input ports of this node.

        Subclasses may override this enum to define additional
        input ports."""

        INPUT = "input"

    class OutputKeys(str, Enum):
        """Denotes the names for the output ports of this node.

        Subclasses may override this enum to define additional
        output ports."""

        OUTPUT = "output"

    def __init__(self, name: str = "Node") -> None:
        """
        Args:
            name (str, optional): The name of this node. Defaults to "Node".
        """
        super().__init__()
        self.name = name
        self.mode: EvaluationMode = EvaluationMode.ALWAYS

    @property
    @abstractmethod
    def input_ports(self) -> list[Port]:
        """Returns all of the input ports of this node.

        Returns:
            list[Port]: A list of input ports.
        """

    @property
    @abstractmethod
    def output_ports(self) -> list[Port]:
        """Returns all of the output ports of this node.

        Returns:
            list[Port]: A list of output ports.
        """

    # TODO: Can we make input better than a dictionary?
    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        if inputs is None:
            if len(self.input_ports.keys()) > 0:
                raise RuntimeError(
                    f"Node needs inputs {list(self.input_ports.keys())}, \
                        but received None."
                )
            inputs = {}
        return self._run(inputs)

    @abstractmethod
    def _run(self, inputs: dict[str, Any]) -> dict[str, Any]:
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

    def __getitem__(self, port_name: str | Variable) -> Port:
        """Allow index of the node with respect to the port keys, names or 
        variables to allow faster access to the ports.

        Args:
            port_name (str | Variable): The name of the port we want to access.

        Raises:
            ValueError: If an unknown port name is provided.

        Returns:
            Port: The port belonging to the input name.
        """ """"""
        if isinstance(port_name, Variable):
            assert len(port_name) == 1, "Can only slice with one single variable"
            var_name: str = next(iter(port_name))
            return self[var_name]

        input_ports = self.input_ports
        if port_name in input_ports.keys():
            return input_ports[port_name]
        output_ports = self.output_ports
        if port_name in output_ports.keys():
            return output_ports[port_name]
        raise ValueError(f"Port {port_name} does not exist")

    def set_mode(self, new_mode: EvaluationMode):
        """Set the when this node should be evaluated, in the training
        process.

        Args:
            new_mode (EvaluationMode): The new evaluation mode.
        """

    def reset(self):
        """Reset the state of the node."""


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
        self.has_run = False

    def receive(self, port_name: str, value):
        """Add an input the to the received inputs

        Args:
            port_name (str): The port this input belongs to.
            value (_type_): The input values.
        """
        self.received_inputs[port_name] = value

    def is_ready(self) -> bool:
        """
        Returns:
            bool: Check if all required inputs have been provided.
        """
        for name, port_info in self.node.input_ports.items():
            if port_info.required and name not in self.received_inputs:
                return False
        return not self.has_run

    def run(self) -> dict[str, Any]:
        """Evaluate the node.

        Raises:
            RuntimeError: If not all required inputs are available.

        Returns:
            dict[str, Any]: The output of the underlying node.
        """
        if not self.is_ready():
            raise RuntimeError("Node is not ready")

        outputs = self.node.run(self.received_inputs)
        self.received_inputs.clear()
        self.has_run = True
        return outputs
