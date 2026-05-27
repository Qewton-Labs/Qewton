from __future__ import annotations
from abc import ABC
from copy import deepcopy
from enum import Enum
from typing import Any, Callable
from typing import Annotated, get_type_hints, get_origin, get_args
import inspect
import warnings

from ..config.data_configurations import DataConfiguration
from ..config.backend import Backend, TensorType
from ..optim.parameters.hyperparameter_base import HyperParameter
from ..optim.base import EvaluationPhase
from ..optim.parameters.trainable_parameters import (
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
        self._value = None

    # # TODO: is this really necessary?
    # def __eq__(self, value: object) -> bool:
    #     if not isinstance(value, Port):
    #         return False
    #     return (
    #         self.data_configuration == value.data_configuration
    #         and self.node == value.node
    #         and self.name == value.name
    #     )

    # def __hash__(self) -> int:
    #     return hash((self.data_configuration, self.node, self.name))

    def duplicate_with_new_owner(
        self, new_owner: Node, new_name: str | None = None
    ) -> Port:
        return type(self)(
            data_configuration=self.data_configuration,
            node=new_owner,
            name=new_name if new_name is not None else self.name,
        )

    def reset_value(self):
        self._value = None

    def set_value(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


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
        self._value = default

    def duplicate_with_new_owner(
        self, new_owner: Node, new_name: str | None = None
    ) -> Port:
        new_port = super().duplicate_with_new_owner(new_owner, new_name)
        new_port.default = self.default  # type: ignore
        return new_port

    @property
    def is_required(self):
        return isinstance(self.default, NO_DEFAULT)

    def clear_value(self):
        self._value = self.default


class OutputPort(Port):
    """Denotes an output port of a node."""

    def __init__(
        self, data_configuration: DataConfiguration, node: Node, name: str = "Output"
    ):
        super().__init__(data_configuration, node, name)


class NodeState(Enum):
    FIXED = 1
    UNINITIALIZED = 2
    INITIALIZED = 3
    TRAINED = 4


class Node(ABC):
    """Base class for all nodes to create a graph.

    TODO: How about save and load methods?
    """

    _node_id_counter = 0
    _tracking_phase: bool = False

    def __init__(
        self,
        name: str = "Node",
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = Backend,
    ) -> None:
        """
        Args:
            name (str, optional): The name of this node. Defaults to "Node".
            state (NodeState, optional): The initial state of this node.
                Defaults to NodeState.FIXED.
        """
        super().__init__()
        self.name = name
        self._state = state
        self.backend = backend
        self.mode: EvaluationPhase = EvaluationPhase.ALWAYS

        self._input_ports, self._output_ports = self._build_ports(
            self.forward, self, backend
        )

        self.node_id = type(self)._node_id_counter
        type(self)._node_id_counter += 1

    @classmethod
    def set_tracking(cls, set_active: bool):
        cls._tracking_phase = set_active

    @classmethod
    def _build_ports(
        cls, func: Callable, owner: Node, backend: type[Backend[TensorType]]
    ) -> tuple[list[InputPort], list[OutputPort]]:
        call_sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        input_ports = []
        output_ports = []
        # Build input ports:
        for name, param in call_sig.parameters.items():
            hint = type_hints.get(name, param.annotation)
            _, config = cls._unwrap_annotated(hint, owner)
            input_ports.append(
                InputPort(
                    config,
                    node=owner,
                    name=name,
                    default=(
                        NO_DEFAULT
                        if param.default is inspect.Parameter.empty
                        else param.default
                    ),
                )
            )
        cls._set_port_backend(input_ports, backend)
        # Build output ports
        return_values = type_hints.get("return", inspect.Signature.empty)
        if return_values is None or return_values is inspect.Signature.empty:
            return input_ports, output_ports

        if get_origin(return_values) is tuple:
            outputs = list(get_args(return_values))
        else:
            outputs = [return_values]

        for i, output in enumerate(outputs):
            _, config = cls._unwrap_annotated(output, owner)
            output_ports.append(OutputPort(config, node=owner, name=f"output_{i}"))

        cls._set_port_backend(output_ports, backend)

        return input_ports, output_ports

    @classmethod
    def _set_port_backend(cls, ports: list[Port], backend: type[Backend[TensorType]]):
        if backend == Backend:
            return
        for port in ports:
            port.data_configuration.set_dtype(backend.standard_datatype())

    @classmethod
    def _unwrap_annotated(cls, type_hint, owner):
        """Return (base_type, config)."""
        if get_origin(type_hint) is Annotated:
            base, *meta = get_args(type_hint)
            config = (
                next(
                    (m for m in meta if isinstance(m, (DataConfiguration, Callable))),
                    None,
                )
                or DataConfiguration.empty()
            )
            if isinstance(config, Callable):
                config = config(owner)
            return base, config
        return type_hint, DataConfiguration.empty()

    def copy_data_configs(self):
        copy_memo = {}
        dynamic_data_configs = {}
        for port in self.input_ports + self.output_ports:
            dynamic_data_configs[port] = self.copy_data_config_of_port(port, copy_memo)
        return dynamic_data_configs

    def copy_data_config_of_port(self, port, copy_memo):
        return deepcopy(port.data_configuration, copy_memo)

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
        return self._input_ports

    @property
    def output_ports(self) -> list[OutputPort]:
        """Returns all of the output ports of this node.

        Returns:
            list[OutputPort]: A list of output ports.
        """
        return self._output_ports

    def run(self) -> None:
        input_data = [in_port.value for in_port in self.input_ports]
        output_data = self(*input_data)
        if len(self.output_ports) == 1:
            self.output_ports[0].set_value(output_data)
        elif len(self.output_ports) > 1:
            for i, out_port in enumerate(self.output_ports):
                out_port.set_value(output_data[i])

    def __call__(self, *args, **kwargs) -> Any:
        if self._tracking_phase:
            return self._track(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The default node can not be called, "
            "this method needs to be overwritten in the subclasses."
        )

    def _track(self, *args, **kwargs):
        """Track the data passed through this node. This can be used to implement
        graph tracking for debugging or visualization purposes.
        """
        from .graphs import TrackingObject

        for i, tracking_object in enumerate(args):
            if isinstance(tracking_object, TrackingObject):
                if tracking_object.last_output_port is not None:
                    tracking_object.current_graph_tracked.connect(  # type: ignore
                        tracking_object.last_output_port, self.input_ports[i]
                    )
                else:
                    tracking_object.add_to_port(self.input_ports[i])
            else:  # some default value was set:
                self.input_ports[i].default = tracking_object
        for key, tracking_object in kwargs.items():
            if isinstance(tracking_object, TrackingObject):
                if tracking_object.last_output_port is not None:
                    tracking_object.current_graph_tracked.connect(  # type: ignore
                        tracking_object.last_output_port, self.get_input_port(key)
                    )
                else:
                    tracking_object.add_to_port(self.get_input_port(key))
            else:
                self.get_input_port(key).default = tracking_object

        output_trackers = []
        for out_port in self.output_ports:
            output_trackers.append(TrackingObject(out_port))
        if len(output_trackers) > 0:
            return (
                tuple(output_trackers) if len(output_trackers) > 1 else output_trackers[0]
            )
        return None

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
        return TrainableParameters.create_empty(self.node_id)

    @property
    def _trainable_parameters(self) -> _TrainableParameterBase:
        """Internal method to return trainable parameters of this node. This is used
        to collect trainable parameters from all nodes in a graph.
        """
        return (
            self.trainable_parameters
            if self.state != NodeState.FIXED
            else TrainableParameters.create_empty(self.node_id)
        )

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
        """Reset the node."""

    def set_state(self, new_state: NodeState):
        """Set the current state of this node. This can be used to track the
        state of the node during the training process.

        Args:
            new_state (NodeState): The new state of this node.
        """
        self._state = new_state

    @property
    def state(self):
        return self._state

    def fix_node_state(self) -> None:
        """Fix the state of the node so it will not be
        trained or recreated!
        """
        if self.state == NodeState.UNINITIALIZED:
            warnings.warn(
                "This Algorithm is not initialized, fixing it now may lead \
                    to unexpected behavior. Maybe call .setup() first?",
                UserWarning,
            )
            return
        self._state = NodeState.FIXED

    def update_data_configs(self, updated_port, config_dict, dynamic_configs):
        """
        Default implementation, could be overridden in subclasses where necessary.
        """
        # First we check if the port that was connected has been changed:
        port_config = dynamic_configs[updated_port]
        # print(port_config)
        port_config_was_updated = port_config.update_config(config_dict)
        # print(port_config, port_config_was_updated)
        if not port_config_was_updated:
            # No change -> we are done
            return set()
        # Iterate over all ports of the current node, since updates to the axes
        # may not happen in place (e.g. ellipsis are replaced), we need to check all
        # port configurations separately.
        updated_ports = (
            {updated_port} if not isinstance(updated_port, InputPort) else set()
        )
        for c_port, config in dynamic_configs.items():
            if c_port != updated_port:  # original port already updated
                if config.update_config(config_dict):
                    updated_ports.add(c_port)
        return updated_ports

    def get_input_port(self, name):
        for port in self.input_ports:
            if port.name == name:
                return port
        raise ValueError(f"No input port with name {name} found in node {self.name}.")

    def get_output_port(self, name):
        for port in self.output_ports:
            if port.name == name:
                return port
        raise ValueError(f"No output port with name {name} found in node {self.name}.")

    def copy(self):
        """Creates a copy of this node, with the same inner operations, parameters
        etc., but with new input and output ports.
        """
        from .control_nodes.graph_node import CopiedNode

        return CopiedNode(self)
