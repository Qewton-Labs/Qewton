from __future__ import annotations
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Union,
    Annotated,
    get_type_hints,
    get_origin,
    get_args,
)
import inspect
import warnings

from qewton.config.data_configurations import DataConfiguration
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.trainable_parameters import (
    _TrainableParameterBase,
    TrainableParameters,
)

# region: Ports


class NO_DEFAULT:
    """Sentinel value to denote that no default value is provided for a parameter."""


class Port:
    """Represents an input or output connection of a node. Ports can be connected
    in a graph to create a computation graph structure.
    Each port has a data configuration that denotes the expected shape of the data
    that is passed through this port. This can be used to check the consistency of
    the graph and to automatically update the data configurations of other ports in
    the graph when one port is updated.

    Args:
        data_configuration (DataConfiguration): The configuration denoting the
            expected shape of the data
        owner (Node): The parent node.
        name (str): A name for this port.
    """

    def __init__(
        self,
        data_configuration: DataConfiguration,
        node: Node,
        name: str,
    ) -> None:
        self._static_data_configuration = data_configuration
        self.node = node
        self.name = name
        self._value = None

    @property
    def data_configuration(self) -> DataConfiguration:
        return self._static_data_configuration

    def get_data_configuration(self, graph) -> DataConfiguration | None:
        if self.node in graph.dynamic_data_configs:
            return graph.dynamic_data_configs[self.node][self]
        return None

    def update_static_data_configuration(self, new_config: DataConfiguration):
        self._static_data_configuration = new_config

    def duplicate_with_new_owner(
        self, new_owner: Node, new_name: str | None = None
    ) -> Port:
        """Copies this port information and transfers to a new node.

        Args:
            new_owner (Node): The new owner of the copied port.
            new_name (str | None, optional): A new name. Defaults to None
                and just copied the name of this port.

        Returns:
            Port: The new port.
        """
        return type(self)(
            data_configuration=self.data_configuration,
            node=new_owner,
            name=new_name if new_name is not None else self.name,
        )

    def reset_value(self):
        self._value = None

    def set_value(self, value):
        """Stores a value inside this port to pass through information in
        the graph. Other nodes/ports can access this value if they are
        connected via edges.

        Args:
            value (_type_): The value to be stored in this port.
        """
        self._value = value

    @property
    def value(self):
        """Returns the value stored in this port."""
        return self._value


class InputPort(Port):
    """Denotes an input port of a node.

    Args:
        data_configuration (DataConfiguration): The configuration denoting the
            expected shape of the data
        owner (Node): The parent node.
        name (str): A name for this port.
        default (any, optional): A default value for this port.
            If no value is passed, will use a placeholder NO_DEFAULT to denote
            that no default value is provided.
    """

    def __init__(
        self,
        data_configuration: DataConfiguration,
        node: Node,
        name: str = "Input",
        default: Any = NO_DEFAULT,
    ):
        super().__init__(data_configuration, node, name)
        self.default = default
        self._value = default

    def duplicate_with_new_owner(
        self, new_owner: Node, new_name: str | None = None
    ) -> InputPort:
        new_port = super().duplicate_with_new_owner(new_owner, new_name)
        new_port.default = self.default  # type: ignore
        new_port.set_value(self.value)  # type: ignore
        return new_port  # type: ignore

    @property
    def is_required(self) -> bool:
        return self.default is NO_DEFAULT

    def clear_value(self):
        self._value = self.default


class OutputPort(Port):
    """Denotes an output port of a node."""

    def __init__(
        self, data_configuration: DataConfiguration, node: Node, name: str = "Output"
    ):
        super().__init__(data_configuration, node, name)


# endregion
# region: Node Properties
# A registry of all node types that have been defined. This is used to
# reconstruct nodes from their type identifier when loading a graph
# from a file.
NODE_REGISTRY: dict[str, type[Node]] = {}


class NodeState(Enum):
    """Denotes different states a node can be in.

    FIXED
        The node has some fixed behavior that never changes.
    UNINITIALIZED
        The node needs to be initialized before it can be used, e.g.
        by calling .setup().
    INITIALIZED
        The node is ready to be used and can be trained.
    TRAINED
        The node has been trained and is ready to be used for inference.
    """

    FIXED = 1
    UNINITIALIZED = 2
    INITIALIZED = 3
    TRAINED = 4


@dataclass
class NodeConfig:
    """Immutable constructor configuration for reconstructing a node.

    Args:
        node_identifier (str | None): The type identifier of the node. Used to
            reconstruct the node from the NODE_REGISTRY.
        node_id (int): The unique identifier of the node.
        mode (EvaluationPhase): The evaluation phase of the node.
        hyperparameters (dict[str, HyperParameter]): The hyperparameters of
            the node.
        other_args (dict[str, Any]): Any other arguments that were used to
            construct the node.
        state (NodeState): The state of the node.
        nested_graphs (dict[str, Graphs]): Any nested graphs that are part
            of this node.
    """

    node_identifier: str | None
    node_id: int
    mode: EvaluationPhase
    hyperparameters: dict[
        str, HyperParameter | list[HyperParameter] | tuple[HyperParameter, ...]
    ]
    other_args: dict[str, Any]
    state: NodeState
    nested_graphs: dict = field(default_factory=dict)


# endregion
# region: Main Node Class


class Node(ABC, Generic[TensorType]):
    """Base class for all nodes to create a graph.

    Args:
        name (str, optional): The name of the node. Defaults to "Node".
        state (NodeState, optional): The state of the node.
            Defaults to NodeState.FIXED.
        backend (type[Backend[TensorType]], optional): What backend the node
            should use for computations, parameters, etc. Defaults to Backend.
    """

    _node_id_counter = 0
    _tracking_phase: bool = False
    # Identifier for this node type. Used to reconstruct the node from
    # its configuration. If it is None, the class name will be used as the
    # identifier (but this is unsafe if it will be changed at some point).
    _type_identifier: str | None = None

    def __init__(
        self,
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs  # unused for now, but can be used in subclasses to
        # pass additional arguments
        self._name = name
        self._state = state
        self.backend = backend
        self.mode: EvaluationPhase = EvaluationPhase.ALWAYS

        self._input_ports, self._output_ports = self._build_ports(self.forward, self)

        self.node_id = Node._node_id_counter
        Node._node_id_counter += 1

    def __init_subclass__(cls) -> None:
        if cls._type_identifier is None:
            type_id = cls.__name__
        else:
            type_id = cls._type_identifier
        if type_id not in NODE_REGISTRY:
            NODE_REGISTRY[type_id] = cls
        return super().__init_subclass__()

    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name if self._name is not None else self.__class__.__name__
        return self.__class__.__name__

    @classmethod
    def set_tracking(cls, set_active: bool):
        cls._tracking_phase = set_active

    @classmethod
    def _build_ports(
        cls, func: Callable, owner: Node
    ) -> tuple[list[InputPort], list[OutputPort]]:
        """Automatically builds input and output ports for this node based
        on the signature of the forward function and the type hints of its
        parameters and return value.
        """
        call_sig = inspect.signature(func)
        type_hints = get_type_hints(func, include_extras=True)

        input_ports = []
        output_ports = []
        # Build input ports:
        for name, param in call_sig.parameters.items():
            hint = type_hints.get(name, param.annotation)
            config, _ = cls._unwrap_annotated(hint, owner)
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
        # Build output ports
        return_values = type_hints.get("return", inspect.Signature.empty)
        if return_values is None or return_values is inspect.Signature.empty:
            return input_ports, output_ports

        if get_origin(return_values) is tuple:
            outputs = list(get_args(return_values))
        else:
            outputs = [return_values]

        for i, output in enumerate(outputs):
            config, _ = cls._unwrap_annotated(output, owner)
            output_ports.append(OutputPort(config, node=owner, name=f"output_{i}"))

        return input_ports, output_ports

    @classmethod
    def get_dtype(cls, type_hint, backend: type[Backend[TensorType]]):
        if type_hint == Any:
            return type_hint
        if type_hint is not TensorType and isinstance(type_hint, type):
            if type_hint is not inspect.Signature.empty:
                return type_hint
        if backend == Backend:
            return Any
        return backend.default_dtype

    @classmethod
    def _unwrap_annotated(cls, type_hint, owner):
        """Return (base_type, config)."""

        if get_origin(type_hint) in [Optional, Union]:
            type_hint = get_args(type_hint)[0]
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
            config.set_dtype(base)
            return config, True
        empty_conf = DataConfiguration.empty()
        empty_conf.set_dtype(type_hint)
        return empty_conf, False

    def copy_data_configs(self):
        copy_memo = {}
        dynamic_data_configs = {}
        for port in self.input_ports + self.output_ports:
            dynamic_data_configs[port] = self.copy_data_config_of_port(port, copy_memo)
        return dynamic_data_configs

    def copy_data_config_of_port(self, port: Port, copy_memo):
        copied_config = deepcopy(port.data_configuration, copy_memo)
        # Now write the concrete type into the port:
        copied_config.set_dtype(
            self.get_dtype(port.data_configuration.dtype, self.backend)
        )
        return copied_config

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
        """Evaluates the node in the graph run. This will read all the
        values from the ports connected to the input port of this node,
        pass them to the call-method of this node and write the outputs
        into the output ports.
        """
        input_data = [in_port.value for in_port in self.input_ports]
        output_data = self(*input_data)
        if len(self.output_ports) == 1:
            self.output_ports[0].set_value(output_data)
        elif len(self.output_ports) > 1:
            for i, out_port in enumerate(self.output_ports):
                out_port.set_value(output_data[i])

    def __call__(self, *args, **kwargs) -> Any:
        """Apply this node to the provided inputs."""
        if self._tracking_phase:
            return self._track(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Implements the main functionality of this node. This method
        needs to be implemented in the subclasses
        """
        raise NotImplementedError(
            "The default node can not be called, "
            "this method needs to be overwritten in the subclasses."
        )

    def _track(self, *args, **kwargs):
        """Track the data passed through this node. This can be used to implement
        graph tracking for debugging or visualization purposes.
        """
        from qewton.graphs.graphs import TrackingObject

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
        port_config_was_updated = port_config.update_config(config_dict)
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

    def get_input_port(self, name: str) -> InputPort:
        """Returns the input port with the given name

        Args:
            name (str): The name we are looking for.

        Raises:
            ValueError: If no input port with the given name is found.

        Returns:
            InputPort: The port with the corresponding name.
        """
        for port in self.input_ports:
            if port.name == name:
                return port
        raise ValueError(f"No input port with name {name} found in node {self.name}.")

    def get_output_port(self, name: str) -> OutputPort:
        """Returns the output port with the given name.

        Args:
            name (str): The name we are looking for.

        Raises:
            ValueError: If no output port with the given name is found.

        Returns:
            OutputPort: The port with the corresponding name.
        """
        for port in self.output_ports:
            if port.name == name:
                return port
        raise ValueError(f"No output port with name {name} found in node {self.name}.")

    def copy(self):
        """Creates a copy of this node, with the same inner operations, parameters
        etc., but with new input and output ports.

        Returns
            CopiedNode: A copy of this node.
        """
        from .control_nodes.graph_node import CopiedNode

        return CopiedNode(self)

    def config_dict(self) -> NodeConfig:
        """Returns a configuration object that can be used to reconstruct
        this node. By default we just return the hyperparameters and other
        arguments, but this can be overridden in subclasses to include
        additional information.

        Returns:
            NodeConfig: The configuration object.
        """
        if self._type_identifier is None:
            type_id = self.__class__.__name__
        else:
            type_id = self._type_identifier
        # Read needed information from the constructor of this node to
        # reconstruct it later
        other_args = {}
        hyperparameters = {}
        for name in inspect.signature(self.__class__.__init__).parameters:
            if name in ["self", "kwargs", "args"]:
                continue

            class_atri = getattr(self, name)
            # Hyperparameters are stored in a separate dictionary,
            # so we can easily check which parameters are also shared between
            # different nodes.
            if isinstance(class_atri, HyperParameter):
                hyperparameters[name] = class_atri
            elif isinstance(class_atri, (list, tuple)) and all(
                isinstance(item, HyperParameter) for item in class_atri
            ):
                hyperparameters[name] = class_atri
            else:
                other_args[name] = class_atri

        return NodeConfig(
            node_identifier=type_id,
            node_id=self.node_id,
            mode=self.mode,
            hyperparameters=hyperparameters,
            other_args=other_args,
            state=self.state,
        )

    @classmethod
    def load_from_config(cls, config: NodeConfig) -> Node:
        """Reconstructs a node from a configuration object. By default we just
        use the hyperparameters and other arguments, but this can be overridden
        in subclasses to include additional information.

        Args:
            config (NodeConfig): The configuration object.

        Returns:
            Node: The reconstructed node.
        """
        if config.node_identifier is None:
            raise ValueError(
                "Cannot reconstruct node from config, "
                "node_identifier is None. This is required to reconstruct the node."
            )
        node_class = NODE_REGISTRY.get(config.node_identifier)
        if node_class is None:
            raise ValueError(
                f"Cannot reconstruct node from config, "
                f"node_identifier {config.node_identifier} not found in NODE_REGISTRY."
            )

        # Build the input arguments for the constructor of the node class.
        init_inputs = {}
        for name, param in inspect.signature(node_class.__init__).parameters.items():
            if name in ["self", "kwargs", "args"]:
                continue
            if name in config.hyperparameters:
                init_inputs[name] = config.hyperparameters[name]
            elif name in config.other_args:
                init_inputs[name] = config.other_args[name]
            else:
                init_inputs[name] = param

        node: Node = node_class(**init_inputs)
        node.set_mode(config.mode)
        node.set_state(config.state)
        node.node_id = config.node_id
        return node


# endregion
