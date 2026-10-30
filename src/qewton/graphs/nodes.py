"""Nodes, the building blocks of a :class:`~qewton.graphs.graphs.Graph`,
and their input and output ports."""

from __future__ import annotations
from abc import ABC
from copy import deepcopy
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
from qewton.config.variables import Variable
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.trainable_parameters import (
    _TrainableParameterBase,
    TrainableParameters,
)
from qewton.config.saving.saving import Serializable, Serializer
from qewton.config.saving.loading import Deserializer
from qewton.config.saving.schema_keys import SavingKeys

# region: Ports


class NO_DEFAULT(Serializable):
    """Sentinel denoting that an input port has no default value.

    The class itself is the sentinel; it is never instantiated.
    """

    @classmethod
    def save(cls, serializer: Serializer) -> None:
        """Serialize a reference to the sentinel class.

        Args:
            serializer: The serializer that collects the data.
        """
        node_config = {
            SavingKeys.KEY_TYPE: SavingKeys.KEY_SERIALIZABLE,
            SavingKeys.KEY_CLASS: cls.__name__,
            SavingKeys.KEY_MODULE: cls.__module__,
        }
        serializer.set_serialization_data(id(cls), node_config)

    @classmethod
    def construct_new_object(cls, serializer: Deserializer, data_config: dict) -> Any:
        """Return the sentinel class itself.

        Args:
            serializer: The deserializer managing the loading process.
            data_config: The serialized data of the sentinel.

        Returns:
            type[NO_DEFAULT]: This class.
        """
        return cls


class Port(Serializable):
    """Input or output connection of a node.

    Ports are connected by edges in a :class:`~qewton.graphs.graphs.Graph`.
    Each port has a static data configuration describing the data passing
    through it. A graph unifies the configurations of connected ports and
    keeps the result per port in its own `dynamic_data_configs`, see
    :meth:`get_data_configuration`.

    Args:
        data_configuration: The static data configuration of this port.
        node: The node owning this port.
        name: The name of this port.

    Attributes:
        node (Node): The node owning this port.
        name (str): The name of this port.
        value_persists (bool): Whether the value of this port is kept when
            saving. False by default, see :meth:`make_persistent`.
        data_configuration (DataConfiguration): The static data configuration
            of this port.
        value (Any): The value currently stored in this port.
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
        self.value_persists = False

    @property
    def data_configuration(self) -> DataConfiguration:
        return self._static_data_configuration

    def get_data_configuration(self, graph) -> DataConfiguration | None:
        """Return the data configuration of this port inside `graph`.

        Args:
            graph (Graph): The graph whose unified configuration is returned.

        Returns:
            DataConfiguration or None: The configuration of this port in
                `graph`, or None if the owning node is unknown to `graph`.
        """
        if self.node in graph.dynamic_data_configs:
            return graph.dynamic_data_configs[self.node][self]
        return None

    def update_static_data_configuration(self, new_config: DataConfiguration):
        """Replace the static data configuration of this port.

        Args:
            new_config: The new static data configuration.
        """
        self._static_data_configuration = new_config

    def duplicate_with_new_owner(
        self, new_owner: Node, new_name: str | None = None
    ) -> Port:
        """Create a port of the same type and configuration for another node.

        Args:
            new_owner: The node owning the new port.
            new_name: The name of the new port. None (default) keeps the name
                of this port.

        Returns:
            Port: The new port.
        """
        return type(self)(
            data_configuration=self.data_configuration,
            node=new_owner,
            name=new_name if new_name is not None else self.name,
        )

    def reset_value(self):
        """Set the stored value to None."""
        self._value = None

    def set_value(self, value):
        """Store a value in this port.

        Connected ports read this value when the graph runs.

        Args:
            value (Any): The value to store.
        """
        self._value = value

    @property
    def value(self):
        return self._value

    def make_persistent(self):
        """Keep the value of this port when saving."""
        self.value_persists = True

    def save(self, serializer: Serializer) -> None:
        """Serialize this port, including its value only if it persists.

        Args:
            serializer: The serializer that collects the data.
        """
        current_value = self.value
        if not self.value_persists:
            self._value = None
        super().save(serializer)
        self._value = current_value  # Restore the value after saving


class InputPort(Port):
    """Input port of a node.

    An input port without an incoming edge holds its default value when
    the graph runs.

    Args:
        data_configuration: The static data configuration of this port.
        node: The node owning this port.
        name: The name of this port.
        default: The default value of this port. :class:`NO_DEFAULT` (default)
            marks the port as required.

    Attributes:
        default (Any): The default value of this port.
        is_required (bool): Whether this port has no default value and must be
            connected.
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
        """Create an input port with the same configuration, default and
        value for another node.

        Args:
            new_owner: The node owning the new port.
            new_name: The name of the new port. None (default) keeps the name
                of this port.

        Returns:
            InputPort: The new port.
        """
        new_port = super().duplicate_with_new_owner(new_owner, new_name)
        new_port.default = self.default  # type: ignore
        new_port.set_value(self.value)  # type: ignore
        return new_port  # type: ignore

    @property
    def is_required(self) -> bool:
        return self.default is NO_DEFAULT

    def clear_value(self):
        """Reset the stored value to the default value."""
        self._value = self.default


class OutputPort(Port):
    """Output port of a node.

    Args:
        data_configuration: The static data configuration of this port.
        node: The node owning this port.
        name: The name of this port.
    """

    def __init__(
        self, data_configuration: DataConfiguration, node: Node, name: str = "Output"
    ):
        super().__init__(data_configuration, node, name)


# endregion
# region: Node Properties


class NodeState(Enum):
    """State of a node.

    Attributes:
        FIXED: The node has some fixed behavior that never changes.
        UNINITIALIZED: The node needs to be initialized before it can be
            used, e.g. by calling .setup().
        INITIALIZED: The node is ready to be used and can be trained.
        TRAINED: The node has been trained and is ready to be used for
            inference.
    """

    FIXED = 1
    UNINITIALIZED = 2
    INITIALIZED = 3
    TRAINED = 4


# endregion
# region: Main Node Class


class Node(ABC, Serializable, Generic[TensorType]):
    """Base class for all nodes of a graph.

    The input and output ports are built from the signature of
    :meth:`forward`: every parameter becomes an input port (a parameter
    default becomes the port default), every element of the return type
    (or of a returned tuple) an output port named ``output_<i>``. A
    :class:`~qewton.config.data_configurations.DataConfiguration`, or a
    callable taking the node and returning one, given via
    ``typing.Annotated`` sets the static data configuration of a port.

    Args:
        name: The name of the node. None (default) uses the class name.
        state: The initial state of the node.
        backend: The backend used for computations and parameters.
        **kwargs: Ignored; accepted so subclasses can pass on extra arguments.

    Attributes:
        backend (type[Backend]): The backend used for computations and
            parameters.
        mode (EvaluationPhase): The current evaluation phase, see
            :meth:`set_mode`.
        node_id (int): Unique ID of this node.
        name (str): The name of the node.
        state (NodeState): The current state of the node.
        input_ports (list[InputPort]): The input ports, in the order of the
            parameters of :meth:`forward`.
        output_ports (list[OutputPort]): The output ports, in the order of the
            return values of :meth:`forward`.
        hyperparameters (list[HyperParameter]): The tunable hyperparameters of
            this node. By default all
            :class:`~qewton.optim.parameters.hyperparameter_base.HyperParameter`
            instance attributes.
        trainable_parameters (TrainableParameters): The trainable parameters of
            this node. Empty by default.
        memory_consumption (int): The memory consumption of this node in bytes.
            0 by default.

    Notes:
        Subclasses must implement :meth:`forward`. They may override
        :meth:`setup` to create their underlying algorithm, the properties
        `trainable_parameters`, `hyperparameters` and `memory_consumption`,
        :meth:`to`, :meth:`reset`, and :meth:`update_data_configs` if their
        ports depend on each other in a non-default way.
    """

    _node_id_counter = 0
    _tracking_phase: bool = False

    def __init__(
        self,
        name: str | None = None,
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs
        self._name = name
        self._state = state
        self.backend = backend
        self.mode: EvaluationPhase = EvaluationPhase.ALWAYS

        self._input_ports, self._output_ports = self._build_ports(self.forward, self)

        self.node_id = Node._node_id_counter
        Node._node_id_counter += 1

    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name if self._name is not None else self.__class__.__name__
        return self.__class__.__name__

    @name.setter
    def name(self, value: str):
        self._name = value

    @classmethod
    def set_tracking(cls, set_active: bool):
        """Switch graph tracking on or off for all nodes.

        While tracking is active, calling a node records it in the tracked
        graph instead of running :meth:`forward`, see
        :meth:`~qewton.graphs.graphs.Graph.tracker`.

        Args:
            set_active: Whether tracking is active.
        """
        cls._tracking_phase = set_active

    @classmethod
    def _build_ports(
        cls, func: Callable, owner: Node
    ) -> tuple[list[InputPort], list[OutputPort]]:
        """Build the input and output ports of `owner` from the signature
        and type hints of `func`."""
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
        """Resolve the dtype of a port from its type hint.

        Args:
            type_hint (Any): The type hint of the port.
            backend: The backend of the node.

        Returns:
            Any: `type_hint` if it is ``Any`` or a concrete type other than
                ``TensorType``; otherwise ``Any`` for the generic
                :class:`~qewton.backends.Backend` and the default dtype of
                `backend` for a concrete backend.
        """
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
        """Return (data configuration, whether the hint was Annotated)."""

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
        """Copy the static data configurations of all ports.

        Returns:
            dict[Port, DataConfiguration]: A copy of the configuration of each
                port, with its dtype resolved by :meth:`get_dtype`.
        """
        copy_memo = {}
        dynamic_data_configs = {}
        for port in self.input_ports + self.output_ports:
            dynamic_data_configs[port] = self.copy_data_config_of_port(port, copy_memo)
        return dynamic_data_configs

    def copy_data_config_of_port(self, port: Port, copy_memo):
        """Copy the static data configuration of a single port.

        Args:
            port: The port whose configuration is copied.
            copy_memo (dict): The memo passed to ``deepcopy``, so
                configurations copied together share their objects.

        Returns:
            DataConfiguration: The copy, with its dtype resolved by
                :meth:`get_dtype`.
        """
        copied_config = deepcopy(port.data_configuration, copy_memo)
        copied_config.set_dtype(
            self.get_dtype(port.data_configuration.dtype, self.backend)
        )
        return copied_config

    def setup(self) -> None:
        """Create the underlying algorithm instance, e.g. a neural network.

        Called by :meth:`~qewton.graphs.graphs.Graph.setup`. Does nothing
        by default.
        """

    @property
    def input_ports(self) -> list[InputPort]:
        return self._input_ports

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    def run(self) -> None:
        """Evaluate the node on the values of its input ports.

        The values of the input ports are passed to the node call and the
        results are written to the output ports.
        """
        input_data = [in_port.value for in_port in self.input_ports]
        output_data = self(*input_data)
        if len(self.output_ports) == 1:
            self.output_ports[0].set_value(output_data)
        elif len(self.output_ports) > 1:
            for i, out_port in enumerate(self.output_ports):
                out_port.set_value(output_data[i])

    def __call__(self, *args, **kwargs) -> Any:
        """Apply this node to the inputs.

        Runs :meth:`forward`, or records the call in the tracked graph
        while tracking is active.
        """
        if self._tracking_phase:
            return self._track(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Compute the outputs of this node from its inputs.

        The signature and type hints define the ports of the node.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError(
            "The default node can not be called, "
            "this method needs to be overwritten in the subclasses."
        )

    def _track(self, *args, **kwargs):
        """Connect this node in the tracked graph and return tracking
        objects for its outputs."""
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
        # Scans all instance attributes; subclasses may override this with
        # an explicit list.
        return [v for v in vars(self).values() if isinstance(v, HyperParameter)]

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        return TrainableParameters.create_empty(self.node_id)

    @property
    def memory_consumption(self) -> int:
        return 0

    @property
    def _trainable_parameters(self) -> _TrainableParameterBase:
        """Trainable parameters as collected by a graph; empty while the
        node is FIXED."""
        return (
            self.trainable_parameters
            if self.state != NodeState.FIXED
            else TrainableParameters.create_empty(self.node_id)
        )

    def to(self, device):
        """Move the data stored in this node to `device`.

        Does nothing by default.

        Args:
            device (Device or str): The target device, e.g. a CPU or GPU.
        """

    def set_mode(self, new_mode: EvaluationPhase):
        """Set the current evaluation phase.

        Some nodes behave differently depending on the phase, e.g.
        dropout is disabled during validation.

        Args:
            new_mode: The new evaluation phase.
        """
        self.mode = new_mode

    def reset(self):
        """Reset the node. Does nothing by default."""

    def set_state(self, new_state: NodeState):
        """Set the current state of this node.

        Args:
            new_state: The new state.
        """
        self._state = new_state

    @property
    def state(self):
        return self._state

    def fix_node_state(self) -> None:
        """Set the state to FIXED, so the node is neither trained nor
        recreated.

        An uninitialized node is not fixed; a UserWarning is issued
        instead.
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
        """Apply `config_dict` to the configuration of `updated_port` and of
        all other ports of this node.

        Args:
            updated_port (Port): The port whose configuration is updated.
            config_dict (dict): Maps axes to their replacements, as returned by
                :meth:`~qewton.config.data_configurations.base.DataConfiguration.unify_with`.
            dynamic_configs (dict[Port, DataConfiguration]): The configurations
                of the ports of this node inside a graph; updated in place.

        Returns:
            set[Port]: The ports whose configuration changed, excluding
                `updated_port` if it is an input port. Empty if the
                configuration of `updated_port` did not change.
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

    def get_input_port(self, name: str | Variable) -> InputPort:
        """Returns the input port matching the given name.

        Args:
            name (str | Variable): The name we are looking for, either
                as a string or as a Variable, in which case the variable's
                name is used.

        Raises:
            ValueError: If no input port matching the given name is found
                or the passed variable does not have a name set.

        Returns:
            InputPort: The port with this name.

        Raises:
            ValueError: If no input port has this name.
        """
        if isinstance(name, Variable):
            if name.name is None:
                raise ValueError(f"The passed variable {name} does not have a name set")
            name = name.name
        for port in self.input_ports:
            if port.name == name:
                return port
        raise ValueError(f"No input port with name {name} found in node {self.name}.")

    def get_output_port(self, name: str | Variable) -> OutputPort:
        """Returns the output port matching the given name.

        Args:
            name (str | Variable): The name we are looking for, either
                as a string or as a Variable, in which case the variable's
                name is used.

        Raises:
            ValueError: If no output port matching the given name is found
                or the passed variable does not have a name set.

        Returns:
            OutputPort: The port matching the given name.
        """
        if isinstance(name, Variable):
            if name.name is None:
                raise ValueError(f"The passed variable {name} does not have a name set")
            name = name.name
        for port in self.output_ports:
            if port.name == name:
                return port
        raise ValueError(f"No output port with name {name} found in node {self.name}.")

    def copy(self):
        """Create a node sharing this node's operations and parameters but
        with its own input and output ports.

        Returns:
            CopiedNode: The copy, see
                :class:`~qewton.graphs.control_nodes.graph_node.CopiedNode`.
        """
        from .control_nodes.graph_node import CopiedNode

        return CopiedNode(self)

    def load(self, serializer: Deserializer, data_config: dict) -> None:
        """Load the node state and give it a node ID unique in this session.

        Args:
            serializer: The deserializer managing the loading process.
            data_config: The serialized data of this node.
        """
        super().load(serializer, data_config)
        # Offset the loaded ID by the counter so it cannot clash with nodes
        # created in this session.
        self.node_id += Node._node_id_counter
        serializer.update_reference_node(self)


# endregion


class GraphAwareNode(Node[TensorType]):
    """Base class for nodes whose setup needs the owning graph.

    :meth:`~qewton.graphs.graphs.Graph.setup` passes itself to the `setup`
    of these nodes, so they can read the unified data configurations of
    their ports from the graph's `dynamic_data_configs`.

    Notes:
        Subclasses must implement :meth:`setup` with the signature
        ``setup(self, graph)``, in addition to :meth:`Node.forward`.
    """

    def setup(self, graph) -> None:
        """Set up the node using the graph it is part of.

        Args:
            graph (Graph): The graph being set up.

        Raises:
            NotImplementedError: Always; subclasses must implement this method.
        """
        raise NotImplementedError(
            "Subclasses of GraphAwareNode must implement setup(self, graph)."
        )
