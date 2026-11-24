"""Nodes, the building blocks of a :class:`~qewton.graphs.graphs.Graph`,
and their input and output ports."""

from __future__ import annotations
from abc import ABC
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
import functools
import inspect
import operator
import types
import warnings

from qewton.config.data_configurations import DataConfiguration, DynamicDataConfiguration
from qewton.config.dtypes import element_dtype_in
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
    Each port declares a static data configuration describing the data
    passing through it. A port holds no value: a graph keeps the values and
    the matched configurations of the ports it contains, see
    :meth:`~qewton.graphs.graphs.Graph.value_of` and
    :meth:`get_data_configuration`.

    Args:
        data_configuration: The static data configuration of this port.
        node: The node owning this port.
        name: The name of this port.

    Attributes:
        node (Node): The node owning this port.
        name (str): The name of this port.
        data_configuration (DataConfiguration): The static data configuration
            of this port.
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

    @property
    def data_configuration(self) -> DataConfiguration:
        return self._static_data_configuration

    def __str__(self) -> str:
        return f"{self.node.name}.{self.name}"

    def get_data_configuration(self, graph, within=None) -> DataConfiguration | None:
        """Return the data configuration of this port inside `graph`.

        Args:
            graph (Graph): The graph whose matched configuration is returned.
            within (tuple[GraphNode, ...], optional): The GraphNodes, from
                the outside in, leading to the node of this port; needed only
                where the node occurs at several places of `graph`, see
                :meth:`~qewton.graphs.graphs.Graph.config_of`.

        Returns:
            DataConfiguration or None: The configuration of this port in
                `graph`, or None if the owning node is not part of `graph`.
        """
        return graph.config_of(self, within)

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

    def duplicate_with_new_owner(
        self, new_owner: Node, new_name: str | None = None
    ) -> InputPort:
        """Create an input port with the same configuration and default for
        another node.

        Args:
            new_owner: The node owning the new port.
            new_name: The name of the new port. None (default) keeps the name
                of this port.

        Returns:
            InputPort: The new port.
        """
        new_port = super().duplicate_with_new_owner(new_owner, new_name)
        new_port.default = self.default  # type: ignore
        return new_port  # type: ignore

    @property
    def is_required(self) -> bool:
        return self.default is NO_DEFAULT


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
        inputs, outputs = cls._declarations(func, owner)
        input_ports = [
            InputPort(
                config,
                node=owner,
                name=param.name,
                default=NO_DEFAULT if param.default is inspect.Parameter.empty else param.default,
            )
            for param, config, _ in inputs
        ]
        output_ports = [
            OutputPort(config, node=owner, name=f"output_{i}")
            for i, (config, _) in enumerate(outputs)
        ]
        return input_ports, output_ports

    @classmethod
    def _declarations(cls, func: Callable, owner: Node) -> tuple[list, list]:
        """The configurations `func` declares: (parameter, configuration,
        whether annotated) per input and (configuration, whether annotated)
        per output."""
        type_hints = get_type_hints(func, include_extras=True)
        inputs = []
        for name, param in inspect.signature(func).parameters.items():
            config, annotated = cls._unwrap_annotated(type_hints.get(name, param.annotation), owner)
            inputs.append((param, config, annotated))
        return_values = type_hints.get("return", inspect.Signature.empty)
        if return_values is None or return_values is inspect.Signature.empty:
            return inputs, []
        outputs = (
            list(get_args(return_values)) if get_origin(return_values) is tuple else [return_values]
        )
        return inputs, [cls._unwrap_annotated(output, owner) for output in outputs]

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
        if get_origin(type_hint) in (Union, types.UnionType):
            members = []
            for member in get_args(type_hint):
                resolved = cls.get_dtype(member, backend)
                if resolved not in members:
                    members.append(resolved)
            return functools.reduce(operator.or_, members)
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
            config.element_dtype = element_dtype_in(meta)
            return config, True
        empty_conf = DataConfiguration.empty()
        empty_conf.set_dtype(type_hint)
        return empty_conf, False

    def setup(self, graph=None) -> None:
        """Make the node ready to run: build it (:meth:`build`) unless it is
        built already, i.e. unless its state is other than
        `NodeState.UNINITIALIZED`.

        Called by :meth:`~qewton.graphs.graphs.Graph.setup`. Belongs to the
        library: subclasses override :meth:`build`, not this method.

        Args:
            graph (Graph, optional): The graph being set up; passed on to the
                `build` of nodes that need it (:class:`GraphAwareNode`).

        Raises:
            RuntimeError: If a size of the node is still open (see
                :meth:`require_known_sizes`).
        """
        if self.state != NodeState.UNINITIALIZED:
            return
        self.require_known_sizes()
        self._build_in(graph)
        if self.state == NodeState.UNINITIALIZED:
            self.set_state(NodeState.INITIALIZED)

    def require_known_sizes(self) -> None:
        """Make sure no hyperparameter waits to be inferred, so that the node
        can be built.

        Raises:
            RuntimeError: Naming the hyperparameters that are still open.
        """
        if self.open_hyperparameters:
            names = ", ".join(repr(hp.name) for hp in self.open_hyperparameters)
            raise RuntimeError(
                f"{self.name} can not be built: {names} are inferred from the "
                "connected data. Connect it in a graph and set the graph up, or "
                "give the sizes."
            )

    def build(self) -> None:
        """Create what the node computes with, e.g. the weights or the inner
        graph of a network, from the current hyperparameter values.

        Called by :meth:`setup` for a node that is not built yet; a tuner
        makes a node unbuilt again (:meth:`reset`) before trying other
        hyperparameter values. Does nothing by default.
        """

    def _build_in(self, graph) -> None:
        """Build this node as part of `graph`."""
        self.build()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "setup" in cls.__dict__ and not cls.__dict__.get("_setup_of_library", False):
            raise TypeError(
                f"{cls.__name__} overrides setup(); override build() instead, "
                "setup() belongs to the library."
            )

    @property
    def input_ports(self) -> list[InputPort]:
        return self._input_ports

    @property
    def output_ports(self) -> list[OutputPort]:
        return self._output_ports

    def relate(self, *inputs: DynamicDataConfiguration):
        """The configuration of each output, from those of the inputs at one
        place of a graph, for relations the annotations of :meth:`forward`
        can not express: an operation that changes dims by their index
        (e.g. ``return x.squeeze(self.dim)``, see the shape operations of
        :class:`~qewton.config.data_configurations.DataConfiguration`).

        The inputs are declared by the annotations and already merged with
        what is connected, as known so far (resolved, see
        :class:`~qewton.config.matching.Matching`); they belong to the
        matching of the graph, so what shape operations fix goes there. The
        returned configurations are merged with the annotated ones of the
        outputs. A shape operation that needs more of
        the inputs' structure than is known raises
        :class:`~qewton.config.errors.StructureUnknown`; the graph calls
        `relate` again once the inputs change.

        Args:
            *inputs: The configurations of the input ports, in order.

        Returns:
            DataConfiguration or tuple or None: The configuration of the
                output, a tuple for several outputs, or None if the
                annotations say everything (default).
        """
        return None

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
        objects for its outputs. A node called again in the same graph is
        connected as a copy sharing its weights (:meth:`copy`)."""
        from qewton.graphs.graphs import TrackingObject

        graph = TrackingObject.current_graph_tracked
        if graph is not None:
            if self in graph.nodes:
                return self.copy()._track(*args, **kwargs)
            graph.add_node(self, check_warning=False)

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
    def open_hyperparameters(self) -> list[HyperParameter]:
        """The hyperparameters still waiting to be inferred from the
        connected data (see :attr:`HyperParameter.is_open`); the node can
        not be built before they are."""
        return [hp for hp in self.hyperparameters if hp.is_open]

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
        """Make the node unbuilt, so that the next :meth:`setup` builds it
        again (e.g. for other hyperparameter values); a FIXED node stays as
        it is."""
        if self.state != NodeState.FIXED:
            self.set_state(NodeState.UNINITIALIZED)

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

    def upgrade(self, version: int) -> None:
        """See :meth:`~qewton.config.saving.saving.Serializable.upgrade`.

        Files of version 1 hold the port declarations of their time: the
        ports of this node annotated in :meth:`forward` are declared anew.

        Args:
            version: The format version of the file.
        """
        if version >= 2:
            return
        inputs, outputs = self._declarations(self.forward, self)
        declared = [(config, annotated) for _, config, annotated in inputs] + outputs
        for port, (config, annotated) in zip(self.input_ports + self.output_ports, declared):
            if annotated and port.node is self:
                port.update_static_data_configuration(config)

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
    """Base class for nodes whose build needs the owning graph.

    :meth:`~qewton.graphs.graphs.Graph.setup` passes itself on, so the
    `build` of these nodes gets the graph, e.g. to read the configurations of
    their ports or to run the graph up to them.

    Notes:
        Subclasses implement :meth:`build` with the signature
        ``build(self, graph)``, in addition to :meth:`Node.forward`.
    """

    def build(self, graph) -> None:
        """Create what the node computes with, using the graph it is part of.

        Args:
            graph (Graph): The graph being set up.
        """

    def _build_in(self, graph) -> None:
        self.build(graph)
