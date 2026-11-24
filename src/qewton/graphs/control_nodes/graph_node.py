from __future__ import annotations
from typing import Callable
import inspect
from copy import deepcopy

from qewton.graphs.nodes import InputPort, Node, NodeState, OutputPort, Port
from qewton.graphs.graphs import Graph
from qewton.backends import Backend, TensorType
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase


class GraphNode(Node[TensorType]):
    """
    A node that encapsulates an entire graph.

    The inner graph is executed when the `run` method of this node is called.
    Input and output ports of the inner graph are mapped to the input and output
    ports of this `GraphNode`, allowing a complete graph to be used as a single
    node within a larger graph structure.

    Args:
        graph (Graph | None): The graph to be encapsulated by this node; None
            for a node that builds it later (in `build`, e.g. once sizes are
            inferred). Its `forward` then declares all ports.
        input_ports (list[InputPort] | dict[InputPort, list[InputPort]]):
            Defines how the input ports of this GraphNode map to the input ports of
            the inner graph.
            If a list, new input ports for this node are created, and inner graph
            inputs are mapped by order.
            If a dict, keys are this GraphNode's input ports, values are lists of
            corresponding inner graph input ports.
        output_ports (list[OutputPort] | dict[OutputPort, Port]):
            Defines how the output ports of this GraphNode map to the output ports of
            the inner graph.
            If a list, new output ports for this node are created, and inner graph
            outputs are mapped by order.
            If a dict, keys are this GraphNode's output ports, values are
            corresponding inner graph output ports.
        name (str, optional): The name of this node. Defaults to "GraphNode".
        backend (type[Backend[TensorType]], optional): The backend type for tensor
            operations. Defaults to Backend.

    Attributes:
        inner_graph (Graph): The encapsulated graph.
    """

    def __init__(
        self,
        graph: Graph | None,
        input_ports: list[InputPort] | dict[InputPort, list[InputPort]],
        output_ports: list[OutputPort] | dict[OutputPort, Port],
        name: str = "GraphNode",
        backend: type[Backend[TensorType]] = Backend,
        **kwargs,
    ) -> None:
        super().__init__(name=name, backend=backend, **kwargs)

        if graph is None:
            self._without_inner_graph()
            return
        self._graph = graph

        self._input_ports = []

        self.configs_defined_in_forward = self._configs_were_defined_in_forward()
        in_forward_ports, out_forward_ports = self._build_ports(self.forward, self)
        for i, p in enumerate(input_ports):
            if isinstance(input_ports, dict):
                self._input_ports.append(p)
                for inner_port in input_ports[p]:
                    self._graph.connect_from_outside_of_graph(p, inner_port)
            else:
                if self.configs_defined_in_forward:
                    port_config = in_forward_ports[i].data_configuration
                else:
                    # Its own copy: inner ports may share one declared object.
                    port_config = deepcopy(p.data_configuration)
                self._input_ports.append(
                    InputPort(
                        data_configuration=port_config,
                        node=self,
                        name=p.name,
                    )
                )
                self._graph.connect_from_outside_of_graph(self._input_ports[-1], p)

        self._output_ports = []
        for i, p in enumerate(output_ports):
            if isinstance(output_ports, dict):
                self._output_ports.append(p)
                if output_ports[p].node == self:
                    self._graph.add_skip_connection(output_ports[p], p)
                else:
                    self._graph.connect_to_outside_of_graph(
                        output_ports[p], p  # type: ignore
                    )
            else:
                if self.configs_defined_in_forward:
                    port_config = out_forward_ports[i].data_configuration
                else:
                    # Its own copy: inner ports may share one declared object.
                    port_config = deepcopy(p.data_configuration)
                self._output_ports.append(
                    OutputPort(
                        data_configuration=port_config,
                        node=self,
                        name=p.name,
                    )
                )
                self._graph.connect_to_outside_of_graph(p, self._output_ports[-1])

    def upgrade(self, version: int) -> None:
        """See :meth:`Node.upgrade`; the inner nodes are upgraded first, and
        ports not declared in :meth:`forward` copy the configuration of the
        inner port they lead to.

        Args:
            version: The format version of the file.
        """
        if version >= 2:
            return
        for node in self._graph.nodes:
            node.upgrade(version)
        self.configs_defined_in_forward = self._configs_were_defined_in_forward()
        super().upgrade(version)
        if self.configs_defined_in_forward:
            return
        for edge in self._graph.edges_from_outside:
            if edge.from_port.node is self:
                edge.from_port.update_static_data_configuration(
                    deepcopy(edge.to_port.data_configuration)
                )
        for edge in self._graph.edges_to_outside:
            if edge.to_port.node is self:
                edge.to_port.update_static_data_configuration(
                    deepcopy(edge.from_port.data_configuration)
                )

    def _without_inner_graph(self) -> None:
        """Keep the ports its `forward` declares (see `Node`) with an empty
        inner graph, which `build` replaces."""
        if not self._configs_were_defined_in_forward():
            raise ValueError(
                f"{self.name} has no inner graph yet, so its forward has to declare "
                "all its ports."
            )
        self._graph = Graph()
        self.configs_defined_in_forward = True

    def update_inner_input_ports(
        self, new_input_ports: list[InputPort] | dict[InputPort, list[InputPort]]
    ):
        """
        Updates the connections for the inner graph's input ports.

        This method is typically used during graph setup establish how external inputs
        connect to the encapsulated graph.

        Args:
            new_input_ports (list[InputPort] | dict[InputPort, list[InputPort]]): The
                new input port configuration.
        """
        for i, p in enumerate(new_input_ports):
            if isinstance(new_input_ports, dict):
                for inner_port in new_input_ports[p]:
                    self._graph.connect_from_outside_of_graph(p, inner_port)
            else:
                self._graph.connect_from_outside_of_graph(self._input_ports[i], p)

    def update_inner_output_ports(
        self, new_output_ports: list[OutputPort] | dict[OutputPort, OutputPort]
    ):
        """
        Updates the connections for the inner graph's output ports.

        Similar to `update_inner_input_ports`, this method allows changing
        how the encapsulated graph's outputs are exposed as this GraphNode's outputs.

        Args:
            new_output_ports (list[OutputPort] | dict[OutputPort, OutputPort]): The new
                output port configuration.
        """
        for i, p in enumerate(new_output_ports):
            if isinstance(new_output_ports, dict):
                self._graph.connect_to_outside_of_graph(new_output_ports[p], p)
            else:
                self._graph.connect_to_outside_of_graph(p, self._output_ports[i])

    @property
    def inner_graph(self):
        return self._graph

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        """
        Collects and returns all hyperparameters from the nodes within the encapsulated
        graph.

        Returns:
            list[HyperParameter]: A list of all hyperparameters found in the inner graph.
        """
        hp_list = []
        for node in self._graph.nodes:
            hp_list.extend(node.hyperparameters)
        return hp_list

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        """
        Collects and returns all trainable parameters from the nodes within the
        encapsulated graph.

        Returns:
            _TrainableParameterBase: A collection of trainable parameters from the inner
                graph.
        """
        return self._graph.collect_trainable_parameters()

    @property
    def _trainable_parameters(self) -> _TrainableParameterBase:
        """
        Internal method to return trainable parameters, delegating to the graph's
        collection method.

        Returns:
            _TrainableParameterBase: A collection of trainable parameters.
        """
        return self.trainable_parameters

    def forward(self, *args, **kwargs):
        """Evaluate the inner graph on the inputs.

        Subclasses override this method only to declare the data
        configurations of their ports with ``typing.Annotated``; the body
        stays ``return super().forward(...)``. Inside a graph, the steps of
        the inner graph run in place of this node instead.

        Args:
            *args: The values of the input ports, in order.
            **kwargs: The values of input ports by name.

        Returns:
            Any: The value of the output port, or a tuple for several.

        Raises:
            RuntimeError: If the node is not built and waits for sizes to be
                inferred from connected data.
        """
        if self.state == NodeState.UNINITIALIZED:
            self.require_known_sizes()
        inputs = dict(zip(self.input_ports, args))
        inputs.update({self.get_input_port(key): value for key, value in kwargs.items()})
        outputs = self._graph.evaluate(inputs, self.output_ports, self.mode)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def setup_graph(
        self,
        graph: Graph,
        input_ports: list[InputPort] | dict[InputPort, list[InputPort]],
        output_ports: list[OutputPort] | dict[OutputPort, OutputPort],
    ):
        """
        Sets up the encapsulated graph and its port mappings.

        This method is intended to be called during the `setup` phase of the node,
        not during initialization, to allow for dynamic graph construction.

        Args:
            graph (Graph): The graph to encapsulate.
            input_ports (list[InputPort] | dict[InputPort, list[InputPort]]): Input port
                mapping.
            output_ports (list[OutputPort] | dict[OutputPort, OutputPort]): Output port
                mapping.
        """
        self._graph = graph
        self.update_inner_input_ports(input_ports)
        self.update_inner_output_ports(output_ports)
        self._graph.setup(plan=False)

    def _configs_were_defined_in_forward(self):
        """
        Checks if the data configurations for the GraphNode's ports were explicitly
        defined using `Annotated` type hints in its `forward` method.

        Returns:
            bool: True if configurations were defined in `forward`, False otherwise.
        """
        call_sig = inspect.signature(self.forward)
        for param in call_sig.parameters.values():
            hint = param.annotation
            _, was_annotated = self._unwrap_annotated(hint, self)
            if not was_annotated:
                return False
        return True

    _setup_of_library = True

    def setup(self, graph=None) -> None:
        """Build this node (see :meth:`Node.setup`), then set up its inner
        graph.

        Args:
            graph (Graph, optional): The graph being set up.
        """
        super().setup(graph)
        self._graph.setup(plan=False)

    def set_mode(self, new_mode: EvaluationPhase):
        self._graph.mode = new_mode
        for node in self._graph.nodes:
            node.set_mode(new_mode)
        return super().set_mode(new_mode)

    def reset(self):
        """Make this node and every node of its inner graph unbuilt (see
        :meth:`Node.reset`)."""
        for node in self._graph.nodes:
            node.reset()
        super().reset()

    def to(self, device):
        """
        Moves all nodes within the encapsulated graph to a specified device (e.g.,
        GPU, CPU).

        Args:
            device: The target device.
        """
        for node in self._graph.nodes:
            node.to(device=device)

    def unfreeze(self):
        """Unfreezes the node, allowing modifications. (Currently a placeholder)."""
        # for zooming into the graph and modifying stuff

    def _build_graph_from_function(self, function, backend):
        graph, input_ports, output_ports = Graph.from_function(function)
        outer_input_ports, outer_output_ports = Node._build_ports(function, self)

        # Outputs that are integers are automatically mapped to the
        # corresponding input ports.
        output_ports_remapped = []
        for i, port in enumerate(output_ports):
            if isinstance(port, int):
                output_ports_remapped.append(outer_input_ports[i])
            else:
                output_ports_remapped.append(port)
        output_ports = output_ports_remapped

        # If signature does not include any information about the output,
        # derive them from the return values
        if len(outer_output_ports) == 0:
            for p in output_ports:
                outer_output_ports.append(
                    OutputPort(
                        deepcopy(p.data_configuration),
                        self,  # type: ignore
                        name=p.name,
                    )
                )
        # Make input ports consistent
        input_port_list = []
        for p in input_ports:
            if not isinstance(p, list):
                input_port_list.append([p])
            else:
                input_port_list.append(p)

        # Build graph node:
        input_ports_dict = dict(zip(outer_input_ports, input_port_list))
        output_ports_dict = dict(zip(outer_output_ports, output_ports))
        return graph, input_ports_dict, output_ports_dict

    @property
    def memory_consumption(self) -> int:
        total_bytes = 0
        for node in self._graph.nodes:
            total_bytes += node.memory_consumption
        return total_bytes


class FromFunctionNode(GraphNode):
    """
    A GraphNode that is constructed by analyzing the signature and execution
    flow of a provided function.

    This node automatically builds an internal graph based on how `TrackingObject`s
    are used within the given function.

    Args:
        function (Callable): The function to analyze for building the graph.
        name (str, optional): The name of this node. Defaults to "FromFunctionNode".
        backend (type[Backend[TensorType]], optional): The backend type for tensor
            operations. Defaults to Backend.
    """

    def __init__(
        self,
        function: Callable,
        name="FromFunctionNode",
        backend: type[Backend[TensorType]] = Backend,
    ) -> None:
        graph, input_ports_dict, output_ports_dict = self._build_graph_from_function(
            function, backend=backend
        )
        super().__init__(
            graph, input_ports_dict, output_ports_dict, name=name, backend=backend
        )
        self._graph.setup(plan=False)


class TrackedNode(FromFunctionNode):
    """
    A GraphNode where the internal graph is automatically built by tracking the
    `forward` method's execution.

    Args:
        name (str, optional): The name of this node. Defaults to "TrackedNode".
        backend (type[Backend[TensorType]], optional): The backend type for tensor
            operations. Defaults to Backend.
    """

    def __init__(self, name="TrackedNode", backend: type[Backend[TensorType]] = Backend):
        super().__init__(self.forward, name=name, backend=backend)

    def unfreeze(self):
        """
        Unfreezes the node. For a TrackedNode, this currently does nothing,
        but could be extended to convert it into a modifiable GraphNode.
        """
        # todo, this makes this a normal GraphNode and unfreezes it?

    def build(self) -> None:
        new_graph, input_ports_dict, output_ports_dict = self._build_graph_from_function(
            self.forward, backend=self.backend
        )
        # Tracking builds new outer ports; the inner graph is connected to the
        # ports this node already has, in the same order.
        input_ports_dict = dict(zip(self.input_ports, input_ports_dict.values()))
        output_ports_dict = dict(zip(self.output_ports, output_ports_dict.values()))
        self.setup_graph(
            new_graph,
            input_ports=input_ports_dict,
            output_ports=output_ports_dict,
        )

class CopiedNode(GraphNode):
    """
    A special type of GraphNode that encapsulates a copy of another Node.

    This allows for creating instances of existing nodes within a single graph context
    without directly modifying the original node.

    Args:
        node_to_copy (Node): The node to be copied and encapsulated.
    """

    def __init__(self, node_to_copy: Node) -> None:
        self.copied_node = node_to_copy
        super().__init__(
            Graph(), self.copied_node.input_ports, self.copied_node.output_ports
        )
        self.forward = self.copied_node.forward

    def copy(self):
        """
        Returns a new CopiedNode instance that refers to the same original node.

        This prevents infinite recursion if `copy` is called on an already copied node.
        """
        return CopiedNode(self.copied_node)
