from __future__ import annotations
from typing import get_origin, get_args, Annotated, Callable
import inspect


from qewton.graphs.nodes import InputPort, Node, OutputPort, Port
from qewton.graphs import Graph
from qewton.config.data_configurations import DataConfiguration
from qewton.config.backend import Backend, TensorType
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import _TrainableParameterBase


class GraphNode(Node):
    """
    A node that encapsulates an entire graph.

    The inner graph is executed when the `run` method of this node is called.
    Input and output ports of the inner graph are mapped to the input and output
    ports of this `GraphNode`, allowing a complete graph to be used as a single
    node within a larger graph structure.

    Args:
        graph (Graph): The graph to be encapsulated by this node.
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
    """

    def __init__(
        self,
        graph: Graph,
        input_ports: list[InputPort] | dict[InputPort, list[InputPort]],
        output_ports: list[OutputPort] | dict[OutputPort, Port],
        name: str = "GraphNode",
        backend: type[Backend[TensorType]] = Backend,
        **kwargs,
    ) -> None:
        super().__init__(name=name, backend=backend)

        self._graph = graph

        self._input_ports = []

        self.configs_defined_in_forward = self._configs_were_defined_in_forward()
        in_forward_ports, out_forward_ports = self._build_ports(
            self.forward, self, backend
        )
        for i, p in enumerate(input_ports):
            if isinstance(input_ports, dict):
                self._input_ports.append(p)
                for inner_port in input_ports[p]:
                    self._graph.connect_from_outside_of_graph(p, inner_port)
            else:
                if self.configs_defined_in_forward:
                    port_config = in_forward_ports[i].data_configuration
                else:
                    port_config = p.data_configuration
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
                    self._graph.connect_to_outside_of_graph(output_ports[p], p)
            else:
                if self.configs_defined_in_forward:
                    port_config = out_forward_ports[i].data_configuration
                else:
                    port_config = p.data_configuration
                self._output_ports.append(
                    OutputPort(
                        data_configuration=port_config,
                        node=self,
                        name=p.name,
                    )
                )
                self._graph.connect_to_outside_of_graph(p, self._output_ports[-1])

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
        """
        Executes the forward pass of the encapsulated graph.

        Input values provided to this `GraphNode` are set on its input ports,
        which then propagate to the inner graph. The inner graph is run, and
        its output values are collected from this `GraphNode`'s output ports.

        Args:
            *args: Positional arguments corresponding to the GraphNode's input ports.
            **kwargs: Keyword arguments corresponding to the GraphNode's input ports.

        Returns:
            Any: The output of the encapsulated graph, either a single value or a
                tuple of values.
        """
        for i, arg in enumerate(args):
            self.input_ports[i].set_value(arg)  # type: ignore
        for key, value in kwargs.items():
            self.get_input_port(key).set_value(value)  # type: ignore

        self._graph.run()
        output_values = [port.value for port in self.output_ports]
        if len(output_values) == 1:
            return output_values[0]
        return tuple(output_values)

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
        self._graph.setup()

    def update_data_configs(self, updated_port, config_dict, dynamic_configs):
        """
        Updates the data configurations of the GraphNode's ports, propagating changes
        to the inner graph if necessary.

        If the GraphNode's configurations are defined in its `forward` method,
        it delegates to the superclass's `update_data_configs`. Otherwise, it infers
        updates from the inner graph's data configuration changes.

        Args:
            updated_port (Port): The port that triggered the update.
            config_dict (dict): A dictionary containing configuration updates.
            dynamic_configs (dict[Port, DataConfiguration]): The current dynamic data
                configurations of the node's ports.
        """

        # Efficient case: the subclass has specified its configs, no need to traverse into
        # the inner graph
        if self.configs_defined_in_forward:
            return super().update_data_configs(updated_port, config_dict, dynamic_configs)

        # Otherwise: infer data configs updates from the inner graph
        # search for inner ports at the beginning and end of the graph
        inner_ports = []
        if isinstance(updated_port, InputPort):
            for e in self._graph.edges_from_outside:
                if e.from_port == updated_port:
                    inner_ports.append(e.to_port)
        else:
            for e in self._graph.edges_to_outside:
                if e.to_port == updated_port:
                    inner_ports.append(e.from_port)

        # perform inner update
        connected_to_outside_ports = {}
        for e in self._graph.edges_from_outside:
            connected_to_outside_ports[e.to_port] = e.from_port
        for e in self._graph.edges_to_outside:
            connected_to_outside_ports[e.from_port] = e.to_port

        inner_updated_ports = set()
        for inner_port in inner_ports:
            inner_updated_ports.update(
                self._graph.update_data_configurations(
                    inner_port.node, inner_port, config_dict
                )
            )

        outside_updated_inner_ports = (
            connected_to_outside_ports.keys() & inner_updated_ports
        )

        outer_updated_ports = set(
            connected_to_outside_ports[k] for k in outside_updated_inner_ports
        )

        # check for skip connections that have been updated
        for e in self._graph.skip_connections:
            if e.to_port in outer_updated_ports:
                outer_updated_ports.add(e.from_port)
            if e.from_port in outer_updated_ports:
                outer_updated_ports.add(e.to_port)

        # these data configs should be identical to the dynamic data configs
        # of the outer graph, therefore these are automatically updated
        return outer_updated_ports

    def copy_data_configs(self):
        """
        Creates deep copies of the data configurations for all input and output ports of
        this node.

        If configurations are defined in `forward`, it uses the superclass method.
        Otherwise, it retrieves configurations from the inner graph's dynamic data
        configurations.
        """
        if self.configs_defined_in_forward:
            return super().copy_data_configs()
        dynamic_configs = {}
        for input_port in self.input_ports:
            for e in self._graph.edges_from_outside + self._graph.skip_connections:
                if e.from_port == input_port:
                    inner_config = self._graph.dynamic_data_configs[e.to_port.node][
                        e.to_port
                    ]
                    dynamic_configs[input_port] = inner_config
        for output_port in self.output_ports:
            for e in self._graph.edges_to_outside + self._graph.skip_connections:
                if e.to_port == output_port:
                    inner_config = self._graph.dynamic_data_configs[e.from_port.node][
                        e.from_port
                    ]
                    dynamic_configs[output_port] = inner_config
        return dynamic_configs

    def _configs_were_defined_in_forward(self):
        """
        Checks if the data configurations for the GraphNode's ports were explicitly
        defined using `Annotated` type hints in its `forward` method.

        Returns:
            bool: True if configurations were defined in `forward`, False otherwise.
        """
        call_sig = inspect.signature(self.forward)
        configs_defined = False
        for param in call_sig.parameters.values():
            hint = param.annotation
            _, config = self._unwrap_annotated(hint, self)
            if get_origin(hint) is Annotated:
                _, *meta = get_args(hint)
                config = next(
                    (m for m in meta if isinstance(m, (DataConfiguration, Callable))),
                    None,
                )
                if isinstance(config, Callable):
                    config = config(self)
                if not isinstance(config, DataConfiguration):
                    return False
                configs_defined = True
            else:
                return False
        return configs_defined

    def run(self):
        """
        Executes the encapsulated graph.
        """
        self._graph.run(mode=self.mode)

    def setup(self) -> None:
        """
        Sets up the encapsulated graph.
        """
        self._graph.setup()

    def reset(self):
        """
        Resets all nodes within the encapsulated graph.
        """
        for node in self._graph.nodes:
            node.reset()

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
        outer_input_ports, outer_output_ports = Node._build_ports(function, self, backend)

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
                        p.data_configuration,
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
        self._graph.setup()


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

        if self.run is TrackedNode.run:

            def run() -> None:
                Node.run(self)

            self.run = run

    def unfreeze(self):
        """
        Unfreezes the node. For a TrackedNode, this currently does nothing,
        but could be extended to convert it into a modifiable GraphNode.
        """
        # todo, this makes this a normal GraphNode and unfreezes it?

    def setup(self) -> None:
        new_graph, input_ports_dict, output_ports_dict = self._build_graph_from_function(
            self.forward, backend=self.backend
        )
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
