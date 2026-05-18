from typing import get_origin, get_args, Annotated, Callable
import inspect


from ..nodes import InputPort, Node, OutputPort
from ..graphs import Graph
from ...config.data_configurations import DataConfiguration
from ...config.backend import Backend, TensorType
from ...optim.parameters.hyperparameter_base import HyperParameter
from ...optim.parameters.trainable_parameters import _TrainableParameterBase


class GraphNode(Node):
    """A node that encapsulates a whole graph. The inner graph is executed when
    the run method of this node is called. The input and output ports of the inner
    graph are mapped to the input and output ports of this node, respectively.
    This allows to use a whole graph as a single node in another graph.

    Args:
    graph (Graph):
        The graph that is encapsulated by this node.
    input_ports (list[InputPort]):
        If it is a dict, the keys are the input ports of this GraphNode, and the
        values are lists of the corresponding input ports of the inner graph. If
        it is a list, new input ports for this node are created with the same
        names and DataConfigurations and the input ports of the inner graph are
        mapped to the input ports of this node in the same order.
    output_ports (list[OutputPort] | dict[OutputPort, OutputPort]):
        If it is a dict, the keys are the output ports of this GraphNode, and the
        values are the corresponding output ports of the inner graph. If it is a
        list, new output ports for this node are created with the same names and
        DataConfigurations and the output ports of the inner graph are mapped to
        the output ports of this node in the same order.
    name (str, optional): The name of this node. Defaults to "GraphNode".
    """

    def __init__(
        self,
        graph: Graph,
        input_ports: list[InputPort] | dict[InputPort, list[InputPort]],
        output_ports: list[OutputPort] | dict[OutputPort, OutputPort],
        name: str = "GraphNode",
        backend: type[Backend[TensorType]] | None = None,
    ) -> None:
        super().__init__(name=name, backend=backend)

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
        for i, p in enumerate(new_input_ports):
            if isinstance(new_input_ports, dict):
                for inner_port in new_input_ports[p]:
                    self._graph.connect_from_outside_of_graph(p, inner_port)
            else:
                self._graph.connect_from_outside_of_graph(self._input_ports[i], p)

    def update_inner_output_ports(
        self, new_output_ports: list[OutputPort] | dict[OutputPort, OutputPort]
    ):
        for i, p in enumerate(new_output_ports):
            if isinstance(new_output_ports, dict):
                self._graph.connect_to_outside_of_graph(new_output_ports[p], p)
            else:
                self._graph.connect_to_outside_of_graph(p, self._output_ports[i])

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        hp_list = []
        for node in self._graph.nodes:
            hp_list.extend(node.hyperparameters)
        return hp_list

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        return self._graph.collect_trainable_parameters()

    @property
    def _trainable_parameters(self) -> _TrainableParameterBase:
        return self.trainable_parameters

    def forward(self, *args, **kwargs):
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
        Should not be used in init, only in setup
        """
        self._graph = graph
        self.update_inner_input_ports(input_ports)
        self.update_inner_output_ports(output_ports)
        self._graph.setup()

    def update_data_configs(self, updated_port, config_dict, dynamic_configs):
        # Efficient case: the subclass has specified its configs
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
        # these data configs should be identical to the dynamic data configs
        # of the outer graph, therefore these are automatically updated
        return set(connected_to_outside_ports[k] for k in outside_updated_inner_ports)

    def copy_data_configs(self):
        if self.configs_defined_in_forward:
            return super().copy_data_configs()
        dynamic_configs = {}
        for input_port in self.input_ports:
            for e in self._graph.edges_from_outside:
                if e.from_port == input_port:
                    inner_config = self._graph.dynamic_data_configs[e.to_port.node][
                        e.to_port
                    ]
                    dynamic_configs[input_port] = inner_config
        for output_port in self.output_ports:
            for e in self._graph.edges_to_outside:
                if e.to_port == output_port:
                    inner_config = self._graph.dynamic_data_configs[e.from_port.node][
                        e.from_port
                    ]
                    dynamic_configs[output_port] = inner_config
        return dynamic_configs

    def _configs_were_defined_in_forward(self):
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
        self._graph.run()

    def setup(self) -> None:
        self._graph.setup()

    def reset(self):
        for node in self._graph.nodes:
            node.reset()

    def to(self, device):
        for node in self._graph.nodes:
            node.to(device=device)

    def unfreeze(self):
        # for zooming into the graph and modifying stuff
        pass


class TrackedNode(GraphNode):
    """A GraphNode where the graph is built automatically via tracking the forward
    method. At runtime, we still execute the implemented forward method, the graph
    is only for visualization purposes."""

    def __init__(self, name="TrackedNode"):
        graph, input_ports, output_ports = Graph.from_function(self.forward)
        outer_input_ports, outer_output_ports = Node._build_ports(self.forward, self)
        for i, port in enumerate(output_ports):
            if isinstance(port, int):
                output_ports[i] = outer_input_ports[i]  # type: ignore

        input_ports_dict = dict(zip(outer_input_ports, input_ports))
        output_ports_dict = dict(zip(outer_output_ports, output_ports))
        super().__init__(graph, input_ports_dict, output_ports_dict, name=name)  # type: ignore
        self._graph.setup()

        if self.run is TrackedNode.run:

            def run() -> None:
                Node.run(self)

            self.run = run

    def unfreeze(self):
        # todo, this makes this a normal GraphNode
        pass


class CopiedNode(GraphNode):
    def __init__(self, node_to_copy: Node) -> None:
        self.copied_node = node_to_copy
        super().__init__(
            Graph(), self.copied_node.input_ports, self.copied_node.output_ports
        )
        self.forward = self.copied_node.forward

    def copy(self):
        # avoid iterative copying if copy is called multiple times
        return CopiedNode(self.copied_node)
