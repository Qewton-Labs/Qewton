from ..nodes import InputPort, Node, OutputPort
from ..graphs import Graph

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
        DataConfigurations andthe output ports of the inner graph are mapped to
        the output ports of this node in the same order.
    name (str, optional): The name of this node. Defaults to "GraphNode".
    """

    def __init__(
        self,
        graph: Graph,
        input_ports: list[InputPort] | dict[InputPort, list[InputPort]],
        output_ports: list[OutputPort] | dict[OutputPort, OutputPort],
        name: str = "GraphNode",
    ) -> None:
        super().__init__(name=name)

        self._graph = graph

        self._input_ports = []
        for p in input_ports:
            if isinstance(input_ports, dict):
                self._input_ports.append(p)
                for inner_port in input_ports[p]:
                    self._graph.connect_from_outside_of_graph(p, inner_port)
            else:
                self._input_ports.append(
                    InputPort(
                        data_configuration=p.data_configuration,
                        node=self,
                        name=p.name,
                    )
                )
                self._graph.connect_from_outside_of_graph(self._input_ports[-1], p)

        self._output_ports = []
        for p in output_ports:
            if isinstance(output_ports, dict):
                self._output_ports.append(p)
                self._graph.connect_to_outside_of_graph(output_ports[p], p)
            else:
                self._output_ports.append(
                    OutputPort(
                        data_configuration=p.data_configuration,
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
                output_ports[i] = outer_input_ports[i]

        input_ports_dict = dict(zip(outer_input_ports, input_ports))
        output_ports_dict = dict(zip(outer_output_ports, output_ports))
        super().__init__(graph, input_ports_dict, output_ports_dict, name=name)
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
