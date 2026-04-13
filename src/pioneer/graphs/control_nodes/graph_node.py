from typing import cast

from ..nodes import InputPort, Node, OutputPort, Port
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
    input_ports (list[InputPort] | dict[InputPort, str | InputPort]):
        The input ports of the inner graph that are mapped to the input ports of
        this node. If a list is provided, the ports are mapped in the order they
        are given. If a dict is provided, the keys are the input ports of the inner
        graph and the values are either the names of the input ports of this node or
        the input ports of this node themselves.
    output_ports (list[OutputPort] | dict[OutputPort, str | OutputPort]):
        The output ports of the inner graph that are mapped to the output ports of
        this node. If a list is provided, the ports are mapped in the order they are
        given. If a dict is provided, the keys are the output ports of the inner
        graph and the values are either the names of the output ports of this node or
        the output ports of this node themselves.
    name (str, optional): The name of this node. Defaults to "GraphNode".
    """

    def __init__(
        self,
        graph: Graph,
        input_ports: list[InputPort] | dict[InputPort, str | InputPort],
        output_ports: list[OutputPort] | dict[OutputPort, str | OutputPort],
        name: str = "GraphNode",
    ) -> None:
        super().__init__(name=name)
        self.graph = graph

        new_input_ports, _inner_input_ports = self._create_ports(
            input_ports  # type: ignore
        )
        for port in new_input_ports:
            assert isinstance(port, InputPort)
            assert port.node == self
        self._inner_input_ports = cast(list[InputPort], _inner_input_ports)
        self._input_ports = cast(list[InputPort], new_input_ports)

        new_output_ports, self._inner_output_ports = self._create_ports(
            output_ports  # type: ignore
        )
        for port in new_output_ports:
            assert isinstance(port, OutputPort)
            assert port.node == self
        self._output_ports = cast(list[OutputPort], new_output_ports)

    def _create_ports(
        self,
        provided_ports: (
            list[InputPort | OutputPort]
            | dict[InputPort | OutputPort, str | InputPort | OutputPort]
        ),
    ) -> tuple[list[Port], list[InputPort | OutputPort]]:
        if isinstance(provided_ports, list):
            ports_with_new_owner = [
                port.duplicate_with_new_owner(self) for port in provided_ports
            ]
            return ports_with_new_owner, provided_ports
        if isinstance(provided_ports, dict):
            provided_port_list = list(provided_ports.keys())
            ports_with_new_owner = []
            for port in provided_port_list:
                value = provided_ports[port]
                ports_with_new_owner.append(
                    port.duplicate_with_new_owner(self, new_name=value)
                    if isinstance(value, str)
                    else value
                )
            return ports_with_new_owner, provided_port_list

        raise ValueError("input_ports and output_ports should be either list or dict.")

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        hp_list = []
        for node in self.graph.nodes:
            hp_list.extend(node.hyperparameters)
        return hp_list

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        return self.graph.collect_trainable_parameters()

    def setup(self) -> None:
        self.graph.setup()

    def run(self):
        # TODO: Maybe make faster by removing the loop
        for i, port in enumerate(self._input_ports):  # type: ignore
            self._inner_input_ports[i].input_received_from_outside_graph = True
            self._inner_input_ports[i].set_value(port.value)
        self.graph.run(self.mode)
        # Write the inner information into the own output ports
        for i, out_port in enumerate(self._inner_output_ports):
            self._output_ports[i].set_value(out_port.value)  # type: ignore

    def reset(self):
        for node in self.graph.nodes:
            node.reset()
