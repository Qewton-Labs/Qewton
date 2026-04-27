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
        The input ports of the inner graph that are mapped to the input ports of
        this node. The ports should have the same order as the input variables of
        the forward method.
    output_ports (list[OutputPort] | dict[OutputPort, str | OutputPort]):
        The output ports of the inner graph that are mapped to the output ports of
        this node. The ports should have the same order as the outputs of
        the forward method.
    name (str, optional): The name of this node. Defaults to "GraphNode".
    """

    def __init__(
        self,
        graph: Graph,
        input_ports: list[InputPort],
        output_ports: list[OutputPort],
        name: str = "GraphNode",
    ) -> None:
        super().__init__(name=name)
        self.graph = graph

        assert len(self._input_ports) == len(
            input_ports
        ), "Graph node inputs and and inner graph inputs are different"
        self._inner_input_ports = input_ports

        assert len(self._output_ports) == len(
            output_ports
        ), "Graph node outputs and and inner graph outputs are different"
        self._inner_output_ports = output_ports

    def update_inner_input_ports(self, new_input_ports: list[InputPort]):
        for i, port in enumerate(new_input_ports):
            self._inner_input_ports[i] = port

    def update_inner_output_ports(self, new_output_ports: list[OutputPort]):
        for i, port in enumerate(new_output_ports):
            self._inner_output_ports[i] = port

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        hp_list = []
        for node in self.graph.nodes:
            hp_list.extend(node.hyperparameters)
        return hp_list

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        return self.graph.collect_trainable_parameters()

    @property
    def _trainable_parameters(self) -> _TrainableParameterBase:
        return self.trainable_parameters

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

    def to(self, device):
        for node in self.graph.nodes:
            node.to(device=device)
