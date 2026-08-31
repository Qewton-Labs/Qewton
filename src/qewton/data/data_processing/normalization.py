from typing import Annotated

from qewton.graphs.control_nodes.data_processing_node import DataProcessingNode
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.algorithms.building_blocks.math import Subtract, Divide, Add, Multiply
from qewton.data.dataloaders.base import DataNode
from qewton.backends.base import TensorType, ComputingBackend
from qewton.backends import DEFAULT_DL_BACKEND
from qewton.optim.base import EvaluationPhase
from qewton.graphs.nodes import NodeState
from qewton.graphs.graphs import Graph
from qewton.config.axes import EllipsisAxes, BatchAxes
from qewton.config.data_configurations import DataConfiguration


class StdNormalizationNode(GraphNode[TensorType], DataProcessingNode[TensorType]):
    """Applies a normalization of the input data by a mean shift and afterwards
    a normalization with the standard derivation.

    Args:
        data_source_node (DataNode): The original source of data. Does
            not have to be node which is directly to this node, only
            the original data loader providing the original data set.
        eps (float, optional): A small tolerance added to the standard derivation,
            to avoid dividing by 0. Defaults to 1.0e-6.
        name (str, optional): Defaults to "Normalization Node".
        backend (type[ComputingBackend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

    data_axes = EllipsisAxes()
    batch_axes = BatchAxes(None)

    def __init__(
        self,
        data_source_node: DataNode,
        eps: float = 1.0e-6,
        name: str = "Normalization Node",
        backend: type[ComputingBackend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        graph = Graph()
        self.sub_node = Subtract(backend=backend)
        self.divide_node = Divide(backend=backend)
        graph.connect(self.sub_node, self.divide_node.input_ports[0])

        self.eps = eps
        self.mean: TensorType
        self.std: TensorType

        super().__init__(
            graph=graph,
            input_ports=[self.sub_node.input_ports[0]],
            output_ports=[self.divide_node.output_ports[0]],
            data_source_node=data_source_node,
            name=name,
            backend=backend,
        )
        self._graph.setup()
        self.data_source_node: DataNode = data_source_node
        self.backend: type[ComputingBackend[TensorType]] = backend

    def reset(self):
        self._state = NodeState.UNINITIALIZED
        self._set_port_values(None, None)
        return super().reset()

    def setup(self, graph: Graph) -> None:
        if self._state == NodeState.INITIALIZED:
            return
        # First collect all data:
        total_data = []
        for _ in range(self.data_source_node.training_batches):
            in_edge = graph.run_to(last_node=self, mode=EvaluationPhase.TRAIN)
            total_data.append(in_edge[self.input_ports[0]].from_port.value)
        self.fit(total_data)

    def fit(self, data_batch):
        # The batch is assumed to be on axis = 0 (see also data configs.)
        total_data = self.backend.math.concatenate(data_batch, axis=0)
        # Now compute the mean and std, save them also into the ports:
        self.mean = self.backend.math.mean(total_data, axis=0, keepdims=True)
        self.std = self.backend.math.std(total_data, axis=0, keepdims=True)
        self.std = self.backend.math.mean(
            self.std,
            axis=tuple(i for i in range(1, len(self.std.shape) - 1)),
            keepdims=True,
        )
        self.std += self.eps
        self._set_port_values(self.mean, self.std)
        self._state = NodeState.INITIALIZED

    def _set_port_values(self, mean, std):
        self.sub_node.input_ports[1].default = mean
        self.divide_node.input_ports[1].default = std

    def to(self, device):
        if self.state != NodeState.UNINITIALIZED:
            self.mean = self.backend.to(self.mean, device=device)
            self.std = self.backend.to(self.std, device=device)
            self._set_port_values(self.mean, self.std)
        return super().to(device)

    def forward(
        self, x: Annotated[TensorType, DataConfiguration(batch_axes, data_axes)]
    ) -> Annotated[TensorType, DataConfiguration(batch_axes, data_axes)]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore


class InverseStdNormalizationNode(GraphNode[TensorType], DataProcessingNode[TensorType]):
    """Inverts a normalization.

    Args:
        std_node (StdNormalizationNode[TensorType]): The node that applies
            the initial normalization we want to invert.
        name (str, optional): Defaults to "Inverse Std. Norm. Node".
    """

    data_axes = EllipsisAxes()
    batch_axes = BatchAxes(None)

    def __init__(
        self,
        std_node: StdNormalizationNode[TensorType],
        name: str = "Inverse Std. Norm. Node",
    ) -> None:

        graph = Graph()
        self.add_node = Add(backend=std_node.backend)
        self.multiply_node = Multiply(backend=std_node.backend)
        graph.connect(self.multiply_node, self.add_node.input_ports[0])

        super().__init__(
            graph=graph,
            input_ports=[self.multiply_node.input_ports[0]],
            output_ports=[self.add_node.output_ports[0]],
            data_source_node=std_node,
            name=name,
            backend=std_node.backend,
        )
        self._graph.setup()
        self.data_source_node: StdNormalizationNode = std_node

    def setup(self, graph: Graph) -> None:  # pylint: disable=W0613
        if self.data_source_node.state == NodeState.UNINITIALIZED:
            raise RuntimeError(
                f"Connected StdNormalizationNode {self.data_source_node} has not \
                    been setup yet!"
            )
        self._set_port_values(self.data_source_node.mean, self.data_source_node.std)
        self._state = NodeState.INITIALIZED

    def _set_port_values(self, mean, std):
        self.add_node.input_ports[1].default = mean
        self.multiply_node.input_ports[1].default = std

    def to(self, device):
        if self.data_source_node.state != NodeState.UNINITIALIZED:
            self.data_source_node.to(device)
            self._set_port_values(self.data_source_node.mean, self.data_source_node.std)
        return super().to(device)

    def forward(
        self, x: Annotated[TensorType, DataConfiguration(batch_axes, data_axes)]
    ) -> Annotated[TensorType, DataConfiguration(batch_axes, data_axes)]:
        self.input_ports[0].set_value(x)
        self.run()
        return self.output_ports[0].value  # type: ignore
