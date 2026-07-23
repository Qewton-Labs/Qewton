from qewton.graphs.nodes import Node, NodeState
from qewton.backends.base import Backend, TensorType
from qewton.backends import DEFAULT_DL_BACKEND


class DataProcessingNode(Node[TensorType]):

    def __init__(
        self,
        data_source_node: Node,
        name: str | None = "DataProcessingNode",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
        **kwargs,
    ) -> None:
        self.data_source_node = data_source_node
        super().__init__(name, NodeState.UNINITIALIZED, backend, **kwargs)

    def setup(self, graph) -> None:
        pass
