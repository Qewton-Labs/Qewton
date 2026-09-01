from qewton.graphs.nodes import Node, NodeState, NodeConfig
from qewton.backends.base import Backend, TensorType
from qewton.backends import DEFAULT_DL_BACKEND


class DataProcessingNode(Node[TensorType]):
    """A node that processes data from a data source node.
    The data source node is typically a *DataNode*, but can be
    any node that provides data.
    The idea is that, in the setup phase the data is run through the graph
    to collect all input data to this node, and then the data is processed
    in the fit-method.

    Args:
        data_source_node (Node): The node where the original data comes from.
            This node is expected to be a *DataNode*. Note, that it does not
            have to be the node which is directly connected to this node!
        name (str | None, optional): Defaults to "DataProcessingNode".
        backend (type[Backend[TensorType]], optional):
            Defaults to DEFAULT_DL_BACKEND.
    """

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
        """This node also obtains the graph it belongs to, so that it can
        run the graph in the setup phase to collect all input data to
        this node.

        Args:
            graph (Graph): The graph this node belongs to.
        """
        pass

    def fit(self, data_batch: list[TensorType]) -> None:
        """The main fitting behavior of the node.
        This method is called in the setup phase or can also be called manually
        to fit the node to a batch of data.
        The batch is expected to be a list of tensors, where each entry
        corresponds to a batch of data from the data source node.

        Args:
            data_batch (list[TensorType]): The batch of data to fit
                the node to.
        """
        pass
