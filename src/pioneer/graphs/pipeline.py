from .graphs import Graph, Node
from ..constraints.base import Constraint


class Pipeline(Graph):
    """A pipeline represents a workflow of data getting transformed
    through multiple computation steps and algorithms. Along this
    workflow one can set different constraints which can be used
    to train/validate/test the algorithm properties.
    """

    def __init__(self, name="pipeline"):
        """
        Args:
            name (str, optional): The internal name of this pipeline.
                Defaults to "pipeline".
        """
        super().__init__()
        self.constrain_nodes: set[Constraint] = set[Constraint]()
        self.name = name

    def add_node(self, node: Node, check_warning=True) -> None:
        """Adds a node to this pipeline.

        Args:
            node (Node): The node that is added.
            check_warning (bool, optional): Whether it is checked, that
                a node with the same name already exists in this pipeline.
                Defaults to True.
        """
        super().add_node(node, check_warning=check_warning)
        if isinstance(node, Constraint):
            self.constrain_nodes.add(node)

    def remove_node(self, node: Node) -> None:
        """Deletes a given node from this pipeline.

        Args:
            node (Node): The node that should be deleted.
        """
        self.nodes.remove(node)
        if isinstance(node, Constraint):
            self.constrain_nodes.remove(node)


class SequentialPipeline(Pipeline):
    """
    A pipeline that is initialized as a sequence of nodes.
    """

    def __init__(self, *nodes: Node, name="sequential_pipeline"):
        super().__init__(name=name)
        for i in range(len(nodes) - 1):
            self.connect(nodes[i], nodes[i + 1])
        self.sorted_nodes = list(nodes)
