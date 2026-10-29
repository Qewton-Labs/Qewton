"""Plot of computation graphs: :class:`GraphPlot`."""

from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.graph.layout import GraphLayout
from qewton.visualization.plots.result import GraphLayoutResult


class GraphPlot(Plot):
    """A computation graph as a node-link diagram, drawn left to right in a
    2D cell whose axes it hides.

    Args:
        graph (Graph): The graph to draw; must already be sorted
            (``graph.sort()`` or ``graph.setup()``).
        depth: How many levels of composite nodes
            (:class:`~qewton.graphs.control_nodes.graph_node.GraphNode`
            subclasses) are expanded inline as a labeled cluster instead of
            drawn as a single box. 0 collapses every composite.
        show_port_names: Whether the ports of multi-port nodes are named
            next to the node title.
        show_edge_labels: Whether edges are labeled with the axes of their
            data configuration.
        title: Heading of the plot's cell.

    Attributes:
        graph (Graph): The drawn graph.
        depth (int): Number of expanded composite levels.
        show_port_names (bool): Whether port names are drawn.
        show_edge_labels (bool): Whether edges are labeled.
    """

    dim = 2

    def __init__(
        self,
        graph,
        depth: int = 0,
        show_port_names: bool = False,
        show_edge_labels: bool = True,
        title: str | None = None,
    ) -> None:
        super().__init__(title=title)
        self.graph = graph
        self.depth = depth
        self.show_port_names = show_port_names
        self.show_edge_labels = show_edge_labels

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        return ("x", "y")

    def evaluate(self) -> list[GraphLayoutResult]:
        """The layout of the graph.

        Returns:
            list of GraphLayoutResult: One Result with the positioned
                nodes, ports, edges and clusters.

        See Also:
            :meth:`~qewton.visualization.plots.graph.layout.GraphLayout.compute`
        """
        result = GraphLayout.compute(self.graph, depth=self.depth)
        result.show_port_names = self.show_port_names
        result.show_edge_labels = self.show_edge_labels
        return [result]
