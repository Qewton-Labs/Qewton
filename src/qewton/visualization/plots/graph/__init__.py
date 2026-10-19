"""Plot family for computation graphs: :class:`GraphPlot`."""

from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.graph.layout import GraphLayout


class GraphPlot(Plot):
    """Visualizes a computation graph as a node-link diagram.

    Its own input family: it takes a
    :class:`~qewton.graphs.graphs.Graph` directly, has no specs and
    accepts no controls. Evaluates to a
    :class:`~qewton.visualization.plots.result.GraphLayoutResult`, drawn
    left to right.

    Args:
        graph (Graph): The graph to draw; must already be sorted
            (``graph.sort()`` or ``graph.setup()``).
        depth: How many levels of composite nodes
            (:class:`~qewton.graphs.control_nodes.graph_node.GraphNode`
            subclasses) are expanded inline as a labeled cluster instead of
            drawn as a single box. 0 collapses every composite.
        show_port_names: Whether the ports of multi-port nodes additionally get
            a name annotation next to the node title. Only affects drawing, not
            the layout.
        show_edge_labels: Whether edges are labeled with the axes of their data
            configuration. Only affects drawing, not the layout.
        title (str, optional): Panel heading.
        theme (Theme, optional): Theme used to draw this plot.

    Attributes:
        graph (Graph): The drawn graph.
        depth (int): Number of expanded composite levels.
        show_port_names (bool): Whether port names are annotated.
        show_edge_labels (bool): Whether edges are labeled.
        embedding_dim (None): Always None; a node-link diagram has no cartesian
            axes.
    """

    def __init__(
        self,
        graph,
        depth: int = 0,
        show_port_names: bool = False,
        show_edge_labels: bool = True,
        title=None,
        theme=None,
    ):
        super().__init__(title=title, theme=theme)
        self.graph = graph
        self.depth = depth
        self.show_port_names = show_port_names
        self.show_edge_labels = show_edge_labels

    @property
    def embedding_dim(self):
        return None  # a node-link diagram has no cartesian axes, like ParallelCoordinatesPlot

    def evaluate(self):
        """Compute the layout of the graph.

        Returns:
            GraphLayoutResult: Positioned nodes, ports, edges and clusters.

        See Also:
            :meth:`~qewton.visualization.plots.graph.layout.GraphLayout.compute`
        """
        return GraphLayout.compute(self.graph, depth=self.depth)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        """Create the renderer's ``NodeLinkArtist`` for this plot."""
        return renderer.NodeLinkArtist.create(backend_figure, self, row=row, col=col)
