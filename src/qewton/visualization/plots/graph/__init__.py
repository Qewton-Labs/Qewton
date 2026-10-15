from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.graph.layout import GraphLayout


class GraphPlot(Plot):
    """Visualizes a computation Graph as a node-link diagram.

    Its own input family: it takes a Graph directly, not data mapped to
    roles via specs, so it has no `color`/`x`/`y` and accepts no controls.

    `depth` controls how many levels of composite nodes (e.g. an FCN, or any
    other GraphNode subclass - including Linear, which wraps its own
    weight/bias parameters) are expanded inline as a labeled cluster, rather
    than drawn as a single collapsed box. 0 (default) collapses every
    composite, matching how a graph is normally read at a glance; each level
    deeper reveals one more layer of internal structure.

    `show_port_names` is read by NodeLinkArtist, not by GraphLayout - it
    doesn't change any position, only whether a multi-port node's own ports
    (a GraphNode's expanded boundary ports included) additionally get a name
    annotation next to the node's title.

    `show_edge_labels` is also read by NodeLinkArtist only: False suppresses
    every edge's dynamic-data-config label (and its background box) without
    touching the edges themselves - useful once a graph has enough of them
    that the labels start cluttering more than they clarify.
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
        return GraphLayout.compute(self.graph, depth=self.depth)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.NodeLinkArtist.create(backend_figure, self, row=row, col=col)
