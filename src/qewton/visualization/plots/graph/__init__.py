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
    """

    def __init__(self, graph, depth: int = 0, title=None, theme=None):
        super().__init__(title=title, theme=theme)
        self.graph = graph
        self.depth = depth

    @property
    def embedding_dim(self):
        return None  # a node-link diagram has no cartesian axes, like ParallelCoordinatesPlot

    def evaluate(self):
        return GraphLayout.compute(self.graph, depth=self.depth)

    def create_artist(self, backend_figure, renderer, row=None, col=None):
        return renderer.NodeLinkArtist.create(backend_figure, self, row=row, col=col)
