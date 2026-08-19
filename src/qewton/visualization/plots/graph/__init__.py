from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.graph.layout import GraphLayout


class GraphPlot(Plot):
    """Visualizes a computation Graph as a node-link diagram - its own input
    family (implementation plan, section 5): takes a Graph, not data mapped
    to roles, so it has no specs and no controls at all. A computation graph
    isn't a value source mapped to a role - trying to force one would repeat
    the mistake PlotParameter/ParameterSpec was discarded for (section 11).

    `depth` controls how many levels of composite nodes (FCN, and any other
    GraphNode subclass - Linear included, since it wraps its own weight/bias
    parameters) get expanded inline as a labeled cluster, rather than drawn
    as a single collapsed box. 0 (default) collapses every composite,
    matching how a graph is normally read at a glance; each level deeper
    reveals one more layer of internal structure.
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
