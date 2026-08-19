from dataclasses import dataclass

import numpy as np


@dataclass
class GridResult:
    """Result of a StructuredGridPlot family evaluate().

    x/y/z are reserved for real coordinate labels (see the "Tick coordinates"
    open item) - not populated yet, so artists still fall back to indices.
    """

    values: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    color: np.ndarray | None = None


@dataclass
class MeshResult:
    """Result of a MeshPlot family evaluate(), and of GeometryPlot."""

    vertices: np.ndarray
    cells: np.ndarray
    color: np.ndarray | None = None


@dataclass
class VectorResult:
    """Result of MeshVectorPlot.evaluate()."""

    positions: np.ndarray
    vectors: np.ndarray
    magnitude: np.ndarray


@dataclass
class CurveResult:
    """Result of LinePlot.evaluate() - one curve's x/y values.

    x falls back to plain indices until "Tick coordinates" lands, same as
    GridResult.x/y.
    """

    x: np.ndarray
    y: np.ndarray


@dataclass
class PathResult:
    """Result of PathPlot.evaluate() - an ordered sequence of positions in
    space (2D or 3D), e.g. a trajectory or streamline."""

    positions: np.ndarray


@dataclass
class ScatterResult:
    """Result of ScatterPlot.evaluate() - one point per sample.

    Its own dataclass rather than reusing CurveResult: x and y here are both
    value roles over an implicit samples axis (no structural domain like
    LinePlot.x), and color needs its own field for Scale/color_values() to
    find generically, same reasoning as GridResult/MeshResult.
    """

    x: np.ndarray
    y: np.ndarray
    color: np.ndarray | None = None


@dataclass
class Column:
    """One table column, as normalized by TablePlot.__init__.

    `labels` is set for categorical columns, whose `values` are integer
    codes; artists turn labels into axis ticks (Plotly: tickvals/ticktext).
    Coding happens once, in TablePlot, so categories survive into the plot
    instead of being flattened by the caller.
    """

    values: np.ndarray
    labels: list[str] | None = None

    def __getitem__(self, mask) -> "Column":
        return Column(values=self.values[mask], labels=self.labels)

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class TableResult:
    """Result of a TablePlot family evaluate() - ParallelCoordinatesPlot,
    ScatterMatrixPlot and TableScatterPlot all share this, same reasoning as
    one MeshResult for three mesh plots."""

    columns: dict[str, Column]
    color: np.ndarray | None = None


@dataclass
class PortLayout:
    """One port's anchor point - on its node's box edge for a leaf node, or
    directly on the cluster boundary for a composite's own ports (which
    have no box of their own, just a position on the rectangle
    GraphLayoutResult.clusters already draws). Input ports anchor on the
    left, output ports on the right (GraphPlot always draws left to right).

    `show_label` is a structural signal, not a rendering choice: a leaf
    node with only one port doesn't need its name disambiguated (the box
    alone says what it is), but a composite's boundary ports have no box at
    all to lean on, so they always carry a label. `is_input` tells the
    artist which side a label should extend away from (a cluster-boundary
    port has no owning NodeLayout to infer that from, unlike a leaf node's
    own input_ports/output_ports list)."""

    name: str
    x: float
    y: float
    is_input: bool = True
    show_label: bool = True


@dataclass
class NodeLayout:
    """One drawn box - always a real Node. A composite's own ports, when
    expanded, live directly on its ClusterBox's boundary instead (see
    PortLayout) - there is no box for them, so no NodeLayout either."""

    label: str
    kind: str  # type(node).__name__ - shown in hover text
    category: str  # "constraint" | "datanode" | "graphnode" | "default" - what the theme colors by
    x: float
    y: float
    width: float
    height: float
    input_ports: list[PortLayout]
    output_ports: list[PortLayout]
    hover: str = ""


@dataclass
class ClusterBox:
    """The outline drawn around an expanded composite node's contents."""

    label: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class EdgeLayout:
    points: list[tuple[float, float]]
    label: str = ""


@dataclass
class GraphLayoutResult:
    """Result of GraphLayout.compute() - the layout counterpart to
    evaluate() for GraphPlot (implementation plan, section 10, item 7):
    renderer-agnostic node/edge/cluster geometry, with no drawing decisions
    made yet. Flat lists, not a nested tree: an expanded composite's inner
    nodes/edges are already positioned in the outer frame by
    GraphLayout.compute(), so NodeLinkArtist never needs to recurse -
    `clusters` is only the outline rectangles drawn around expanded
    composites' contents, not a container of their nodes.

    `ports` is every port anywhere - both the ones already reachable via
    `nodes[i].input_ports/output_ports` (a leaf node's own) and a
    composite's own boundary ports, which have no owning NodeLayout at all
    once expanded (they live on a ClusterBox's edge instead). A single flat
    list rather than leaving the artist to reconstruct it from two
    different places every draw.

    No `color` field, unlike every other *Result: GraphPlot has no ColorSpec
    at all (no specs of any kind, section 5) - node color comes from `kind`
    via the theme at draw time, not from evaluate().
    """

    nodes: list[NodeLayout]
    ports: list[PortLayout]
    edges: list[EdgeLayout]
    clusters: list[ClusterBox]
