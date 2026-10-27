"""Renderer-agnostic result dataclasses returned by ``Plot.evaluate()``.

Results of colorable plots have a ``color`` field, which
:meth:`~qewton.visualization.plots.base.Plot.color_values` reads to train a
shared :class:`~qewton.visualization.plots.spec.Scale`.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GridResult:
    """Result of a structured-grid plot (heatmap, surface, image).

    Positions are implicit grid indices; axis titles come from the plot's
    :class:`~qewton.visualization.plots.spec.AxisSpec`.

    Args:
        values: Grid values with the y dimension first and the x dimension
            second.
        color: Color values in the same orientation, or None.

    See Also:
        :class:`~qewton.visualization.plots.data.grid.StructuredGridPlot`
    """

    values: np.ndarray
    color: np.ndarray | None = None


@dataclass
class ParametricGridResult:
    """Result of a structured grid drawn at explicit coordinates.

    Args:
        values: Grid values.
        x: X coordinates of the grid. For an embedded grid, the x component of
            the 3D position of each grid point.
        y: Y coordinates of the grid. For an embedded grid, the y component of
            the 3D position of each grid point.
        z: Z coordinates of the grid. For an embedded grid, the z component of
            the 3D position of each grid point.
        color: Color values, or None.

    See Also:
        :class:`~qewton.visualization.plots.data.grid.EmbeddedGridPlot`,
        :class:`~qewton.visualization.plots.table.heatmap.TableHeatMap`
    """

    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    color: np.ndarray | None = None


@dataclass
class PointCloudResult:
    """Result of an unstructured point cloud plot.

    Args:
        positions: Point positions, shape ``(n_points, n_components)`` with 1,
            2 or 3 components.
        color: One color value per point, or None.

    See Also:
        :class:`~qewton.visualization.plots.data.points.PointCloudPlot`
    """

    positions: np.ndarray
    color: np.ndarray | None = None


@dataclass
class MeshResult:
    """Result of a mesh plot or a geometry plot.

    Args:
        vertices: Vertex positions, shape ``(n_vertices, dim)``.
        cells: Cells to draw, as indices into `vertices`.
        color: One color value per vertex, or None.

    See Also:
        :class:`~qewton.visualization.plots.data.mesh.MeshPlot`,
        :class:`~qewton.visualization.plots.geometry.GeometryPlot`
    """

    vertices: np.ndarray
    cells: np.ndarray
    color: np.ndarray | None = None


@dataclass
class VectorResult:
    """Result of an arrow-field plot.

    Args:
        positions: Arrow base positions, shape ``(n_arrows, dim)``.
        vectors: Arrow vectors after normalization and scaling, same shape.
        magnitude: Magnitude of each raw vector, before normalization and
            scaling.
        color: One color value per arrow (the magnitude by default), or None
            for flat, uncolored arrows.

    See Also:
        :class:`~qewton.visualization.plots.data.grid.QuiverPlot`,
        :class:`~qewton.visualization.plots.data.mesh.MeshVectorPlot`
    """

    positions: np.ndarray
    vectors: np.ndarray
    magnitude: np.ndarray
    color: np.ndarray | None = None


@dataclass
class CurveResult:
    """Result of a line plot: one curve's x and y values.

    Args:
        x: Curve x values.
        y: Curve y values.

    See Also:
        :class:`~qewton.visualization.plots.data.curve.LinePlot`
    """

    x: np.ndarray
    y: np.ndarray


@dataclass
class PathResult:
    """Result of a path plot: an ordered sequence of positions in space.

    Args:
        positions: Ordered positions, shape ``(n_points, dim)`` with dim 2 or
            3.

    See Also:
        :class:`~qewton.visualization.plots.data.curve.PathPlot`
    """

    positions: np.ndarray


@dataclass
class ScatterResult:
    """Result of a scatter plot: one point per sample.

    Args:
        x: X value of each sample.
        y: Y value of each sample.
        color: One color value per sample, or None.

    See Also:
        :class:`~qewton.visualization.plots.data.samples.ScatterPlot`,
        :class:`~qewton.visualization.plots.table.scatter_table.TableScatter`
    """

    x: np.ndarray
    y: np.ndarray
    color: np.ndarray | None = None


@dataclass
class Column:
    """One table column, as normalized by a table plot.

    Args:
        values: Column values; integer codes for a categorical column.
        labels: Category labels indexed by code, set only for categorical
            columns. Artists turn them into axis ticks.
        log_scale: Whether a numeric column is drawn on a logarithmic axis.

    See Also:
        :class:`~qewton.visualization.plots.table.base.TablePlot`
    """

    values: np.ndarray
    labels: list[str] | None = None
    log_scale: bool = False  # for numeric columns, not categorical ones

    def __getitem__(self, mask) -> "Column":
        """Return a new Column with the rows selected by `mask`.

        `labels` and `log_scale` are kept.
        """
        return Column(
            values=self.values[mask], labels=self.labels, log_scale=self.log_scale
        )

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.values)


@dataclass
class TableResult:
    """Result of a table plot, e.g. a parallel coordinates plot.

    Args:
        columns: The control-reduced columns by name.
        color: One color value per row, or None.

    See Also:
        :class:`~qewton.visualization.plots.table.base.TablePlot`
    """

    columns: dict[str, Column]
    color: np.ndarray | None = None
    # True when `color` already holds log10 of the plotted values.
    color_log_scale: bool = False


@dataclass
class PortLayout:
    """One port's anchor point in a graph layout.

    A leaf node's port sits on its node box's edge; an expanded composite's
    own port sits directly on its :class:`ClusterBox` boundary. Input ports
    are on the left, output ports on the right.

    Args:
        name: Port name.
        x: Anchor x position.
        y: Anchor y position.
        is_input: Whether this is an input port; tells the artist on which side
            the label is placed.
        show_label: Whether the port name is drawn. False for ports of a leaf
            node with only one port; always True for a composite's boundary
            ports.
    """

    name: str
    x: float
    y: float
    is_input: bool = True
    show_label: bool = True


@dataclass
class NodeLayout:
    """One drawn node box in a graph layout.

    Used for leaf nodes and collapsed composites; an expanded composite is
    drawn as a :class:`ClusterBox` instead.

    Args:
        label: Node title.
        kind: Class name of the node, shown in hover text.
        category: Coloring bucket used by the theme: ``"constraint"``,
            ``"datanode"``, ``"graphnode"`` or ``"default"``.
        x: Center x of the box.
        y: Center y of the box.
        width: Width of the box.
        height: Height of the box.
        input_ports: The node's ports on its left edge.
        output_ports: The node's ports on its right edge.
        hover: Extra hover text.
    """

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
    """The card drawn around an expanded composite node's contents.

    Sized to contain the inner nodes and edges, with the composite's own
    ports on its left and right boundary and its title in a corner.

    Args:
        label: Cluster title.
        category: Coloring bucket, the same value the node would have as a
            collapsed :class:`NodeLayout`.
        x0: X coordinate of the first corner of the rectangle.
        y0: Y coordinate of the first corner of the rectangle.
        x1: X coordinate of the opposite corner of the rectangle.
        y1: Y coordinate of the opposite corner of the rectangle.
        input_ports: The composite's own ports on the left boundary.
        output_ports: The composite's own ports on the right boundary.

    Attributes:
        x (float): Center x of the rectangle.
        y (float): Center y of the rectangle.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
    """

    label: str
    category: str
    x0: float
    y0: float
    x1: float
    y1: float
    input_ports: list[PortLayout]
    output_ports: list[PortLayout]

    @property
    def x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class EdgeLayout:
    """One drawn edge between two ports, as a polyline.

    Args:
        points: Polyline points from the source port to the target port.
        label: Edge label (the source port's data configuration axes), or
            empty.
    """

    points: list[tuple[float, float]]
    label: str = ""


@dataclass
class GraphLayoutResult:
    """Renderer-agnostic layout of a graph, the result of a
    :class:`~qewton.visualization.plots.graph.GraphPlot`.

    All entries are already positioned in one global frame, including the
    contents of expanded composites, so the lists are flat. Node colors
    come from ``category`` via the theme.

    Args:
        nodes: Every drawn node box.
        ports: Every port, both those of leaf nodes and the boundary ports of
            expanded composites.
        edges: Every drawn edge.
        clusters: The outline of every expanded composite.

    See Also:
        :meth:`~qewton.visualization.plots.graph.layout.GraphLayout.compute`
    """

    nodes: list[NodeLayout]
    ports: list[PortLayout]
    edges: list[EdgeLayout]
    clusters: list[ClusterBox]
