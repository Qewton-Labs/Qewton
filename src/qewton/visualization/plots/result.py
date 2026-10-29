"""Renderer-neutral results returned by ``Plot.evaluate()``.

A Result is one state of one drawn primitive, including how it is drawn;
a renderer draws it with the artist for its class. The layout classes
(:class:`NodeLayout` and the like) are parts of a
:class:`GraphLayoutResult`.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from qewton.visualization.scales import ColorScale


@dataclass(kw_only=True)
class Result:
    """One state of one drawn primitive.

    Positions have one column per position channel (x, y, z); `values`
    are the values of the color channel, one per element.

    Args:
        label: Legend entry, or None for no entry.
        color_index: Index of the categorical color of every element (one
            per drawn copy of the figure) if no color channel is bound;
            the renderer's theme gives the color.
        color_scale: Maps `values` to colors if a color channel is bound,
            else None.
        color_title: Title of the colorbar.
        opacity: Opacity of every element.
    """

    label: str | None = None
    color_index: int = 0
    color_scale: ColorScale | None = None
    color_title: str | None = None
    opacity: float = 1.0


@dataclass(kw_only=True)
class HeatmapResult(Result):
    """Values on a rectilinear 2D grid, drawn as colored cells.

    Args:
        x: Coordinates along x, shape ``(nx,)``.
        y: Coordinates along y, shape ``(ny,)``.
        values: Shape ``(ny, nx)``.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray


@dataclass(kw_only=True)
class SurfaceResult(Result):
    """Filled triangles, flat in 2D or a surface in 3D.

    Args:
        positions: Shape ``(n_points, dim)``.
        cells: Triangles as point indices, shape ``(n_cells, 3)``.
        values: One value per point, or None for a uniform color.
        show_edges: Whether the cell edges are drawn.
    """

    positions: np.ndarray
    cells: np.ndarray
    values: np.ndarray | None = None
    show_edges: bool = False


@dataclass(kw_only=True)
class LineResult(Result):
    """A line through points, colored along its length if `values` is set.

    Args:
        positions: Shape ``(n_points, dim)``.
        segments: Point index pairs to join, shape ``(n_segments, 2)``.
        values: One value per point, or None for a uniform color.
        width: Line width, or None for the theme's.
        dash: Dash pattern, e.g. ``"dash"``, or None for a solid line.
    """

    positions: np.ndarray
    segments: np.ndarray
    values: np.ndarray | None = None
    width: float | None = None
    dash: str | None = None


@dataclass(kw_only=True)
class MarkerResult(Result):
    """A marker per point.

    Args:
        positions: Shape ``(n_points, dim)``.
        values: One value per point, or None for a uniform color.
        size: Marker size, or None for the theme's.
        symbol: Marker symbol, or None for the default.
    """

    positions: np.ndarray
    values: np.ndarray | None = None
    size: float | None = None
    symbol: str | None = None


@dataclass(kw_only=True)
class BarResult(Result):
    """A bar per point, from 0 to its height.

    Args:
        positions: Shape ``(n_points, 2)``: the position along x and the
            height.
        values: One value per bar, or None for a uniform color.
    """

    positions: np.ndarray
    values: np.ndarray | None = None


@dataclass(kw_only=True)
class ImageResult(Result):
    """Colors on a rectilinear 2D grid with uniform spacing.

    Args:
        x: Coordinates along x, shape ``(nx,)``.
        y: Coordinates along y, shape ``(ny,)``.
        colors: RGB or RGBA components in [0, 1], shape ``(ny, nx, 3)`` or
            ``(ny, nx, 4)``.
    """

    x: np.ndarray
    y: np.ndarray
    colors: np.ndarray


@dataclass(kw_only=True)
class ArrowResult(Result):
    """An arrow per point.

    Args:
        positions: Tails of the arrows, shape ``(n_points, dim)``.
        vectors: The arrows in data units, shape ``(n_points, dim)``.
        values: One value per arrow, or None for a uniform color.
    """

    positions: np.ndarray
    vectors: np.ndarray
    values: np.ndarray | None = None


@dataclass(frozen=True)
class ParallelDimension:
    """One axis of a parallel coordinates plot.

    Args:
        title: Axis title.
        values: One value per drawn row, in drawn values (e.g. log10 of a
            log column, codes of a categorical one).
        range: Range of the axis, in drawn values.
        ticks: Tick positions and labels, or None for automatic ticks.
    """

    title: str
    values: np.ndarray
    range: tuple[float, float]
    ticks: tuple[list[float], list[str]] | None = None


@dataclass(kw_only=True)
class ParallelCoordinatesResult(Result):
    """A line per row through one vertical axis per column.

    Args:
        dimensions: The axes, in order.
        values: One value per row, or None for a uniform color.
    """

    dimensions: list[ParallelDimension]
    values: np.ndarray | None = None


class NodeCategory(Enum):
    """What a node of a computation graph is, for its colors; the value
    names its colors in the theme (``Theme.node_color_by_type``).

    Attributes:
        DATA: A data source, e.g. a data loader or sampler.
        MODEL: A model, algorithm or any other node.
        CONSTRAINT: A loss or residual.
    """

    DATA = "datanode"
    MODEL = "graphnode"
    CONSTRAINT = "constraint"


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
        category: What the node is, for its colors.
        x: Center x of the box.
        y: Center y of the box.
        width: Width of the box.
        height: Height of the box.
        input_ports: The node's ports on its left edge.
        output_ports: The node's ports on its right edge.
        hover: Extra hover text.
    """

    label: str
    kind: str
    category: NodeCategory
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
        category: What the composite is, as for a collapsed
            :class:`NodeLayout`.
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
    category: NodeCategory
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


@dataclass(kw_only=True)
class GraphLayoutResult(Result):
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
        show_port_names: Whether the ports of multi-port nodes are named
            next to the node title.
        show_edge_labels: Whether edges are labeled with the axes of their
            data configuration.

    See Also:
        :meth:`~qewton.visualization.plots.graph.layout.GraphLayout.compute`
    """

    nodes: list[NodeLayout]
    ports: list[PortLayout]
    edges: list[EdgeLayout]
    clusters: list[ClusterBox]
    show_port_names: bool = False
    show_edge_labels: bool = True
