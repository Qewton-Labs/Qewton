from plotly import graph_objects as go

from qewton.visualization.renderers.plotly.common import PlotlyArtist


class NodeLinkArtist(PlotlyArtist):
    """Draws a GraphPlot's node-link layout.

    Deliberately not a single Plotly primitive: node and cluster rectangles
    are `layout.shapes` (cheap regardless of graph size, but shapes alone
    can't hover or show a legend); labels are `layout.annotations`; and
    three go.Scatter traces carry everything interactive - one invisible
    marker per node for hover, one small circular marker per port (at the
    exact anchor edges connect to, so a port's position on the node border
    is never just implied by where a line happens to end), and one combined
    "lines+markers" trace for every edge (the same (start, tip, None)-
    segment + rotated-arrowhead-marker pattern ArrowField2DArtist uses for
    2D vectors).

    Shape/annotation count scales with the graph; trace count never does
    (always exactly 3), which is what keeps a large graph fast to draw.
    """

    def __init__(self, node_trace_idx, port_trace_idx, edge_trace_idx, shape_range, annotation_range):
        super().__init__(node_trace_idx)
        self.port_trace_idx = port_trace_idx
        self.edge_trace_idx = edge_trace_idx
        self.shape_range = shape_range  # (start, count) - this artist's own slice
        self.annotation_range = annotation_range

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        layout = plot.evaluate()
        shapes, annotations = cls._shapes_and_annotations(plot, layout)

        shape_start = len(backend_figure.layout.shapes)
        annotation_start = len(backend_figure.layout.annotations)
        backend_figure.layout.shapes = tuple(backend_figure.layout.shapes) + tuple(shapes)
        backend_figure.layout.annotations = tuple(backend_figure.layout.annotations) + tuple(annotations)

        node_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._node_hover_trace(layout), row=row, col=col)
        port_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._port_trace(layout, plot.theme), row=row, col=col)
        edge_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._edge_trace(layout, plot.theme), row=row, col=col)

        backend_figure.update_xaxes(visible=False, row=row, col=col)
        backend_figure.update_yaxes(visible=False, row=row, col=col, scaleanchor="x")

        return cls(
            node_idx, port_idx, edge_idx, (shape_start, len(shapes)), (annotation_start, len(annotations))
        )

    @staticmethod
    def _node_color(category, theme):
        return theme.node_color_by_type.get(category, theme.node_color_default)

    @classmethod
    def _shapes_and_annotations(cls, plot, layout):
        theme = plot.theme
        shapes, annotations = [], []
        for cluster in layout.clusters:
            shapes.append(
                dict(
                    type="rect",
                    x0=cluster.x0,
                    y0=cluster.y0,
                    x1=cluster.x1,
                    y1=cluster.y1,
                    line=dict(color=theme.cluster_outline_color, dash="dash", width=1),
                    fillcolor="rgba(0,0,0,0)",
                    layer="below",
                )
            )
            annotations.append(
                dict(
                    x=cluster.x0,
                    y=cluster.y1,
                    text=cluster.label,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=10, color=theme.cluster_outline_color),
                )
            )

        for node in layout.nodes:
            color = cls._node_color(node.category, theme)
            x0, x1 = node.x - node.width / 2, node.x + node.width / 2
            y0, y1 = node.y - node.height / 2, node.y + node.height / 2
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color=theme.line_color, width=1),
                    fillcolor=color,
                    layer="above",
                )
            )
            annotations.append(
                dict(
                    x=node.x,
                    y=node.y,
                    text=node.label,
                    showarrow=False,
                    font=dict(size=10, color=theme.text_color),
                    xanchor="center",
                    yanchor="middle",
                )
            )

        # One pass over every port anywhere (leaf-node and cluster-boundary
        # alike - GraphLayoutResult.ports is already flat), rather than
        # walking each NodeLayout's own port lists: a cluster-boundary port
        # has no owning NodeLayout to walk from at all.
        for port in layout.ports:
            if port.show_label:
                annotations.append(cls._port_annotation(port, theme))
        return shapes, annotations

    @staticmethod
    def _port_annotation(port, theme):
        # A label extends away from the side its port sits on: input ports
        # are on a box's/cluster's LEFT edge, so the label's right edge
        # anchors there and the text grows further left (outside); output
        # ports mirror that on the right.
        xanchor = "right" if port.is_input else "left"
        return dict(
            x=port.x,
            y=port.y,
            text=port.name,
            showarrow=False,
            font=dict(size=7, color=theme.text_color),
            xanchor=xanchor,
            yanchor="middle",
            xshift=(-3 if port.is_input else 3),
        )

    @staticmethod
    def _node_hover_trace(layout):
        return go.Scatter(
            x=[n.x for n in layout.nodes],
            y=[n.y for n in layout.nodes],
            mode="markers",
            marker=dict(size=[max(n.width, n.height) * 35 for n in layout.nodes], opacity=0),
            hovertext=[
                f"{n.label} ({n.kind})" + (f"<br>{n.hover}" if n.hover else "") for n in layout.nodes
            ],
            hoverinfo="text",
            showlegend=False,
        )

    @staticmethod
    def _port_trace(layout, theme):
        # A port circle reads as a small "hole" through the node border -
        # filled with the page background, not a fixed white that would
        # clash against a dark theme.
        return go.Scatter(
            x=[p.x for p in layout.ports],
            y=[p.y for p in layout.ports],
            mode="markers",
            marker=dict(
                size=6, color=theme.background_color, line=dict(color=theme.line_color, width=1)
            ),
            hovertext=[f"{p.name} ({'in' if p.is_input else 'out'})" for p in layout.ports],
            hoverinfo="text",
            showlegend=False,
        )

    @staticmethod
    def _edge_trace(layout, theme):
        xs, ys, hovertext, sizes = [], [], [], []
        for edge in layout.edges:
            for i, (px, py) in enumerate(edge.points):
                xs.append(px)
                ys.append(py)
                hovertext.append(edge.label)
                sizes.append(8 if i == len(edge.points) - 1 else 0)
            xs.append(None)
            ys.append(None)
            hovertext.append(None)
            sizes.append(0)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color=theme.line_color, width=1),
            marker=dict(symbol="arrow", angleref="previous", size=sizes, color=theme.line_color),
            hovertext=hovertext,
            hoverinfo="text",
            showlegend=False,
        )

    def update(self, backend_figure, plot):
        layout = plot.evaluate()
        shapes, annotations = self._shapes_and_annotations(plot, layout)

        current_shapes = list(backend_figure.layout.shapes)
        start, count = self.shape_range
        current_shapes[start : start + count] = shapes
        backend_figure.layout.shapes = tuple(current_shapes)

        current_annotations = list(backend_figure.layout.annotations)
        start, count = self.annotation_range
        current_annotations[start : start + count] = annotations
        backend_figure.layout.annotations = tuple(current_annotations)

        node_trace = backend_figure.data[self.figure_idx]
        new_nodes = self._node_hover_trace(layout)
        node_trace.x, node_trace.y = new_nodes.x, new_nodes.y
        node_trace.hovertext = new_nodes.hovertext
        node_trace.marker.size = new_nodes.marker.size

        port_trace = backend_figure.data[self.port_trace_idx]
        new_ports = self._port_trace(layout, plot.theme)
        port_trace.x, port_trace.y = new_ports.x, new_ports.y
        port_trace.hovertext = new_ports.hovertext

        edge_trace = backend_figure.data[self.edge_trace_idx]
        new_edges = self._edge_trace(layout, plot.theme)
        edge_trace.x, edge_trace.y = new_edges.x, new_edges.y
        edge_trace.hovertext = new_edges.hovertext
        edge_trace.marker.size = new_edges.marker.size
