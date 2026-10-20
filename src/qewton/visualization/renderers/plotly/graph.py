"""Plotly artist for computation-graph plots."""

from plotly import graph_objects as go

from qewton.visualization.renderers.plotly import card_geometry as cg
from qewton.visualization.renderers.plotly.common import PlotlyArtist


class NodeLinkArtist(PlotlyArtist):
    """Draws a GraphPlot's node-link layout as beveled cards.

    Draws the :class:`~qewton.visualization.plots.result.GraphLayoutResult`
    of a :class:`~qewton.visualization.plots.graph.GraphPlot`. Every
    :class:`~qewton.visualization.plots.result.NodeLayout` and
    :class:`~qewton.visualization.plots.result.ClusterBox` becomes a card
    (see :mod:`~qewton.visualization.renderers.plotly.card_geometry`) with
    one notch per port; node titles are uppercase, rotated 90 degrees and
    centered, cluster titles sit in the top-left corner. Edges are bezier
    wires without arrowheads, optionally labeled.

    Cards, wires, port dots and label backgrounds are ``layout.shapes``,
    texts are ``layout.annotations``. Three invisible-marker
    ``go.Scatter`` traces carry hover text for nodes, ports and edges, so
    the trace count stays 3 regardless of graph size. Shapes are stacked
    back to front: card bodies (clusters before nodes), wires and label
    backgrounds, port dots/rings, edge end dots and edge labels.

    For a non-faceted figure without explicit size, :meth:`create` sets the
    figure size from ``theme.graph_px_per_unit`` and the layout extent. The
    y-axis is reversed (layout y grows downward) and scale-anchored to x.

    Args:
        node_trace_idx (int): Index of the node hover trace.
        port_trace_idx (int): Index of the port hover trace.
        edge_trace_idx (int): Index of the edge hover trace.
        shape_range (tuple of int): ``(start, count)`` of the artist's entries
            in ``layout.shapes``.
        annotation_range (tuple of int): ``(start, count)`` of the artist's
            entries in ``layout.annotations``.

    Attributes:
        port_trace_idx (int): Index of the port hover trace.
        edge_trace_idx (int): Index of the edge hover trace.
        shape_range (tuple of int): ``(start, count)`` of the artist's entries
            in ``layout.shapes``.
        annotation_range (tuple of int): ``(start, count)`` of the artist's
            entries in ``layout.annotations``.
    """

    def __init__(self, node_trace_idx, port_trace_idx, edge_trace_idx, shape_range, annotation_range):
        super().__init__(node_trace_idx)
        self.port_trace_idx = port_trace_idx
        self.edge_trace_idx = edge_trace_idx
        self.shape_range = shape_range
        self.annotation_range = annotation_range

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Draw `plot` into `backend_figure`.

        See :meth:`~qewton.visualization.renderers.base.Artist.create`.
        """
        layout = plot.evaluate()
        shapes, annotations = cls._shapes_and_annotations(plot, layout)

        shape_start = len(backend_figure.layout.shapes)
        annotation_start = len(backend_figure.layout.annotations)
        backend_figure.layout.shapes = tuple(backend_figure.layout.shapes) + tuple(shapes)
        backend_figure.layout.annotations = tuple(backend_figure.layout.annotations) + tuple(annotations)

        node_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._node_hover_trace(layout, plot.theme), row=row, col=col)
        port_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._port_trace(layout, plot.theme), row=row, col=col)
        edge_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._edge_trace(layout, plot.theme), row=row, col=col)

        backend_figure.update_xaxes(visible=False, row=row, col=col)
        # GraphLayout and the card geometry use a y axis growing downward,
        # so the y range below is descending.
        backend_figure.update_yaxes(visible=False, row=row, col=col, scaleanchor="x")

        x0, x1, y0, y1 = cls._layout_bounds(layout)
        backend_figure.update_xaxes(range=[x0, x1], row=row, col=col)
        backend_figure.update_yaxes(range=[y1, y0], row=row, col=col)

        # Pixel size from graph_px_per_unit, only for an unsized,
        # non-faceted figure.
        if row is None and col is None and backend_figure.layout.width is None and backend_figure.layout.height is None:
            px = plot.theme.graph_px_per_unit
            backend_figure.update_layout(
                width=max(400, (x1 - x0) * px), height=max(300, (y1 - y0) * px)
            )

        return cls(
            node_idx, port_idx, edge_idx, (shape_start, len(shapes)), (annotation_start, len(annotations))
        )

    #: Data-unit padding around every node/cluster's own bounding box when
    #: computing the figure's axis range - just enough that a card's own
    #: depth/corner bevel never butts up against the plot's edge.
    _BOUNDS_PADDING = 0.6

    @classmethod
    def _layout_bounds(cls, layout) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) covering every node and cluster card, padded."""
        xs0, xs1, ys0, ys1 = [], [], [], []
        for n in layout.nodes:
            xs0.append(n.x - n.width / 2)
            xs1.append(n.x + n.width / 2)
            ys0.append(n.y - n.height / 2)
            ys1.append(n.y + n.height / 2)
        for c in layout.clusters:
            xs0.append(c.x0)
            xs1.append(c.x1)
            ys0.append(c.y0)
            ys1.append(c.y1)
        if not xs0:
            return -cls._BOUNDS_PADDING, cls._BOUNDS_PADDING, -cls._BOUNDS_PADDING, cls._BOUNDS_PADDING
        p = cls._BOUNDS_PADDING
        return min(xs0) - p, max(xs1) + p, min(ys0) - p, max(ys1) + p

    @staticmethod
    def _node_colors(category, theme) -> tuple[str, str]:
        """(fill, border) for a node/cluster category, from
        theme.node_color_by_type. Unknown categories (e.g. "default") use
        the "graphnode" pair; theme.node_color_default is only used if that
        is missing too."""
        pair = theme.node_color_by_type.get(category) or theme.node_color_by_type.get("graphnode")
        if pair is not None:
            return pair["fill"], pair["border"]
        return theme.node_color_default, theme.line_color

    @classmethod
    def _owned_port_positions(cls, layout) -> set[tuple[float, float]]:
        """(x, y) of every port of a node or cluster card."""
        owned = set()
        for card in (*layout.nodes, *layout.clusters):
            for port in card.input_ports + card.output_ports:
                owned.add((port.x, port.y))
        return owned

    @classmethod
    def _shapes_and_annotations(cls, plot, layout):
        theme = plot.theme
        show_port_names = getattr(plot, "show_port_names", False)
        show_edge_labels = getattr(plot, "show_edge_labels", True)
        shapes, annotations = [], []
        cards = (*layout.nodes, *layout.clusters)

        # Wired output ports get a halo ring. An edge's start point is
        # exactly its port's anchor (GraphLayout reuses the same values), so
        # an exact match is used.
        wired_from = {tuple(e.points[0]) for e in layout.edges}
        card_boxes = [
            (c.x - c.width / 2, c.y - c.height / 2, c.x + c.width / 2, c.y + c.height / 2) for c in cards
        ]

        # Plotly draws shapes in list order, so layers are built back to
        # front. Layer 1: card bodies, clusters first so their inner nodes
        # draw on top.
        for cluster in layout.clusters:
            shapes += cls._card_body_shapes(cluster, theme)
            annotations.append(cls._cluster_annotation(cluster, theme))
        for node in layout.nodes:
            shapes += cls._card_body_shapes(node, theme)
            annotations += cls._card_annotations(node, theme)

        # Layer 2: wire lines + their dynamic-data-config label backgrounds.
        for edge in layout.edges:
            shapes += cls._edge_shapes(edge, theme, card_boxes)

        # Layer 3: port dots/rings, plus port names inside the card if
        # show_port_names is set (same for nodes and clusters).
        for card in cards:
            shapes += cls._card_port_dot_shapes(card, theme, wired_from)
        if show_port_names:
            for card in cards:
                for port in card.input_ports + card.output_ports:
                    if port.show_label:
                        annotations.append(cls._port_annotation_inside(port, theme))

        # Plain circle for a port that belongs to no card (not produced by
        # GraphLayout).
        owned = cls._owned_port_positions(layout)
        for port in layout.ports:
            if (port.x, port.y) in owned:
                continue
            shapes.append(
                dict(
                    type="circle",
                    x0=port.x - theme.card_port_inner_radius,
                    y0=port.y - theme.card_port_inner_radius,
                    x1=port.x + theme.card_port_inner_radius,
                    y1=port.y + theme.card_port_inner_radius,
                    line=dict(color=theme.line_color, width=1),
                    fillcolor=theme.background_color,
                    layer="above",
                )
            )
            if port.show_label:
                annotations.append(cls._port_annotation(port, theme))

        # Layer 4: each edge's end dot and its label, above the port dots.
        for edge in layout.edges:
            p2 = tuple(edge.points[1])
            inner = theme.card_port_inner_radius
            shapes.append(
                dict(
                    type="circle", x0=p2[0] - inner, y0=p2[1] - inner, x1=p2[0] + inner, y1=p2[1] + inner,
                    fillcolor=theme.line_color, line=dict(width=0), layer="above",
                )
            )
            if show_edge_labels and edge.label:
                p1 = tuple(edge.points[0])
                mid = cg.edge_label_point(p1, p2, card_boxes)
                bx0, by0, bx1, by1 = cg.edge_label_box(edge.label, *mid)
                label_bg = cls._label_background_color(mid, cards, theme)
                shapes.append(
                    dict(
                        type="rect", x0=bx0, y0=by0, x1=bx1, y1=by1,
                        fillcolor=label_bg, line=dict(width=0), layer="above",
                    )
                )
                annotations.append(
                    dict(
                        x=mid[0], y=mid[1], text=edge.label, showarrow=False,
                        font=dict(size=theme.card_edge_formula_size, color=theme.line_color),
                        xanchor="center", yanchor="middle", textangle=-90,
                    )
                )

        return shapes, annotations

    @classmethod
    def _label_background_color(cls, mid, cards, theme):
        """Fill color of the smallest card containing `mid`, or
        theme.background_color if none does - so a label background blends
        into what it sits on."""
        mx, my = mid
        best, best_area = None, None
        for card in cards:
            x0, x1 = card.x - card.width / 2, card.x + card.width / 2
            y0, y1 = card.y - card.height / 2, card.y + card.height / 2
            if not (x0 <= mx <= x1 and y0 <= my <= y1):
                continue
            area = card.width * card.height
            if best_area is None or area < best_area:
                best, best_area = card, area
        if best is None:
            return theme.background_color
        return cls._node_colors(best.category, theme)[0]

    @classmethod
    def _card_body_shapes(cls, node, theme):
        """The card body without port dots: back polygon (fill darkened
        from the category fill), front polygon (category fill) and crease
        lines, all stroked in theme.line_color."""
        x0 = node.x - node.width / 2
        y0 = node.y - node.height / 2

        face_color, _border_color = cls._node_colors(node.category, theme)
        outline_color = theme.line_color
        back_color = cg.darken_hex(face_color, 0.87) if face_color.startswith("#") else face_color

        r, outer = theme.card_corner_radius, theme.card_port_outer_radius
        left_notches = [cg.Notch(cy=p.y, r=outer) for p in node.input_ports]
        right_notches = [cg.Notch(cy=p.y, r=outer) for p in node.output_ports]

        back_poly, back_registry = cg.build_back_polygon(
            x0, y0, node.width, node.height, r, left_notches, right_notches, theme.card_depth, 20
        )
        front_registry: dict = {}
        front_poly = cg.build_flat_polygon(
            x0, y0, node.width, node.height, r, left_notches, right_notches, 20, registry=front_registry
        )

        shapes = [
            dict(
                type="path", path=cg.path_from_poly(back_poly, back_registry, True), fillcolor=back_color,
                line=dict(color=outline_color, width=theme.card_line_width), layer="above",
            ),
            dict(
                type="path", path=cg.path_from_poly(front_poly, front_registry, True), fillcolor=face_color,
                line=dict(color=outline_color, width=theme.card_line_width), layer="above",
            ),
        ]
        for cx1, cy1, cx2, cy2 in cg.build_crease_lines(
            x0, y0, node.width, node.height, r, left_notches, right_notches, theme.card_depth
        ):
            shapes.append(
                dict(
                    type="path", path=f"M {cx1:.2f} {cy1:.2f} L {cx2:.2f} {cy2:.2f}",
                    line=dict(color=outline_color, width=theme.card_line_width), layer="above",
                )
            )
        return shapes

    @classmethod
    def _card_port_dot_shapes(cls, node, theme, wired_from):
        """Dots and halo rings of a card's output ports. A ring is drawn
        for wired ports and every constraint port, a dot (category fill)
        for every non-constraint port; strokes are theme.line_color."""
        fill_color, _border_color = cls._node_colors(node.category, theme)
        shapes = []
        is_constraint = node.category == "constraint"
        for port in node.output_ports:
            is_wired = (port.x, port.y) in wired_from
            if is_constraint or is_wired:
                ring = theme.card_port_ring_radius
                shapes.append(
                    dict(
                        type="circle", x0=port.x - ring, y0=port.y - ring, x1=port.x + ring, y1=port.y + ring,
                        fillcolor=theme.background_color, line=dict(color=theme.line_color, width=theme.card_line_width),
                        layer="above",
                    )
                )
            if not is_constraint:
                inner = theme.card_port_inner_radius
                shapes.append(
                    dict(
                        type="circle", x0=port.x - inner, y0=port.y - inner, x1=port.x + inner, y1=port.y + inner,
                        fillcolor=fill_color, line=dict(color=theme.line_color, width=theme.card_line_width),
                        layer="above",
                    )
                )
        return shapes

    @classmethod
    def _edge_shapes(cls, edge, theme, node_boxes):
        """One edge's wire as a bezier path shape without arrowhead - a
        shape, not a trace line, so it can be layered between card bodies
        and port dots."""
        p1, p2 = tuple(edge.points[0]), tuple(edge.points[1])
        return [
            dict(
                type="path", path=cg.wire_path(p1, p2),
                line=dict(color=theme.line_color, width=theme.card_line_width),
                layer="above",
            )
        ]

    @staticmethod
    def _card_annotations(node, theme):
        """The node's uppercase title, rotated 90 degrees and centered."""
        return [
            dict(
                x=node.x,
                y=node.y,
                text=node.label.upper(),
                showarrow=False,
                font=dict(size=theme.card_title_font_size, color=theme.text_color, family=theme.card_title_font_family),
                xanchor="center",
                yanchor="middle",
                textangle=-90,
            )
        ]

    @staticmethod
    def _cluster_annotation(cluster, theme):
        """A cluster's title in its top-left corner, with the same font as
        node titles."""
        pad = theme.card_corner_radius + 0.08
        return dict(
            x=cluster.x0 + pad,
            y=cluster.y0 + pad,
            text=cluster.label,
            showarrow=False,
            font=dict(size=theme.card_title_font_size, color=theme.text_color, family=theme.card_title_font_family),
            xanchor="left",
            yanchor="top",
        )

    @staticmethod
    def _port_annotation(port, theme):
        # The label grows outward: left of input ports, right of output
        # ports.
        xanchor = "right" if port.is_input else "left"
        return dict(
            x=port.x,
            y=port.y,
            text=port.name,
            showarrow=False,
            font=dict(size=theme.card_edge_formula_size, color=theme.text_color),
            xanchor=xanchor,
            yanchor="middle",
            xshift=(-3 if port.is_input else 3),
        )

    @staticmethod
    def _port_annotation_inside(port, theme):
        """A port label growing into the card, clear of the notch.
        xshift/yshift are in pixels (notch radius converted via
        graph_px_per_unit); yshift keeps the label off horizontal wires."""
        clearance_px = theme.card_port_outer_radius * theme.graph_px_per_unit * 1.15
        xanchor = "left" if port.is_input else "right"
        return dict(
            x=port.x,
            y=port.y,
            text=port.name,
            showarrow=False,
            font=dict(size=theme.card_edge_formula_size, color=theme.text_color),
            xanchor=xanchor,
            yanchor="middle",
            xshift=(clearance_px if port.is_input else -clearance_px),
            yshift=8,
        )

    @classmethod
    def _port_owner_colors(cls, layout, theme) -> dict[tuple[float, float], str]:
        """(x, y) -> fill color of the card owning that port, for per-port
        hover label colors."""
        colors: dict[tuple[float, float], str] = {}
        for card in (*layout.nodes, *layout.clusters):
            fill_color, _border = cls._node_colors(card.category, theme)
            for port in card.input_ports + card.output_ports:
                colors[(port.x, port.y)] = fill_color
        return colors

    @classmethod
    def _node_hover_trace(cls, layout, theme):
        return go.Scatter(
            x=[n.x for n in layout.nodes],
            y=[n.y for n in layout.nodes],
            mode="markers",
            marker=dict(size=[max(n.width, n.height) * 35 for n in layout.nodes], opacity=0),
            hovertext=[
                f"{n.label} ({n.kind})" + (f"<br>{n.hover}" if n.hover else "") for n in layout.nodes
            ],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor=[cls._node_colors(n.category, theme)[0] for n in layout.nodes],
                font=dict(color=theme.text_color),
            ),
            showlegend=False,
        )

    @classmethod
    def _port_trace(cls, layout, theme):
        # Invisible hover carrier; the visible port marks are shapes.
        owner_colors = cls._port_owner_colors(layout, theme)
        return go.Scatter(
            x=[p.x for p in layout.ports],
            y=[p.y for p in layout.ports],
            mode="markers",
            marker=dict(size=16, opacity=0),
            hovertext=[f"{p.name} ({'in' if p.is_input else 'out'})" for p in layout.ports],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor=[owner_colors.get((p.x, p.y), theme.node_color_default) for p in layout.ports],
                font=dict(color=theme.text_color),
            ),
            showlegend=False,
        )

    @classmethod
    def _edge_trace(cls, layout, theme):
        # Invisible hover carrier sampled along each wire; the visible wire
        # is a shape. Edges belong to no single card, so their hover label
        # is black on white.
        xs, ys, hovertext = [], [], []
        for edge in layout.edges:
            pts = edge.points
            if len(pts) == 2:
                pts = cg.sample_bezier(tuple(pts[0]), tuple(pts[1]))
            for px, py in pts:
                xs.append(px)
                ys.append(py)
                hovertext.append(edge.label)
        return go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=10, opacity=0),
            hovertext=hovertext,
            hoverinfo="text",
            hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#000000")),
            showlegend=False,
        )

    def update(self, backend_figure, plot, row=None, col=None):
        """Rebuild this artist's shapes, annotations and hover traces.

        See :meth:`~qewton.visualization.renderers.base.Artist.update`.
        """
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
        new_nodes = self._node_hover_trace(layout, plot.theme)
        node_trace.x, node_trace.y = new_nodes.x, new_nodes.y
        node_trace.hovertext = new_nodes.hovertext
        node_trace.marker.size = new_nodes.marker.size
        node_trace.hoverlabel = new_nodes.hoverlabel

        port_trace = backend_figure.data[self.port_trace_idx]
        new_ports = self._port_trace(layout, plot.theme)
        port_trace.x, port_trace.y = new_ports.x, new_ports.y
        port_trace.hovertext = new_ports.hovertext
        port_trace.hoverlabel = new_ports.hoverlabel

        edge_trace = backend_figure.data[self.edge_trace_idx]
        new_edges = self._edge_trace(layout, plot.theme)
        edge_trace.x, edge_trace.y = new_edges.x, new_edges.y
        edge_trace.hovertext = new_edges.hovertext
        edge_trace.hoverlabel = new_edges.hoverlabel
        edge_trace.marker.size = new_edges.marker.size
