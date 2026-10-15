from plotly import graph_objects as go

from qewton.visualization.renderers.plotly import card_geometry as cg
from qewton.visualization.renderers.plotly.common import PlotlyArtist


class NodeLinkArtist(PlotlyArtist):
    """Draws a GraphPlot's node-link layout as Nick's "card" style (see
    card-editor.html): each NodeLayout becomes a beveled rounded-rect card
    with a notch cut into its edge for every port and an uppercase title
    (rotated 90 degrees, centered in the card - see _card_annotations); a
    ClusterBox (an expanded composite's contents) stays the plain dashed
    outline it always was - only leaf/collapsed nodes get the card
    treatment, since a cluster boundary has no single box of its own to draw
    as a card. Type icons are NOT currently drawn (see card_geometry's
    ICON_PATHS docstring) - dropped per Nick's own request.

    Still deliberately not a single Plotly primitive, same reasoning as
    before: every card/notch/crease/wire/label-background is a
    `layout.shapes` entry (cheap regardless of graph size); text is
    `layout.annotations`; and three go.Scatter traces carry everything
    interactive - one invisible marker per node for hover, one invisible
    marker per port for its own hover (the visible socket is the notch
    itself, or a dot/ring shape - never a marker), and one invisible-marker
    trace sampling every edge's own bezier curve purely for its hover text
    (the edge's VISIBLE line is a `wire_path` shape, no arrowhead - see
    _edge_shapes, matching card-editor.html's wires exactly).

    Shape/annotation build order matters here, not just content: Plotly
    draws same-layer shapes in list order (later = on top), so
    `_shapes_and_annotations` builds, in this exact sequence, cluster
    outlines, then every card's own body (polygon + creases, no ports yet),
    then every wire line + its label background, then every port's own
    dot/ring/notch-circle, then every edge's "plugged-in" endpoint dot + its
    label text - the same back-to-front stacking card-editor.html's own
    gNodes/gWires/gPortDots/gDecor layers give it (see that file's
    initLayers() comment).

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
        backend_figure.add_trace(cls._node_hover_trace(layout, plot.theme), row=row, col=col)
        port_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._port_trace(layout, plot.theme), row=row, col=col)
        edge_idx = len(backend_figure.data)
        backend_figure.add_trace(cls._edge_trace(layout, plot.theme), row=row, col=col)

        backend_figure.update_xaxes(visible=False, row=row, col=col)
        # autorange="reversed": GraphLayout places successive nodes within a
        # rank at INCREASING y (plots/graph/layout.py's own `y = offset_y +
        # i*row_height`), i.e. it already assumes "y grows downward" as the
        # natural top-to-bottom reading order - same as the card geometry's
        # own SVG-style convention (ported from card-editor.html, an SVG
        # canvas). Plotly's default y-axis grows upward instead, which, left
        # unreversed, would flip every card's own top/bottom (notch/bevel
        # orientation mirrored) even though the plain rectangles the old
        # styling drew happened to look identical either way.
        backend_figure.update_yaxes(visible=False, row=row, col=col, scaleanchor="x")

        x0, x1, y0, y1 = cls._layout_bounds(layout)
        backend_figure.update_xaxes(range=[x0, x1], row=row, col=col)
        backend_figure.update_yaxes(range=[y1, y0], row=row, col=col)  # descending: same flip as autorange="reversed"

        # Fixes the figure's own pixel size to plot.theme.graph_px_per_unit
        # times the layout's own data-unit extent - but only for a
        # standalone (non-subplot) figure the caller hasn't already sized,
        # so this never fights an explicit width/height or a faceted/multi-
        # panel layout where one GraphPlot sizing the whole figure from just
        # its own content would be wrong. See graph_px_per_unit's own
        # docstring in themes/base.py for why this matters at all: without
        # it, the exact same data-unit-sized card lands at a different
        # on-screen PIXEL size depending on whatever container ends up
        # displaying the (otherwise unsized, "responsive") figure, while a
        # card's title/port-label font sizes are fixed in points - so the
        # same graph could look right in one notebook cell and comically
        # oversized-text in another, purely from the container's width, not
        # anything about the graph itself.
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
        """(x0, x1, y0, y1) covering every node AND cluster card, padded -
        computed directly from the layout's own known geometry rather than
        relying on Plotly's trace-based autorange, which (via the node hover
        trace's deliberately oversized invisible marker - see
        _node_hover_trace) was only ever an approximation anyway."""
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
        """(fill, border) for a node/cluster category.

        theme.node_color_by_type now carries BOTH per category (see
        themes/base.py's QEWTON_NODE_COLORS-sourced dict - each entry is a
        {"fill": ..., "border": ...} pair), the SAME pair in every theme:
        the old light/dark role-swap (a category's fill/outline literally
        traded places in dark mode, via the now-removed
        theme.card_face_color_dark escape hatch) is gone entirely. A
        theme's light/dark difference lives purely in its general figure-
        chrome fields now (background_color/text_color/line_color), not in
        any per-category swap.

        Only 3 roles are visually distinguished at all - constraint,
        datanode, graphnode. A node whose own _node_category() falls
        through to "default" (anything that's neither a Constraint, a
        DataNode, nor a GraphNode) reads as a graphnode card: under the OLD
        flat-hex palette this bucket's color happened to be numerically
        identical to the graphnode blue, so it was never visually distinct
        from a model/algorithm card to begin with - falling back to that
        SAME theme.node_color_by_type["graphnode"] pair here (rather than
        theme.node_color_default, a since-introduced and genuinely distinct
        brown swatch) keeps that unchanged now that the coincidence no
        longer holds numerically on its own. theme.node_color_default is
        kept only as a last-resort fallback for the - currently impossible -
        case where even "graphnode" itself is missing from the dict."""
        pair = theme.node_color_by_type.get(category) or theme.node_color_by_type.get("graphnode")
        if pair is not None:
            return pair["fill"], pair["border"]
        return theme.node_color_default, theme.line_color

    @classmethod
    def _owned_port_positions(cls, layout) -> set[tuple[float, float]]:
        """(x, y) of every port that belongs to a card - a leaf NodeLayout's
        own, or a ClusterBox's boundary ports (a composite is now ALSO a
        card - see that class's own docstring - so its ports get the exact
        same notch + dot/ring treatment, not the old plain-circle
        fallback)."""
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

        # Wired output ports get a halo ring behind their dot (see
        # buildPortDots in card-editor.html) - "wired" here means some
        # edge's own start point lands exactly on that port's anchor, which
        # is true whenever GraphLayout placed an EdgeLayout there (it reuses
        # the very same port_anchor dict it built the port's own (x, y)
        # from, so this is an exact float match, not a tolerance check).
        wired_from = {tuple(e.points[0]) for e in layout.edges}
        card_boxes = [
            (c.x - c.width / 2, c.y - c.height / 2, c.x + c.width / 2, c.y + c.height / 2) for c in cards
        ]

        # Layer 1: every card's own body (polygon + creases) - a leaf node
        # AND a ClusterBox alike, both drawn by the exact same
        # _card_body_shapes (they share the same x/y/width/height/category/
        # input_ports/output_ports interface - see ClusterBox's docstring),
        # a cluster first so its inner nodes draw (and so read as "inside
        # it") right after, in this same pass. See the class docstring for
        # why this whole method builds shapes in a fixed back-to-front
        # sequence rather than grouping them by node.
        for cluster in layout.clusters:
            shapes += cls._card_body_shapes(cluster, theme)
            annotations.append(cls._cluster_annotation(cluster, theme))
        for node in layout.nodes:
            shapes += cls._card_body_shapes(node, theme)
            annotations += cls._card_annotations(node, theme)

        # Layer 2: wire lines + their dynamic-data-config label backgrounds.
        for edge in layout.edges:
            shapes += cls._edge_shapes(edge, theme, card_boxes)

        # Layer 3: every port's own visible mark - always a card's own
        # notch + dot/ring now (see _owned_port_positions), plus its name
        # annotation INSIDE the card when called for. A cluster's own
        # boundary ports used to always get one regardless of
        # show_port_names (reasoning at the time: no title-less box
        # otherwise distinguishes them) - but that made show_port_names
        # behave inconsistently between a leaf node and a GraphNode's own
        # boundary, which is confusing from the caller's side: one flag,
        # same effect on every card, a cluster included.
        for card in cards:
            shapes += cls._card_port_dot_shapes(card, theme, wired_from)
        if show_port_names:
            for card in cards:
                for port in card.input_ports + card.output_ports:
                    if port.show_label:
                        annotations.append(cls._port_annotation_inside(port, theme))

        # Fallback plain-circle style for any port that somehow isn't part
        # of a card (shouldn't happen via GraphLayout today - every port is
        # either a leaf node's own or a ClusterBox's boundary port - kept
        # only so an unexpected future port kind degrades visibly instead
        # of silently not being drawn at all).
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

        # Layer 4: each edge's "plugged-in" endpoint dot + its visible
        # label text, drawn last so both sit above the port dots/rings -
        # same stacking as card-editor.html's own gDecor layer.
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
        """theme.background_color, unless `mid` lands inside some card's own
        area - almost always a ClusterBox's interior, since an edge fully
        contained inside an expanded composite (e.g. the inner wire between
        two of its own inner nodes) has nowhere else for its label to go -
        in which case that card's own face color, so the label's background
        rect blends into whatever it's actually sitting on top of instead of
        punching a mismatched plain-background rectangle into a colored
        card. Prefers the SMALLEST matching card (by area) so a nested
        cluster's own background wins over an outer one it also happens to
        sit inside."""
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
        """The beveled card body alone (back + front polygon, crease lines)
        - NOT its ports' dot/ring, which `_card_port_dot_shapes` draws
        separately and later, so wires can be sandwiched in between (see the
        class docstring on shape build order).

        The body's own OUTLINE (back/front polygon stroke + crease lines)
        is always theme.line_color, same as every other card regardless of
        category - only the FILL is category-specific. `_node_colors`'s
        second ("border") shade is deliberately NOT used here: it's reserved
        for the port dot/ring accent (`_card_port_dot_shapes`), where it
        reads as that card's own color family; stroking the whole card body
        with it as well would make a card's outline (and the crease lines
        that are really just a continuation of that same outline) stop
        reading as a plain black line - which is what actually happened
        here for a while after node_color_by_type gained its own per-
        category border field, until Nick caught it."""
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
        """A node's own output ports' visible dot (+ optional halo ring) -
        split out from `_card_body_shapes` so the overall shape list can
        put every card BODY down first, then every wire, then every port
        dot on top of the wires (see the class docstring).

        Matches the Qewton Karten-Editor artifact's own buildPortDots()
        exactly: the dot's FILL is the card's own category color
        (`_node_colors`'s fill), but both the dot's and the ring's STROKE
        are always `var(--ink-line)` there - i.e. theme.line_color, the
        same plain black every card body/crease outline uses - never the
        category's own second ("border") shade. That second shade is for
        the card body's darker BACK-bevel fill (see `_card_body_shapes`),
        not a stroke anywhere - using it as this dot/ring's outline instead
        was a mistake made while adapting to the new fill+border theme
        pair, now reverted."""
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
        """One edge's visible wire - a plain bezier stroke with no
        arrowhead, exactly card-editor.html's own wirePath() styling (see
        that file's renderCanvasOnly()) - as a Plotly path shape rather than
        a Scatter line, so it can be layered precisely between card bodies
        and port dots instead of being stuck at "above every trace"."""
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
        """The card's own title - rotated 90 degrees (vertical) and
        centered in the box, so a node's height (not width) is what needs
        to grow for a long name (see plots/graph/layout.py's _node_size) -
        card-editor's node.formula (a second, centered LaTeX row) has no
        equivalent yet in Node's own data model (just `.name`), so it's
        left for a future pass rather than guessed at here."""
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
        """A ClusterBox's own title, in its top-left corner, inside the
        card - NOT rotated/centered like a leaf node's (see
        _card_annotations), since the center of a cluster's card is already
        taken up by its own inner nodes and there's no "middle" to put a
        title in. Everything else about it - font family/size/color - is
        the SAME as a leaf node's own title (theme.card_title_font_size/
        _font_family/text_color), not a separate hardcoded size, so a
        cluster's title scales exactly in step with every other card's when
        theme.graph_px_per_unit changes; a hardcoded font size here was
        exactly the kind of per-node special case that made text sizing
        read as inconsistent across a graph."""
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
            font=dict(size=theme.card_edge_formula_size, color=theme.text_color),
            xanchor=xanchor,
            yanchor="middle",
            xshift=(-3 if port.is_input else 3),
        )

    @staticmethod
    def _port_annotation_inside(port, theme):
        """Same label as `_port_annotation`, but anchored to grow INWARD
        (into the card) rather than away from it - an input port's label
        starts right at its own x and extends rightward, an output port's
        mirrors that leftward, both just clearing the port's own notch.
        `xshift`/`yshift` are in PIXELS (unlike every other position here,
        which is in data units), so the notch radius - a data-unit size -
        has to be converted via graph_px_per_unit first.

        The small `yshift` lifts the label just clear of its own port's y -
        without it, a port whose wire happens to leave exactly horizontally
        (common for a cluster's own boundary port, wired straight across to
        an inner node at the same height) draws the label right on top of
        that wire line, same y as both."""
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
        """(x, y) -> that port's OWNING card's own type color - a leaf
        node's or a ClusterBox's, whichever actually has this port in its
        input_ports/output_ports. Used so a port's (or an edge leaving one)
        hover tooltip reads as "this belongs to that colored card" instead
        of Plotly's own default per-trace color cycle, which is what made
        hover colors look "irgendwie random" (arbitrary) in the first place
        - every hover trace here is ONE shared go.Scatter covering every
        port/edge in the whole graph, so without this there was never any
        per-point color at all, just whatever default Plotly happened to
        assign trace-wide."""
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
        # Invisible - every port's visible mark is now a shape (a notch cut
        # into its node's card, or a dot/ring/old-style circle; see
        # _shapes_and_annotations). This trace exists purely so hovering a
        # port's position still shows its name/direction.
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
        # Purely a hover carrier now - the visible wire is a `wire_path`
        # shape (see _edge_shapes), not this trace, so there's no line/
        # arrowhead styling left to set here, just invisible markers
        # sampled along the same bezier curve the shape draws. Unlike a
        # node's or a port's own hover (which reads as "belonging to" that
        # card's color), an edge belongs to no single node - it only ever
        # connects two of them - so its hover stays the plain black-on-white
        # every other tooltip used before the per-category hover coloring
        # was added, rather than borrowing its source port's owning card's
        # color.
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
