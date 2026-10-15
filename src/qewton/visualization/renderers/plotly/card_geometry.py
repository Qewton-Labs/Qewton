"""Pure geometry helpers for drawing a GraphPlot's nodes as beveled "cards",
in the exact visual style of Nick's card-editor prototype (rounded rect body,
a 3D bevel/depth edge, semicircular port notches, and a type icon) - ported
1:1 from that prototype's JS into Plotly path-shape strings.

Nothing here touches Plotly or qewton's own graph classes - it only turns
numbers (positions, sizes, port counts) into SVG path "d" strings and point
lists that `renderers/plotly/graph.py` feeds into `go.layout.Shape` /
`go.Scatter`. That split mirrors GraphLayout (renderer-agnostic layout) vs.
NodeLinkArtist (Plotly-specific drawing) one level up.

The bevel shape itself is a real 2D polygon union (front face, a copy offset
by `depth` along both axes, and small triangular/bridge pieces connecting
them at the corners and at each port notch's tangent points) - the same
construction as the original prototype's use of polybooljs, done here with
shapely. A hand-derived per-case union was tried first in the prototype and
missed edge cases; reusing a real boolean-geometry library avoids that.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

SQ = 0.70710678  # sin/cos(45 deg) - the 45-degree point on a corner's arc


# ---------------------------------------------------------------------------
# Low-level polygon construction (ported from card-editor.html's
# arc()/buildFlatPolygon()/buildFlatEdge()/bridgeTriangle()/buildBackPolygon()/
# buildCreaseLines() - see that file for the original, heavily-commented JS).
# ---------------------------------------------------------------------------


def arc(
    cx: float, cy: float, r: float, a0: float, a1: float, n: int = 16,
    registry: dict | None = None,
) -> list[tuple[float, float]]:
    """Samples `n+1` points of a circular arc from angle a0 to a1 (radians).

    When `registry` is given (a plain dict, mutated in place), every sampled
    point is also tagged with its originating circle's (cx, cy, r) - keyed
    by the point's own rounded coordinate, the same rounding `_round_poly`
    applies before the shapely union. `path_from_poly` later looks a point
    back up in this registry to tell "this point is part of a true circular
    arc" from "this is a straight/bridge vertex", and collapses a run of
    consecutive same-circle points into one real SVG `A` command - see its
    own docstring for why that round-trip survives the union untouched."""
    pts = [
        (cx + r * math.cos(a0 + (a1 - a0) * i / n), cy + r * math.sin(a0 + (a1 - a0) * i / n))
        for i in range(n + 1)
    ]
    if registry is not None:
        sig = (round(cx, 6), round(cy, 6), round(r, 6))
        for px, py in pts:
            registry[(round(px, 6), round(py, 6))] = sig
    return pts


@dataclass
class Notch:
    cy: float
    r: float


def _build_flat_edge(
    x_edge: float, y_a: float, y_b: float, notches: list[Notch], entry_angle: float, arc_n: int,
    registry: dict | None = None,
) -> list[tuple[float, float]]:
    direction = 1 if y_b >= y_a else -1
    ordered = sorted(notches, key=lambda nt: direction * nt.cy)
    pts: list[tuple[float, float]] = []
    cur = y_a
    for nt in ordered:
        exit_y = nt.cy - nt.r * math.sin(entry_angle)
        pts.append((x_edge, cur))
        pts.extend(arc(x_edge, nt.cy, nt.r, entry_angle, entry_angle - math.pi, arc_n, registry=registry))
        cur = exit_y
    pts.append((x_edge, cur))
    pts.append((x_edge, y_b))
    return pts


def build_flat_polygon(
    x: float, y: float, w: float, h: float, r: float,
    left_notches: list[Notch], right_notches: list[Notch], arc_n: int = 16,
    registry: dict | None = None,
) -> list[tuple[float, float]]:
    """The card's own flat silhouette: a rounded rect with a semicircular
    notch cut into the left edge per input port and the right edge per
    output port (or, for a loss-ring node, one reserved right-side slot -
    see `node_out_count_for_layout`).

    Pass a (possibly shared) `registry` dict to have every corner/notch arc
    point tagged with its originating circle, for `path_from_poly` to later
    render as a true SVG arc instead of straight segments - see `arc()`."""
    x2, y2 = x + w, y + h
    poly: list[tuple[float, float]] = [(x + r, y), (x2 - r, y)]
    poly += arc(x2 - r, y + r, r, -math.pi / 2, 0, arc_n, registry=registry)
    poly += _build_flat_edge(x2, y + r, y2 - r, right_notches, -math.pi / 2, arc_n, registry=registry)
    poly += arc(x2 - r, y2 - r, r, 0, math.pi / 2, arc_n, registry=registry)
    poly += [(x2 - r, y2), (x + r, y2)]
    poly += arc(x + r, y2 - r, r, math.pi / 2, math.pi, arc_n, registry=registry)
    poly += _build_flat_edge(x, y2 - r, y + r, left_notches, math.pi / 2, arc_n, registry=registry)
    poly += arc(x + r, y + r, r, math.pi, 1.5 * math.pi, arc_n, registry=registry)
    return poly


def _bridge_triangle(ax: float, ay: float, depth: float, vertical: bool) -> list[tuple[float, float]]:
    bx, by = ax + depth, ay + depth
    cx, cy = (ax, by) if vertical else (bx, ay)
    return [(ax, ay), (bx, by), (cx, cy)]


def _round_poly(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # Snaps to 1e-6 before the union - two points meant to coincide (e.g. a
    # corner's 45-degree crossing, computed twice via independent cos/sin
    # calls) can differ by ~1e-14 of float noise, which shapely would
    # otherwise see as a genuine sliver rather than the same point.
    return [(round(px, 6), round(py, 6)) for px, py in poly]


def _poly_union_all(polys: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # buffer(0) is the standard shapely fix for a polygon that's very
    # slightly self-touching/invalid (e.g. a port notch whose radius leaves
    # only a hair's breadth of clearance to the corner arc) - it rebuilds
    # the same shape through GEOS's own validity-repair pass, so a union
    # that would otherwise raise "TopologyException: side location
    # conflict" on tightly-packed ports instead just renders them touching.
    shapes = [Polygon(_round_poly(p)).buffer(0) for p in polys if len(p) >= 3]
    merged = unary_union(shapes)
    if merged.geom_type == "Polygon":
        return list(merged.exterior.coords)[:-1]
    # Degenerate parameters could in principle split the union into several
    # pieces - the card silhouette is whichever one has the most vertices
    # (bridge triangles are always tiny), same fallback as the original JS.
    largest = max(merged.geoms, key=lambda g: len(g.exterior.coords))
    return list(largest.exterior.coords)[:-1]


def build_back_polygon(
    x: float, y: float, w: float, h: float, r: float,
    left_notches: list[Notch], right_notches: list[Notch], depth: float, arc_n: int = 16,
) -> tuple[list[tuple[float, float]], dict]:
    """The beveled "back" silhouette: the front polygon, a copy shifted by
    `depth` along both x and y, and small bridge triangles at the two
    diagonal corners and at each port notch's outer tangent point - unioned
    into one outline exactly like the front+back+bridges construction in
    card-editor.html's buildBackPolygon().

    Returns (polygon, registry) - the registry carries every corner/notch
    arc's provenance through to the union's own output points (a shifted
    copy of the front polygon's own registry, re-centered by `depth` to
    match where those points actually land after the shift), for
    `path_from_poly` to render true arcs instead of a faceted outline."""
    x2, y2 = x + w, y + h
    front_registry: dict = {}
    front = build_flat_polygon(x, y, w, h, r, left_notches, right_notches, arc_n, registry=front_registry)
    back = []
    registry = dict(front_registry)
    for px, py in front:
        bx, by = px + depth, py + depth
        back.append((bx, by))
        sig = front_registry.get((round(px, 6), round(py, 6)))
        if sig is not None:
            cx, cy, rr = sig
            registry[(round(bx, 6), round(by, 6))] = (cx + depth, cy + depth, rr)
    polys = [front, back]
    polys.append(_bridge_triangle(x2 - r + r * SQ, y + r - r * SQ, depth, True))  # top-right corner
    polys.append(_bridge_triangle(x + r - r * SQ, y2 - r + r * SQ, depth, False))  # bottom-left corner
    for nt in right_notches:
        polys.append(_bridge_triangle(x2, nt.cy + nt.r, depth, True))  # output notch: bottom tangent
    for nt in left_notches:
        polys.append(_bridge_triangle(x, nt.cy - nt.r, depth, False))  # input notch: top tangent
    return _poly_union_all(polys), registry


def build_crease_lines(
    x: float, y: float, w: float, h: float, r: float,
    left_notches: list[Notch], right_notches: list[Notch], depth: float,
) -> list[tuple[float, float, float, float]]:
    """The thin diagonal "3D edge" stroke at each output notch's TOP
    tangent only - the one place nothing else already reads as a visible
    edge there.

    An earlier version of this also redrew a line at the 2 bridged corners
    and each output notch's BOTTOM tangent, following card-editor.html's own
    reasoning that the union "swallows" those as interior edges needing a
    redraw - but that reasoning was about polybooljs's union specifically;
    shapely's own union of the same front/back/bridge pieces (see
    build_back_polygon) already leaves those as part of the back
    silhouette's own VISIBLE boundary stroke, so redrawing them here just
    doubled the line. Only the un-bridged top tangent (no bridge triangle
    connects it at all - see build_back_polygon) has no boundary stroke of
    its own there, so it's the one place this is still needed. Deliberately
    none on the input side either, same as before."""
    lines = []
    for nt in right_notches:
        ax, ay = x + w, nt.cy - nt.r
        lines.append((ax, ay, ax + depth, ay + depth))
    return lines


def darken_hex(hex_color: str, factor: float) -> str:
    m = re.match(r"^#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$", hex_color)
    if not m:
        return hex_color
    channels = [max(0, min(255, round(int(m.group(i), 16) * factor))) for i in (1, 2, 3)]
    return "#" + "".join(f"{c:02x}" for c in channels)


def port_layout(n: int, card_h: float, corner_r: float, outer_r: float, min_gap: float, min_margin: float):
    """n ports, evenly spaced with a minimum gap/margin, any extra height
    distributed across every gap+margin slot so ports spread out
    symmetrically instead of staying packed at the minimum. Returns
    (list of center-y offsets from the card's own top edge, minimum card
    height this port count needs)."""
    if n <= 0:
        return [], 2 * corner_r
    usable = card_h - 2 * corner_r
    min_usable = 2 * min_margin + n * 2 * outer_r + (max(0, n - 1) * min_gap)
    min_height = 2 * corner_r + min_usable
    extra = max(0.0, usable - min_usable)
    slots = n + 1
    extra_per_slot = extra / slots
    margin = min_margin + extra_per_slot
    gap = min_gap + extra_per_slot
    cys = []
    cy = corner_r + margin + outer_r
    for _ in range(n):
        cys.append(cy)
        cy += 2 * outer_r + gap
    return cys, min_height


#: Largest single sub-arc (radians) _bezier_arc_segments will approximate
#: with one cubic bezier - a standard quarter-circle is the usual rule of
#: thumb for staying visually indistinguishable from a true circle (a full
#: 180-degree notch is split into two of these, a 90-degree corner stays
#: one piece).
_MAX_BEZIER_ARC = math.pi / 2 + 1e-6


def _bezier_arc_segments(cx: float, cy: float, r: float, a_start: float, total: float):
    """Splits a circular arc (center (cx,cy), radius r, from angle a_start
    sweeping `total` radians, signed) into one cubic-bezier-per-<=90-degree-
    piece, each as ((c1x,c1y), (c2x,c2y), (ex,ey)) - the "A"-free way to
    draw a true circular arc as a Plotly path shape, since Plotly's `d`
    parser only understands M/L/Q/C/Z, never the SVG `A` command (confirmed
    empirically: an "A ..." token in a shape path silently renders as a
    straight chord instead of an arc, not an error - the bug behind the
    first version of this function). A cubic bezier's standard circular-arc
    approximation (the "kappa" construction) is accurate to a fraction of a
    percent of the radius for any span up to 90 degrees, i.e. visually
    exact at any zoom level - genuinely smooth, not a sampled polyline."""
    n = max(1, math.ceil(abs(total) / _MAX_BEZIER_ARC))
    step = total / n
    kappa = 4.0 / 3.0 * math.tan(step / 4.0)
    segments = []
    a0 = a_start
    for _ in range(n):
        a1 = a0 + step
        p0x, p0y = cx + r * math.cos(a0), cy + r * math.sin(a0)
        p3x, p3y = cx + r * math.cos(a1), cy + r * math.sin(a1)
        c1x, c1y = p0x + kappa * r * -math.sin(a0), p0y + kappa * r * math.cos(a0)
        c2x, c2y = p3x - kappa * r * -math.sin(a1), p3y - kappa * r * math.cos(a1)
        segments.append(((c1x, c1y), (c2x, c2y), (p3x, p3y)))
        a0 = a1
    return segments


def path_from_poly(poly: list[tuple[float, float]], registry: dict | None = None, close: bool = True) -> str:
    """A Plotly `shape(type="path")` "d" string for the card silhouette.

    Without a `registry`, every vertex becomes a straight `L` segment (used
    for anything with no arcs, e.g. the crease lines). With one (as
    returned by `build_flat_polygon`/`build_back_polygon`), a RUN of
    consecutive points that share the same originating circle - looked up
    by each point's own rounded coordinate - is collapsed into one or more
    true cubic-bezier arc segments instead (see `_bezier_arc_segments`), so
    a corner or port notch stays a real curve under arbitrary zoom rather
    than the old polyline approximation (sampled arc points joined by
    straight segments, which looked "pixelig"/faceted once zoomed in - the
    exact complaint this was built to fix). Only the run's two true
    endpoints and the circle's own (cx, cy, r) are used to rebuild the
    curve - the original in-between sample points are discarded entirely,
    not stitched together, so sampling density no longer limits zoom
    fidelity at all.

    This lookup survives `_poly_union_all` because the back polygon's front/
    back/bridge pieces only ever TOUCH each other at single points (shared
    boundary), never overlap in area - GEOS's union of touching-not-
    overlapping polygons re-stitches their existing boundary vertices rather
    than synthesizing new ones, so an output vertex's rounded coordinate
    still matches one of the input vertices the registry was built from (a
    `buffer(0)` repair pass changing a coordinate by more than the 1e-6
    rounding tolerance would be the one way this silently degrades a vertex
    back to a plain `L` segment instead of a found arc - harmless, since the
    dense arc_n=20 sampling still makes that segment short enough to look
    smooth, just not a true bezier arc there)."""
    if not poly:
        return ""
    registry = registry or {}

    def sig_at(idx):
        x, y = poly[idx]
        return registry.get((round(x, 6), round(y, 6)))

    n = len(poly)
    d = f"M {poly[0][0]:.3f},{poly[0][1]:.3f}"
    i = 1
    while i < n:
        sig = sig_at(i)
        if sig is not None and sig == sig_at(i - 1):
            run_start = i - 1
            j = i
            while j < n and sig_at(j) == sig:
                j += 1
            cx, cy, r = sig
            angles = [math.atan2(poly[k][1] - cy, poly[k][0] - cx) for k in range(run_start, j)]
            total = 0.0
            for k in range(1, len(angles)):
                delta = angles[k] - angles[k - 1]
                while delta > math.pi:
                    delta -= 2 * math.pi
                while delta < -math.pi:
                    delta += 2 * math.pi
                total += delta
            for (c1x, c1y), (c2x, c2y), (ex, ey) in _bezier_arc_segments(cx, cy, r, angles[0], total):
                d += f" C {c1x:.4f},{c1y:.4f} {c2x:.4f},{c2y:.4f} {ex:.4f},{ey:.4f}"
            # Snap to the run's own exact end point rather than whatever the
            # bezier math landed on (sub-micron float noise at worst) - the
            # polygon's recorded endpoint is what the next L segment must
            # butt up against exactly.
            ex, ey = poly[j - 1]
            d = d.rsplit(" ", 1)[0] + f" {ex:.3f},{ey:.3f}"
            i = j
        else:
            x, y = poly[i]
            d += f" L {x:.3f},{y:.3f}"
            i += 1
    if close:
        d += " Z"
    return d


#: Minimum bezier-handle length for wire_control_points, in GraphLayout's
#: own data units (COLUMN_GAP=1.4, MIN_WIDTH=1.5 - see plots/graph/
#: layout.py) - NOT card-editor.html's "40" (a pixel-canvas value, 25-100x
#: too large at this scale: it would make the handle dominate the whole
#: curve instead of just rounding it, bowing edges far outside the plot).
MIN_WIRE_HANDLE = 0.3


def wire_control_points(p1: tuple[float, float], p2: tuple[float, float]):
    dx_span = max(MIN_WIRE_HANDLE, abs(p2[0] - p1[0]) * 0.4)
    return (p1[0] + dx_span, p1[1]), (p2[0] - dx_span, p2[1])


def wire_path(p1: tuple[float, float], p2: tuple[float, float]) -> str:
    """A cubic-bezier Plotly path - same horizontal-handle construction as
    card-editor.html's wirePath(), so edges curve the same way."""
    (c1x, c1y), (c2x, c2y) = wire_control_points(p1, p2)
    return f"M {p1[0]:.2f} {p1[1]:.2f} C {c1x:.2f} {c1y:.2f}, {c2x:.2f} {c2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"


def sample_bezier(p1: tuple[float, float], p2: tuple[float, float], n: int = 24) -> list[tuple[float, float]]:
    """Samples the same cubic bezier `wire_path` draws, as plain points -
    for the interactive go.Scatter edge trace (hover only, now - the wire
    itself is drawn as a `wire_path` shape with no arrowhead, matching
    card-editor.html exactly), which needs real (x, y) samples rather than
    an SVG path string."""
    (c1x, c1y), (c2x, c2y) = wire_control_points(p1, p2)
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p1[0] + 3 * mt**2 * t * c1x + 3 * mt * t**2 * c2x + t**3 * p2[0]
        y = mt**3 * p1[1] + 3 * mt**2 * t * c1y + 3 * mt * t**2 * c2y + t**3 * p2[1]
        pts.append((x, y))
    return pts


def _bezier_point(p1: tuple[float, float], p2: tuple[float, float], t: float) -> tuple[float, float]:
    (c1x, c1y), (c2x, c2y) = wire_control_points(p1, p2)
    mt = 1 - t
    x = mt**3 * p1[0] + 3 * mt**2 * t * c1x + 3 * mt * t**2 * c2x + t**3 * p2[0]
    y = mt**3 * p1[1] + 3 * mt**2 * t * c1y + 3 * mt * t**2 * c2y + t**3 * p2[1]
    return x, y


def _point_in_box(pt: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def edge_label_point(
    p1: tuple[float, float], p2: tuple[float, float], node_boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """Picks a point along the same bezier `wire_path` draws, for an edge's
    dynamic-data-config label to sit at. Prefers the geometric midpoint
    (t=0.5), but for a long diagonal wire that happens to pass straight
    through an unrelated third node's body (common once a graph has several
    nodes), tries a few other points along the same curve until it finds one
    that doesn't land inside any node's box - the same fallback cascade as
    card-editor.html's own edgeLabelPoint()."""
    for t in (0.5, 0.35, 0.65, 0.22, 0.78):
        pt = _bezier_point(p1, p2, t)
        if not any(_point_in_box(pt, box) for box in node_boxes):
            return pt
    return _bezier_point(p1, p2, 0.5)


#: Per-character size for the small background box drawn behind a permanent
#: edge-label annotation (see NodeLinkArtist's edge-label shapes) - smaller
#: than an early version of this, and now sized along HEIGHT rather than
#: width: the label text itself is drawn rotated 90 degrees (same reasoning
#: as a card's own title - see NodeLinkArtist._card_annotations), so a long
#: dynamic-data-config string like "[x_i, y_i]" needs vertical room, not
#: horizontal, to stay compact next to a narrow wire.
EDGE_LABEL_CHAR_SIZE = 0.05
EDGE_LABEL_PAD = 0.045
EDGE_LABEL_THICKNESS = 0.11  # the box's now-fixed WIDTH (the rotated text's own thickness)


def edge_label_box(label: str, cx: float, cy: float) -> tuple[float, float, float, float]:
    half_h = max(0.06, len(label) * EDGE_LABEL_CHAR_SIZE * 0.5) + EDGE_LABEL_PAD
    half_w = EDGE_LABEL_THICKNESS / 2
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


# ---------------------------------------------------------------------------
# Type icons - the exact "d"/viewBox pairs from card-editor.html's own
# buildIconDefs(), one per node category. A non-rotating, non-skewing
# (scale + translate only) remap into the node's own icon box, so curves
# stay exact - see transform_svg_path().
#
# NOT currently drawn: NodeLinkArtist no longer calls ICON_PATHS/
# transform_svg_path (dropped per Nick's own request - they weren't
# rendering well yet). Left defined here, unused, rather than deleted, so a
# future pass can wire them back in once that's worth revisiting.
# ---------------------------------------------------------------------------

ICON_PATHS: dict[str, tuple[str, str]] = {
    "datanode": (
        "12.07 8.51 4.28 4.28",
        "m 14.016189,8.7108982 c -0.481735,0 -0.893307,0.07583 -1.234417,0.227792 -0.341085,0.151962 -0.511532,0.334715 -0.511532,0.548109 v 1.9398648 c 0,0.145495 0.06614,0.267622 0.198689,0.366241 0.132588,0.09861 0.293138,0.178271 0.481734,0.239158 0.188642,0.06091 0.386444,0.104917 0.593364,0.131854 0.206922,0.02693 0.385384,0.03994 0.535153,0.03889 -0.02483,-0.05066 -0.04624,-0.102997 -0.06454,-0.156859 -0.01837,-0.0539 -0.03248,-0.110034 -0.04209,-0.168234 -0.151948,-0.003 -0.305972,-0.01792 -0.461705,-0.04433 -0.15573,-0.02636 -0.298086,-0.05984 -0.427367,-0.100257 -0.129369,-0.04041 -0.238858,-0.08685 -0.328273,-0.139128 -0.08948,-0.05228 -0.143394,-0.103732 -0.161673,-0.15436 v -0.567436 c 0.168111,0.097 0.362154,0.172695 0.581997,0.227109 0.219863,0.05442 0.48381,0.08388 0.792031,0.08821 h 0.08096 c 0.02483,-0.06036 0.05294,-0.118563 0.08477,-0.174591 0.0317,-0.05607 0.06869,-0.108893 0.110698,-0.158459 -0.03985,0.0043 -0.07829,0.0071 -0.115475,0.0082 -0.03706,0.0014 -0.0758,0.0016 -0.115712,0.0016 -0.262951,0 -0.522425,-0.02846 -0.778416,-0.08503 -0.255931,-0.05658 -0.469484,-0.144538 -0.640859,-0.264167 V 9.8505712 c 0.173514,0.1024 0.384649,0.1815348 0.633618,0.2375658 0.248951,0.05603 0.510822,0.08411 0.785657,0.08411 0.271594,0 0.531412,-0.02808 0.779304,-0.08411 0.247889,-0.05603 0.463345,-0.1351658 0.646584,-0.2375658 v 0.5626558 c 0.05709,0.0097 0.11271,0.02371 0.166622,0.04206 0.05385,0.01831 0.106182,0.03977 0.156855,0.06457 V 9.4868322 c 0,-0.21339 -0.170662,-0.396143 -0.51175,-0.548105 -0.341109,-0.151965 -0.752466,-0.227794 -1.234224,-0.227794 z m 0,0.323274 c 0.280239,0 0.56603,0.04316 0.857506,0.129358 0.29156,0.08623 0.472378,0.179391 0.542454,0.279621 -0.06792,0.10023 -0.247825,0.193263 -0.539905,0.278941 -0.292079,0.08568 -0.578734,0.128444 -0.860055,0.128444 -0.292056,0 -0.580095,-0.04174 -0.864072,-0.125261 -0.283997,-0.08354 -0.463822,-0.177593 -0.539256,-0.282124 0.07543,-0.105631 0.256275,-0.200199 0.542434,-0.283718 0.286116,-0.08351 0.573159,-0.125261 0.860894,-0.125261 z m 1.196021,1.6811548 c -0.25755,0 -0.478212,0.09126 -0.661993,0.273942 -0.183735,0.18268 -0.275527,0.402821 -0.275527,0.660409 0,0.25973 0.09179,0.481445 0.275527,0.665187 0.183781,0.183776 0.404443,0.275763 0.661993,0.275763 0.257593,0 0.478278,-0.09199 0.662013,-0.275763 0.183778,-0.183742 0.275765,-0.405457 0.275765,-0.665187 0,-0.257588 -0.09199,-0.477729 -0.275765,-0.660409 -0.183735,-0.182678 -0.40442,-0.273942 -0.662013,-0.273942 z m -0.08069,0.374882 h 0.16163 v 0.818183 l 0.365332,-0.365334 0.113205,0.11299 -0.559481,0.559476 -0.559245,-0.559476 0.113012,-0.11299 0.365547,0.365334 z",
    ),
    "graphnode": (
        "0 -960 960 960",
        "m 380,-360 q 42,0 71,-29 l 160,-160 q 29,-29 29,-71 0,-42 -29,-71 -29,-29 -71,-29 -42,0 -71,29 -37,-13 -73,-6 -36,7 -61,32 -25,25 -32,61 -7,36 6,73 -29,29 -29,71 0,42 29,71 29,29 71,29 z M 240,-80 V -252 Q 183,-304 151.5,-373.5 120,-443 120,-520 q 0,-150 105,-255 105,-105 255,-105 125,0 221.5,73.5 Q 798,-733 827,-615 l 54,213.67 q 4.33,15.66 -5.67,28.5 -10,12.83 -26.66,12.83 H 760 v 133.33 q 0,27.5 -19.58,47.09 Q 720.83,-160 693.33,-160 H 600 v 80 h -66.67 v -146.67 h 160 v -200 h 112 l -42.66,-171.66 q -23.67,-95 -102.67,-155 -79,-60 -180,-60 -122,0 -207.67,84.66 -85.66,84.67 -85.66,206.36 0,62.95 25.71,119.6 25.72,56.65 72.95,100.71 l 21.34,20 v 202 z m 256,-366.67 z",
    ),
    "constraint": (
        "0 -960 960 960",
        "M 220,-520 80,-600 v -160 l 140,-80 140,80 v 160 z m 0,-76.67 73.33,-42 v -82.66 l -73.33,-42 -73.33,42 v 82.66 z m 433.33,88.34 V -586 L 800,-500 v 280 L 560,-80 320,-220 v -280 l 146.67,-85.33 v 77 l -80,47 V -258 L 560,-157.67 733.33,-258 v -203.33 z m -126.66,95 V -880 h 354 l -75.34,113.33 75.34,113.34 H 593.33 v 240 z M 560,-333 Z M 220,-680 Z",
    ),
}
ICON_PATHS["default"] = ICON_PATHS["graphnode"]


_CMD_RE = re.compile(r"([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)")
_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
# x/y role of each numeric argument, cycled - only axis-aligned scale +
# translate is ever applied (never rotation/skew), so a relative command's
# deltas scale correctly without first resolving them to absolute points.
_XY_ROLES: dict[str, str] = {
    "M": "xy", "L": "xy", "T": "xy",
    "H": "x", "V": "y",
    "C": "xyxyxy", "S": "xyxy", "Q": "xyxy",
    "Z": "",
}


def transform_svg_path(d: str, viewbox: str, x: float, y: float, w: float, h: float) -> str:
    """Remaps an SVG path's own coordinates from its `viewbox` ("minx miny
    vw vh") into the axis-aligned box (x, y, w, h) - used to drop one of the
    fixed ICON_PATHS into a specific node's icon slot at render time."""
    vx, vy, vw, vh = (float(v) for v in viewbox.split())
    sx, sy = (w / vw if vw else 1.0), (h / vh if vh else 1.0)
    out = []
    for cmd, args_str in _CMD_RE.findall(d):
        nums = [float(n) for n in _NUM_RE.findall(args_str)]
        roles = _XY_ROLES.get(cmd.upper(), "")
        absolute = cmd.isupper()
        scaled = []
        for i, n in enumerate(nums):
            role = roles[i % len(roles)] if roles else None
            if role == "x":
                scaled.append(x + (n - vx) * sx if absolute else n * sx)
            elif role == "y":
                scaled.append(y + (n - vy) * sy if absolute else n * sy)
            else:
                scaled.append(n)
        out.append(cmd + " " + " ".join(f"{v:.3f}" for v in scaled))
    return " ".join(out)
