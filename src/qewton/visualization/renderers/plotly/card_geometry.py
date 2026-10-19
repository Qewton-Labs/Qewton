"""Geometry helpers for drawing GraphPlot nodes as beveled "cards".

A card is a rounded rectangle with a 3D bevel (depth) edge and a
semicircular notch per port. The functions here turn positions, sizes and
port counts into polygons, SVG path ``d`` strings and point lists; they use
neither Plotly nor qewton's graph classes.
:class:`~qewton.visualization.renderers.plotly.graph.NodeLinkArtist` turns
the results into Plotly shapes and traces. The bevel outline is a polygon
union computed with ``shapely`` (imported lazily).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

SQ = 0.70710678  # sin/cos(45 deg) - the 45-degree point on a corner's arc


# ---------------------------------------------------------------------------
# Polygon construction
# ---------------------------------------------------------------------------


def arc(
    cx: float, cy: float, r: float, a0: float, a1: float, n: int = 16,
    registry: dict | None = None,
) -> list[tuple[float, float]]:
    """Sample ``n + 1`` points of a circular arc.

    Args:
        cx: x-coordinate of the circle center.
        cy: y-coordinate of the circle center.
        r: Radius.
        a0: Start angle in radians.
        a1: End angle in radians.
        n: Number of segments.
        registry: If given, mutated in place: every sampled point, rounded to 6
            decimals, is mapped to its circle's ``(cx, cy, r)``, so
            :func:`path_from_poly` can draw the run as a true arc.

    Returns:
        list of tuple of float: The sampled ``(x, y)`` points from `a0` to
            `a1`.
    """
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
    """A semicircular port notch in a vertical card edge.

    Attributes:
        cy (float): y-coordinate of the notch center.
        r (float): Notch radius.
    """

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
    """The card's flat silhouette.

    A rounded rectangle with one semicircular notch cut into the left edge
    per input port and into the right edge per output port.

    Args:
        x: x-coordinate of the top-left corner.
        y: y-coordinate of the top-left corner (y grows downward).
        w: Width.
        h: Height.
        r: Corner radius.
        left_notches: Notches of the left (input) edge.
        right_notches: Notches of the right (output) edge.
        arc_n: Number of segments per sampled arc.
        registry: Optional dict tagging every arc point with its circle, see
            :func:`arc`.

    Returns:
        list of tuple of float: The polygon's ``(x, y)`` vertices.
    """
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
    # Snaps to 1e-6 before the union, so points meant to coincide but
    # differing by float noise don't become slivers.
    return [(round(px, 6), round(py, 6)) for px, py in poly]


def _poly_union_all(polys: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    # buffer(0) repairs slightly invalid polygons (e.g. tightly packed
    # ports touching a corner arc), which would otherwise make the union
    # raise a TopologyException.
    shapes = [Polygon(_round_poly(p)).buffer(0) for p in polys if len(p) >= 3]
    merged = unary_union(shapes)
    if merged.geom_type == "Polygon":
        return list(merged.exterior.coords)[:-1]
    # Degenerate parameters may split the union; the silhouette is the
    # piece with the most vertices (bridge triangles are tiny).
    largest = max(merged.geoms, key=lambda g: len(g.exterior.coords))
    return list(largest.exterior.coords)[:-1]


def build_back_polygon(
    x: float, y: float, w: float, h: float, r: float,
    left_notches: list[Notch], right_notches: list[Notch], depth: float, arc_n: int = 16,
) -> tuple[list[tuple[float, float]], dict]:
    """The beveled "back" silhouette of a card.

    The union of the front polygon (:func:`build_flat_polygon`), a copy
    shifted by `depth` along x and y, and bridge triangles at the top-right
    and bottom-left corners and at each notch's outer tangent point.

    Args:
        x: As in :func:`build_flat_polygon`.
        y: As in :func:`build_flat_polygon`.
        w: As in :func:`build_flat_polygon`.
        h: As in :func:`build_flat_polygon`.
        r: As in :func:`build_flat_polygon`.
        left_notches: As in :func:`build_flat_polygon`.
        right_notches: As in :func:`build_flat_polygon`.
        arc_n: As in :func:`build_flat_polygon`.
        depth: Bevel offset along both x and y.

    Returns:
        tuple: ``(polygon, registry)``, where ``polygon`` (list of tuple of
            float) holds the outline's ``(x, y)`` vertices and ``registry``
            (dict) is the arc registry (see :func:`arc`) covering the front
            polygon and its shifted copy, for :func:`path_from_poly`.
    """
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
    """Diagonal bevel edge lines at the top tangent of each output notch.

    These are the only bevel edges not already part of the back
    silhouette's outline (:func:`build_back_polygon`).

    Args:
        x: As in :func:`build_flat_polygon`.
        y: As in :func:`build_flat_polygon`.
        w: As in :func:`build_flat_polygon`.
        h: As in :func:`build_flat_polygon`.
        r: As in :func:`build_flat_polygon`.
        left_notches: As in :func:`build_flat_polygon`.
        right_notches: As in :func:`build_flat_polygon`.
        depth: Bevel offset along both x and y.

    Returns:
        list of tuple of float: One ``(x1, y1, x2, y2)`` line per output notch.
    """
    lines = []
    for nt in right_notches:
        ax, ay = x + w, nt.cy - nt.r
        lines.append((ax, ay, ax + depth, ay + depth))
    return lines


def darken_hex(hex_color: str, factor: float) -> str:
    """Scale the RGB channels of a ``#rrggbb`` color.

    Args:
        hex_color: Color as ``#rrggbb`` (the ``#`` is optional).
        factor: Multiplier per channel; results are clipped to 0..255.

    Returns:
        str: The scaled color as ``#rrggbb``, or `hex_color` unchanged if it is
            not a 6-digit hex color.
    """
    m = re.match(r"^#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$", hex_color)
    if not m:
        return hex_color
    channels = [max(0, min(255, round(int(m.group(i), 16) * factor))) for i in (1, 2, 3)]
    return "#" + "".join(f"{c:02x}" for c in channels)


def port_layout(n: int, card_h: float, corner_r: float, outer_r: float, min_gap: float, min_margin: float):
    """Vertical port positions on a card edge.

    Ports are spaced by at least `min_gap`, with at least `min_margin` to
    the corner arcs; extra height is distributed evenly over all gaps and
    both margins.

    Args:
        n: Number of ports.
        card_h: Card height.
        corner_r: Corner radius.
        outer_r: Notch radius.
        min_gap: Minimum gap between neighboring notches.
        min_margin: Minimum margin between the outermost notches and the
            corners.

    Returns:
        tuple: ``(centers, min_height)``, where ``centers`` (list of float)
            holds the port center y-offsets from the card's top edge and
            ``min_height`` (float) is the minimum card height for `n` ports.
    """
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


#: Largest sub-arc (radians) _bezier_arc_segments approximates with one
#: cubic bezier.
_MAX_BEZIER_ARC = math.pi / 2 + 1e-6


def _bezier_arc_segments(cx: float, cy: float, r: float, a_start: float, total: float):
    """Splits a circular arc (from angle a_start, sweeping signed `total`
    radians) into cubic beziers of at most 90 degrees each, as
    ((c1x, c1y), (c2x, c2y), (ex, ey)) per piece. Plotly shape paths do
    not support the SVG `A` command."""
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
    """SVG path ``d`` string for a Plotly ``type="path"`` shape.

    Each vertex becomes a straight ``L`` segment, except runs of
    consecutive vertices that `registry` maps to the same circle: such a
    run is drawn as true cubic-bezier arc segments between its first and
    last vertex.

    Args:
        poly: Polygon vertices.
        registry: Arc registry from :func:`arc`, :func:`build_flat_polygon` or
            :func:`build_back_polygon`; None draws straight segments only.
        close: Whether to close the path with ``Z``.

    Returns:
        str: The path string; empty for an empty `poly`.

    Notes:
        The registry lookup also works on the output of
        :func:`build_back_polygon`: its pieces only touch, so the union keeps
        the input vertices. A vertex moved by more than the 1e-6 rounding
        tolerance falls back to an ``L`` segment.
    """
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
            # Snap to the run's exact end point, so the next segment starts
            # there without float noise.
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


#: Minimum bezier-handle length of a wire, in GraphLayout data units.
MIN_WIRE_HANDLE = 0.3


def wire_control_points(p1: tuple[float, float], p2: tuple[float, float]):
    """Control points of the cubic bezier wire from `p1` to `p2`.

    Both handles are horizontal, with length 0.4 times the x-distance but
    at least :data:`MIN_WIRE_HANDLE`.

    Args:
        p1: Start point.
        p2: End point.

    Returns:
        tuple of tuple of float: The two control points ``(c1, c2)``.
    """
    dx_span = max(MIN_WIRE_HANDLE, abs(p2[0] - p1[0]) * 0.4)
    return (p1[0] + dx_span, p1[1]), (p2[0] - dx_span, p2[1])


def wire_path(p1: tuple[float, float], p2: tuple[float, float]) -> str:
    """SVG path ``d`` string of the cubic bezier wire from `p1` to `p2`.

    Args:
        p1: Start point.
        p2: End point.

    Returns:
        str: The path string, with control points from
            :func:`wire_control_points`.
    """
    (c1x, c1y), (c2x, c2y) = wire_control_points(p1, p2)
    return f"M {p1[0]:.2f} {p1[1]:.2f} C {c1x:.2f} {c1y:.2f}, {c2x:.2f} {c2y:.2f}, {p2[0]:.2f} {p2[1]:.2f}"


def sample_bezier(p1: tuple[float, float], p2: tuple[float, float], n: int = 24) -> list[tuple[float, float]]:
    """Sample the wire :func:`wire_path` draws as points.

    Args:
        p1: Start point.
        p2: End point.
        n: Number of segments.

    Returns:
        list of tuple of float: ``n + 1`` points along the curve.
    """
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
    """Point on the wire from `p1` to `p2` to place its label at.

    Tries the curve parameters 0.5, 0.35, 0.65, 0.22 and 0.78 in this order
    and returns the first point outside all `node_boxes`, or the midpoint if
    none is.

    Args:
        p1: Start point of the wire.
        p2: End point of the wire.
        node_boxes: ``(x0, y0, x1, y1)`` boxes to avoid.

    Returns:
        tuple of float: The label position.
    """
    for t in (0.5, 0.35, 0.65, 0.22, 0.78):
        pt = _bezier_point(p1, p2, t)
        if not any(_point_in_box(pt, box) for box in node_boxes):
            return pt
    return _bezier_point(p1, p2, 0.5)


#: Height per character of the background box behind an edge label (the
#: label text is rotated 90 degrees, so the box grows vertically).
EDGE_LABEL_CHAR_SIZE = 0.05
#: Vertical padding of the edge-label box.
EDGE_LABEL_PAD = 0.045
#: Fixed width of the edge-label box.
EDGE_LABEL_THICKNESS = 0.11


def edge_label_box(label: str, cx: float, cy: float) -> tuple[float, float, float, float]:
    """Background box for a vertically drawn edge label.

    Args:
        label: The label text; its length sets the box height.
        cx: x-coordinate of the box center.
        cy: y-coordinate of the box center.

    Returns:
        tuple of float: The box as ``(x0, y0, x1, y1)``.
    """
    half_h = max(0.06, len(label) * EDGE_LABEL_CHAR_SIZE * 0.5) + EDGE_LABEL_PAD
    half_w = EDGE_LABEL_THICKNESS / 2
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


# ---------------------------------------------------------------------------
# Type icons (currently not drawn by NodeLinkArtist)
# ---------------------------------------------------------------------------

#: Per node category, an SVG icon as ``(viewbox, path d string)``; placed
#: into a card with transform_svg_path().
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
# x/y role of each numeric argument, cycled. Only axis-aligned scale and
# translate are applied, so relative deltas can be scaled directly.
_XY_ROLES: dict[str, str] = {
    "M": "xy", "L": "xy", "T": "xy",
    "H": "x", "V": "y",
    "C": "xyxyxy", "S": "xyxy", "Q": "xyxy",
    "Z": "",
}


def transform_svg_path(d: str, viewbox: str, x: float, y: float, w: float, h: float) -> str:
    """Map an SVG path from its viewbox into an axis-aligned box.

    Supports the commands M, L, T, H, V, C, S, Q and Z (absolute and
    relative).

    Args:
        d: SVG path ``d`` string.
        viewbox: The path's viewbox as ``"minx miny width height"``.
        x: x-coordinate of the target box's origin.
        y: y-coordinate of the target box's origin.
        w: Width of the target box.
        h: Height of the target box.

    Returns:
        str: The transformed path string.
    """
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
