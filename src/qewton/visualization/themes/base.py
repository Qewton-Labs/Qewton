from dataclasses import dataclass, field

from qewton.visualization.themes import cmaps
from qewton.visualization.themes.palette import (
    QEWTON_BORDER,
    QEWTON_CATEGORICAL_COLORS,
    QEWTON_FAMILIES,
    QEWTON_INK,
    QEWTON_NODE_COLORS,
    QEWTON_SURFACE_CANVAS,
)


@dataclass(frozen=True)
class Theme:
    """Consistent theme for Qewton plots - every field here is actually
    applied somewhere in renderers/plotly (figure-level chrome via
    PlotlyRenderer.setup(), per-artist fills/lines/text elsewhere), not just
    declared. A corporate theme is just another `Theme(...)` instance with
    whichever fields it wants to override - see LIGHT_THEME/DARK_THEME below
    for the two standard ones.
    """

    # Figure chrome - from the design system's theme-dependent chrome
    # tokens (project/tokens.json: surface-canvas/ink/border), not the
    # 7 SVG color families below, which are theme-INDEPENDENT instead.
    background_color: str = QEWTON_SURFACE_CANVAS["light"]
    text_color: str = QEWTON_INK["light"]
    line_color: str = QEWTON_INK["light"]  # borders, edges, wireframes - non-data lines
    grid_color: str = QEWTON_BORDER["light"]  # axis gridlines, when show_grid is on

    # Typography - a Computer Modern / Latin Modern stack, falling back to a
    # generic serif where those aren't installed. This is a font
    # substitution, not real TeX typesetting - for genuine TeX-rendered
    # math, wrap a title/label in $...$: Plotly renders that through
    # MathJax (in real Computer Modern) independently of this theme.
    font_family: str = "Latin Modern Roman, CMU Serif, Georgia, serif"
    font_size_title: int = 20
    font_size_axes: int = 14
    font_size_labels: int = 12

    show_grid: bool = False  # pgfplots/TikZ draw no major gridlines by default
    show_legend: bool = True

    # Axis frame - TikZ/pgfplots draws a full box around the plot area with
    # outward tick marks by default, rather than Plotly's bare left/bottom
    # lines with no ticks.
    show_axis_line: bool = True
    axis_box: bool = True  # mirror the line to all four sides, not just left/bottom

    # Plot-specific
    marker_size: int = 6
    line_width: float = 1.25
    opacity_default: float = 0.8

    # Geometry-specific
    surface_opacity: float = 0.9
    wireframe_opacity: float = 0.3
    geometry_color: str = QEWTON_FAMILIES["brown"]["A"]  # a mesh/geometry surface with no data to color by
    vector_color: str = QEWTON_INK["light"]  # arrow fields (MeshVectorPlot, QuiverPlot) with color=False

    # Color scheme - primary_color_palette is the CVD-checked categorical
    # order over the 7 SVG families' C-shade (qewton.visualization.themes.
    # palette.QEWTON_CATEGORICAL_COLORS); same order in both themes, since
    # the family shades themselves are theme-independent.
    primary_color_palette: list[str] = field(
        default_factory=lambda: list(QEWTON_CATEGORICAL_COLORS)
    )
    # secondary_color_palette currently has no consumer anywhere in
    # renderers/plotly (get_color(secondary=True) is never called from
    # plotting code) - left as the old placeholder values rather than
    # silently repointing dead code at the new palette.
    secondary_color_palette: list[str] = field(
        default_factory=lambda: ["#bcbd22", "#17becf", "#aec7e8"]
    )
    # What each colormap role (cmaps.sequential/diverging/cyclic) resolves
    # to under this theme - see resolve_cmap(). Each is a concrete qewton
    # colormap (cmaps.meadow, ...) or a renderer-native one (a Plotly
    # colorscale name or list), never another role. sequential_cmap is also
    # the default for a ColorSpec with no cmap at all.
    sequential_cmap: "cmaps.QewtonColormap | str | list" = cmaps.meadow
    diverging_cmap: "cmaps.QewtonColormap | str | list" = cmaps.redblue
    cyclic_cmap: "cmaps.QewtonColormap | str | list" = cmaps.phase

    # Computation-graph plots (GraphPlot / NodeLinkArtist) - Nick's own
    # literal SVG-picked palette (graph_farben.svg, via themes/palette.py):
    # each role is a fixed (fill, border) hex pair - the SAME pair in BOTH
    # LIGHT_THEME and DARK_THEME (no per-theme role-swap any more; see
    # NodeLinkArtist._node_colors()). "datanode" = data/input cards
    # (yellow), "graphnode"/the default bucket = model/algorithm cards
    # (blue), "constraint" = loss/residual cards (green) - matching
    # Nick's own card-editor prototype's roles, just the new swatches.
    node_color_by_type: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "constraint": dict(QEWTON_NODE_COLORS["loss"]),
            "datanode": dict(QEWTON_NODE_COLORS["data"]),
            "graphnode": dict(QEWTON_NODE_COLORS["model"]),
        }
    )
    # Flat fallback FILL for any category with no family of its own (paired
    # with the theme's own line_color as its border - see _node_colors) -
    # brown is the one SVG family not claimed by a node role or kept as a
    # "variant" accent, so it reads as a neutral default rather than
    # competing with the three marked roles above.
    node_color_default: str = QEWTON_FAMILIES["brown"]["B"]
    cluster_outline_color: str = QEWTON_FAMILIES["violet"]["D"]

    # Card geometry (NodeLinkArtist's "card" drawing - the beveled rounded-
    # rect body, port notches and icon from card-editor.html, all ported via
    # renderers/plotly/card_geometry.py). These radii/depth are in
    # GraphLayout's own data units (BASE_HEIGHT=0.55, PORT_ROW_HEIGHT=0.35,
    # NODE_WIDTH=1.0 - see plots/graph/layout.py), NOT the literal pixel
    # values card-editor.html's own P object used (corner=15, outer=12, ...)
    # - those assumed a human manually sizing a handful of cards on a fixed
    # canvas, which doesn't hold for an auto-laid-out graph of arbitrary
    # size. Port notch positions themselves are NOT recomputed here -
    # they're read directly off PortLayout.y, GraphLayout's own already-
    # placed port anchors, so the notch always lines up with the port it
    # belongs to; only its radius is a theme constant.
    #
    # The four below keep card-editor.html's own ratios to its corner radius
    # (inner=7/corner=15, outer=12/15, ring=9/15, depth=5/15) rather than the
    # smaller ad-hoc ones an earlier pass used - ports and the 3D bevel read
    # as too dainty next to the card body at those smaller sizes.
    card_corner_radius: float = 0.12
    card_port_inner_radius: float = 0.056  # the solid port dot
    card_port_outer_radius: float = 0.096  # the notch cut into the card edge
    card_port_ring_radius: float = 0.072  # halo (wired output) / loss-ring (constraint) radius
    card_depth: float = 0.04  # 3D bevel offset
    card_line_width: float = 1.0  # Plotly line widths are in points, not data units
    card_icon_size: float = 0.13  # constant - never scaled by card size - currently unused, see card_geometry.ICON_PATHS
    card_title_font_size: int = 11  # Plotly annotation font sizes are always points
    # A stack of fonts that are universally available wherever the figure
    # gets rendered (a browser via save_html(), or kaleido for save_png()/
    # save_svg()) - "Sharp Sans" (the card-editor.html prototype's own
    # title font) isn't a web-safe/system font, so it silently fell back
    # anyway almost everywhere; Arial/Helvetica are as close to it as a
    # universally-installed sans-serif gets.
    card_title_font_family: str = "Arial, Helvetica, sans-serif"
    card_edge_formula_size: int = 8  # port-label annotation font size, in points
    # Pixels-per-data-unit NodeLinkArtist.create() sizes a GraphPlot's own
    # figure at (and computes its x/y axis range from), when the caller
    # hasn't already set an explicit width/height - see that method. Every
    # OTHER card-geometry constant above is a fixed font-size-in-points or a
    # fixed data-unit size, so a card only looks "right" (text neither
    # cramped nor oversized relative to its own box) at a SPECIFIC
    # pixels-per-unit ratio; leaving the actual rendered size up to whatever
    # container happens to display the figure (a notebook cell, a browser
    # window, a `fig.show()` with no size set) is exactly why the title
    # looked "too big" in some views and fine in others - the same
    # DATA-unit-sized card was landing at a different PIXEL size each time,
    # while its point-sized font didn't follow along. Pinning this ratio
    # (rather than, say, scaling font size per node) also keeps every node's
    # text rendered at one consistent size across a whole graph.
    graph_px_per_unit: float = 110.0

    @classmethod
    def default(cls):
        """Qewton default theme (TikZ/paper-style, light) - alias for LIGHT_THEME."""
        return cls()

    @classmethod
    def dark(cls):
        """Dark theme for reduced eye strain - alias for DARK_THEME."""
        return cls(
            background_color=QEWTON_SURFACE_CANVAS["dark"],
            text_color=QEWTON_INK["dark"],
            line_color=QEWTON_INK["dark"],
            grid_color=QEWTON_BORDER["dark"],
            geometry_color=QEWTON_FAMILIES["brown"]["C"],
            vector_color=QEWTON_INK["dark"],
            cluster_outline_color=QEWTON_FAMILIES["violet"]["B"],
            sequential_cmap=cmaps.dusk,
            diverging_cmap=cmaps.orangeviolet_paper,
        )

    def resolve_cmap(self, cmap=None):
        """The renderer-ready colormap for a ColorSpec's `cmap`: None and
        roles (cmaps.sequential/diverging/cyclic) resolve to this theme's
        own field for that role, a concrete qewton colormap to its
        colorscale list, and anything else - a renderer's own colormap name
        (a string) or colorscale list - passes through unchanged."""
        if cmap is None:
            cmap = cmaps.sequential
        if isinstance(cmap, cmaps.ColormapRole):
            resolved = getattr(self, f"{cmap.name}_cmap")
            if isinstance(resolved, cmaps.ColormapRole):
                raise ValueError(
                    f"Theme.{cmap.name}_cmap is the role {resolved!r} - a "
                    "theme's cmap fields must be concrete colormaps."
                )
            cmap = resolved
        if isinstance(cmap, cmaps.QewtonColormap):
            return [list(stop) for stop in cmap.colorscale]
        return cmap

    def get_color(self, index: int, secondary: bool = False) -> str:
        """Get color from palette by index."""
        palette = (
            self.secondary_color_palette if secondary else self.primary_color_palette
        )
        return palette[index % len(palette)]
