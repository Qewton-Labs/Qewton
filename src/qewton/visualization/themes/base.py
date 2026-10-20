"""The Theme dataclass holding the visual settings of a Figure."""

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
    """Colors, fonts and sizes applied to every plot of a Figure.

    A Theme is immutable. Custom themes are created as ``Theme(...)`` with
    the fields to override; :data:`~qewton.visualization.themes.LIGHT_THEME`
    and :data:`~qewton.visualization.themes.DARK_THEME` are the standard
    ones. All parameters are stored as fields of the same name.

    Args:
        background_color: Figure background color.
        text_color: Figure text color.
        line_color: Color of non-data lines: borders, edges, wireframes.
        grid_color: Color of axis gridlines.
        font_family: Font stack for all figure text (Latin Modern by
            default). For TeX-rendered math, wrap a title or label in
            ``$...$``.
        font_size_title: Font size of titles in points.
        font_size_axes: Font size of axis titles in points.
        font_size_labels: Font size of labels in points.
        show_grid: Whether axis gridlines are drawn.
        show_legend: Whether legends are shown.
        show_axis_line: Whether axis lines with outward ticks are drawn.
        axis_box: Whether the axis lines are mirrored to all four sides of
            the plot area.
        marker_size: Default marker size of data marks.
        line_width: Default line width of data marks.
        opacity_default: Default opacity of data marks.
        surface_opacity: Opacity of geometry surfaces.
        wireframe_opacity: Opacity of geometry wireframes.
        geometry_color: Color of a geometry surface with no data to color
            by.
        vector_color: Color of uncolored arrow fields (MeshVectorPlot,
            QuiverPlot).
        primary_color_palette: Categorical colors, assigned to plots by
            ``color_index``.
        sequential_cmap: The concrete colormap the role
            :data:`~qewton.visualization.themes.cmaps.sequential` resolves
            to: a :class:`~qewton.visualization.themes.cmaps.QewtonColormap`
            or a renderer-native colormap (a Plotly colorscale name or
            list), never a role. Also used for a ColorSpec without a cmap.
        diverging_cmap: The concrete colormap the role
            :data:`~qewton.visualization.themes.cmaps.diverging` resolves
            to (same kinds as `sequential_cmap`).
        cyclic_cmap: The concrete colormap the role
            :data:`~qewton.visualization.themes.cmaps.cyclic` resolves to
            (same kinds as `sequential_cmap`).
        node_color_by_type: ``{"fill": ..., "border": ...}`` color pair per
            node category of a computation-graph plot: ``"datanode"``
            (data/input), ``"graphnode"`` (model/algorithm, also used for
            uncategorized nodes) and ``"constraint"`` (loss/residual).
        node_color_default: Fill color used if no category pair applies.
        card_corner_radius: Corner radius of node cards, in graph-layout
            data units (as are the following card sizes).
        card_port_inner_radius: Radius of the port dot.
        card_port_outer_radius: Radius of the port notch cut into the card
            edge.
        card_port_ring_radius: Radius of the ring around wired outputs and
            constraint ports.
        card_depth: Offset of the 3D bevel.
        card_line_width: Card outline width in points.
        card_title_font_size: Font size of card titles in points.
        card_title_font_family: Font stack of card titles.
        card_edge_formula_size: Font size of port labels, in points.
        graph_px_per_unit: Pixels per graph-layout data unit used to size a
            graph plot's figure when no explicit width/height is set.

    See Also:
        :mod:`qewton.visualization.themes.cmaps`
    """

    # Figure chrome: theme-dependent tokens, unlike the color families.
    background_color: str = QEWTON_SURFACE_CANVAS["light"]
    text_color: str = QEWTON_INK["light"]
    line_color: str = QEWTON_INK["light"]
    grid_color: str = QEWTON_BORDER["light"]

    # Typography - a Latin Modern stack, falling back to a generic serif.
    font_family: str = "Latin Modern Roman, CMU Serif, Georgia, serif"
    font_size_title: int = 20
    font_size_axes: int = 14
    font_size_labels: int = 12

    show_grid: bool = False
    show_legend: bool = True

    # Axis frame - a pgfplots-style box with outward ticks.
    show_axis_line: bool = True
    axis_box: bool = True

    # Plot-specific
    marker_size: int = 6
    line_width: float = 1.25
    opacity_default: float = 0.8

    # Geometry-specific
    surface_opacity: float = 0.9
    wireframe_opacity: float = 0.3
    geometry_color: str = QEWTON_FAMILIES["brown"]["A"]
    vector_color: str = QEWTON_INK["light"]

    # Color scheme - the categorical palette is theme-independent.
    primary_color_palette: list[str] = field(
        default_factory=lambda: list(QEWTON_CATEGORICAL_COLORS)
    )
    # What each colormap role resolves to - see resolve_cmap().
    sequential_cmap: "cmaps.QewtonColormap | str | list" = cmaps.meadow
    diverging_cmap: "cmaps.QewtonColormap | str | list" = cmaps.redblue
    cyclic_cmap: "cmaps.QewtonColormap | str | list" = cmaps.phase

    # Computation-graph plots (GraphPlot / NodeLinkArtist): one fixed
    # (fill, border) pair per node role, the same in every theme.
    node_color_by_type: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "constraint": dict(QEWTON_NODE_COLORS["loss"]),
            "datanode": dict(QEWTON_NODE_COLORS["data"]),
            "graphnode": dict(QEWTON_NODE_COLORS["model"]),
        }
    )
    # Fallback fill, paired with line_color as border (see _node_colors).
    node_color_default: str = QEWTON_FAMILIES["brown"]["B"]

    # Card geometry (NodeLinkArtist's card drawing, see
    # renderers/plotly/card_geometry.py), in GraphLayout's data units (see
    # plots/graph/layout.py). Port notch positions come from PortLayout.y;
    # only their radii are set here.
    card_corner_radius: float = 0.12
    card_port_inner_radius: float = 0.056  # the solid port dot
    card_port_outer_radius: float = 0.096  # the notch cut into the card edge
    card_port_ring_radius: float = 0.072  # halo (wired output) / loss-ring (constraint) radius
    card_depth: float = 0.04  # 3D bevel offset
    card_line_width: float = 1.0  # Plotly line widths are in points, not data units
    card_title_font_size: int = 11  # Plotly annotation font sizes are always points
    # Fonts available in browsers and in kaleido (save_png()/save_svg()).
    card_title_font_family: str = "Arial, Helvetica, sans-serif"
    card_edge_formula_size: int = 8  # port-label annotation font size, in points
    # Card sizes are in data units but fonts in points, so the pixel size
    # per data unit is fixed (used by NodeLinkArtist.create()).
    graph_px_per_unit: float = 110.0

    @classmethod
    def default(cls):
        """Create the default light, paper-style theme (``LIGHT_THEME``).

        Returns:
            Theme: A theme with all default field values.
        """
        return cls()

    @classmethod
    def dark(cls):
        """Create the dark theme (``DARK_THEME``).

        Returns:
            Theme: A theme with dark chrome and dark-background colormaps.
        """
        return cls(
            background_color=QEWTON_SURFACE_CANVAS["dark"],
            text_color=QEWTON_INK["dark"],
            line_color=QEWTON_INK["dark"],
            grid_color=QEWTON_BORDER["dark"],
            geometry_color=QEWTON_FAMILIES["brown"]["C"],
            vector_color=QEWTON_INK["dark"],
            sequential_cmap=cmaps.dusk,
            diverging_cmap=cmaps.orangeviolet_paper,
        )

    def resolve_cmap(self, cmap=None):
        """Resolve a ColorSpec's `cmap` to a renderer-ready colormap.

        Args:
            cmap (Colormap | str | list | None): None or a
                :class:`~qewton.visualization.themes.cmaps.ColormapRole`
                resolves to this theme's field for that role (None means
                sequential). A
                :class:`~qewton.visualization.themes.cmaps.QewtonColormap`
                becomes its colorscale list. Anything else, e.g. a
                renderer's colormap name or colorscale list, is returned
                unchanged.

        Returns:
            str | list: The renderer-native colormap.

        Raises:
            ValueError: If this theme's field for the requested role is
                itself a role.
        """
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

    def get_color(self, index: int) -> str:
        """Get a categorical color by index, cycling through the palette.

        Args:
            index: Position in the palette, taken modulo its length.

        Returns:
            The color.
        """
        palette = self.primary_color_palette
        return palette[index % len(palette)]
