from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    """Consistent theme for Qewton plots - every field here is actually
    applied somewhere in renderers/plotly (figure-level chrome via
    PlotlyRenderer.setup(), per-artist fills/lines/text elsewhere), not just
    declared. A corporate theme is just another `Theme(...)` instance with
    whichever fields it wants to override - see LIGHT_THEME/DARK_THEME below
    for the two standard ones.
    """

    # Figure chrome
    background_color: str = "white"
    text_color: str = "black"
    line_color: str = "black"  # borders, edges, wireframes - non-data lines
    grid_color: str = "#cccccc"  # axis gridlines, when show_grid is on

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
    geometry_color: str = "lightgray"  # a mesh/geometry surface with no data to color by
    vector_color: str = "black"  # arrow fields (MeshVectorPlot, QuiverPlot) with no color_by_magnitude

    # Color scheme
    primary_color_palette: list[str] = field(
        default_factory=lambda: [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#487d48",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
        ]
    )
    secondary_color_palette: list[str] = field(
        default_factory=lambda: ["#bcbd22", "#17becf", "#aec7e8"]
    )
    default_cmap: str = "viridis"  # any Plotly-recognized colorscale name

    # Computation-graph plots (GraphPlot / NodeLinkArtist)
    node_color_by_type: dict[str, str] = field(
        default_factory=lambda: {
            "constraint": "aquamarine",
            "datanode": "lemonchiffon",
            "graphnode": "lightskyblue",
        }
    )
    node_color_default: str = "lightblue"
    cluster_outline_color: str = "darkslateblue"

    @classmethod
    def default(cls):
        """Qewton default theme (TikZ/paper-style, light) - alias for LIGHT_THEME."""
        return cls()

    @classmethod
    def dark(cls):
        """Dark theme for reduced eye strain - alias for DARK_THEME."""
        return cls(
            background_color="#1a1a1a",
            text_color="#e0e0e0",
            line_color="#cccccc",
            grid_color="#333333",
            geometry_color="#4a4a4a",
            vector_color="#e0e0e0",
            cluster_outline_color="#b8b8f0",
            default_cmap="plasma",
        )

    def get_color(self, index: int, secondary: bool = False) -> str:
        """Get color from palette by index."""
        palette = (
            self.secondary_color_palette if secondary else self.primary_color_palette
        )
        return palette[index % len(palette)]
