from dataclasses import dataclass, field


@dataclass(frozen=True)
class Theme:
    """Consistent theme for Qewton plots."""

    # TODO: these values are just any temporary values

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
    background_color: str = "white"
    text_color: str = "#333333"

    # Typography
    font_family: str = "Courier New"
    font_size_title: int = 20
    font_size_axes: int = 14
    font_size_labels: int = 12

    show_grid: bool = True
    show_legend: bool = True

    # Plot-specific
    marker_size: int = 8
    line_width: float = 2.0
    opacity_default: float = 0.8

    # Geometry-specific
    surface_opacity: float = 0.9
    wireframe_opacity: float = 0.3

    @classmethod
    def default(cls):
        """Qewton default theme (tech-focused, light)."""
        return cls()

    @classmethod
    def dark(cls):
        """Dark theme for reduced eye strain."""
        return cls(
            background_color="#1a1a1a",
            text_color="#e0e0e0",
        )

    def get_color(self, index: int, secondary: bool = False) -> str:
        """Get color from palette by index."""
        palette = (
            self.secondary_color_palette if secondary else self.primary_color_palette
        )
        return palette[index % len(palette)]
