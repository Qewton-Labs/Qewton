"""Themes: colormap roles, categorical colors and the presets. How a
renderer applies a theme is checked by the dark variant of test_matrix.py."""

import pytest

from qewton.visualization import DARK_THEME, LIGHT_THEME, Theme, cmaps


def test_none_is_the_sequential_role():
    assert LIGHT_THEME.resolve_cmap(None) == LIGHT_THEME.resolve_cmap(cmaps.sequential)


@pytest.mark.parametrize("role", [cmaps.sequential, cmaps.diverging])
def test_roles_follow_the_theme(role):
    assert LIGHT_THEME.resolve_cmap(role) != DARK_THEME.resolve_cmap(role)


def test_a_renderer_colormap_passes_through():
    assert LIGHT_THEME.resolve_cmap("Viridis") == "Viridis"


def test_a_role_as_a_theme_field_is_rejected():
    theme = Theme(sequential_cmap=cmaps.diverging)
    with pytest.raises(ValueError, match="concrete colormaps"):
        theme.resolve_cmap(cmaps.sequential)


def test_categorical_colors_cycle_through_the_palette():
    theme = Theme(primary_color_palette=["#000000", "#ffffff"])
    assert [theme.get_color(i) for i in range(3)] == ["#000000", "#ffffff", "#000000"]


_TWO = Theme(primary_color_palette=["#000000", "#ffffff"])
_RAMP = cmaps.QewtonColormap.from_stops(
    "ramp",
    [[0.0, "#000000"], [0.25, "#404040"], [0.5, "#808080"], [0.75, "#c0c0c0"], [1.0, "#ffffff"]],
)

#: (cmap, steps) -> the color of every band, in order.
BANDS = {
    "the palette, one band per color": (cmaps.categorical, None, ["#000000", "#ffffff"]),
    "the palette, cycling": (cmaps.categorical, 3, ["#000000", "#ffffff", "#000000"]),
    "a colormap at the band centers": (_RAMP, 2, ["#404040", "#c0c0c0"]),
    "a colormap, odd steps": (_RAMP, 3, ["#404040", "#808080", "#c0c0c0"]),
}


@pytest.mark.parametrize("name", list(BANDS))
def test_bands(name):
    # Two stops per band, of one color, positions strictly increasing over
    # equal bands.
    cmap, steps, colors = BANDS[name]
    stops = _TWO.resolve_cmap(cmap, steps=steps)
    assert [c for _, c in stops[::2]] == colors
    assert [c for _, c in stops[1::2]] == colors
    positions = [p for p, _ in stops]
    assert all(b > a for a, b in zip(positions, positions[1:]))
    assert positions[1::2] == pytest.approx([(i + 1) / len(colors) for i in range(len(colors))])


def test_a_renderer_colormap_takes_no_steps():
    assert LIGHT_THEME.resolve_cmap("Viridis", steps=3) == "Viridis"


def test_the_presets_differ():
    assert LIGHT_THEME.background_color != DARK_THEME.background_color
    assert LIGHT_THEME.text_color != DARK_THEME.text_color
