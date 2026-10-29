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


def test_the_presets_differ():
    assert LIGHT_THEME.background_color != DARK_THEME.background_color
    assert LIGHT_THEME.text_color != DARK_THEME.text_color
