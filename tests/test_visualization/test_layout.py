import pytest

from qewton.visualization.layout import Column, Layout, Overlay, Row, normalize
from qewton.visualization.plots.base import Plot


class _Plot2D(Plot):
    """Minimal concrete Plot for layout tests - embedding_dim=2 (the
    Plot base default)."""


class _Plot3D(Plot):
    """Minimal concrete Plot for layout tests - embedding_dim=3."""

    @property
    def embedding_dim(self):
        return 3


class TestOverlay:
    def test_plots_returns_every_given_plot_in_order(self):
        a, b = _Plot2D(), _Plot2D()
        overlay = Overlay(a, b)
        assert overlay.plots == [a, b]

    def test_embedding_dim_is_the_shared_dim(self):
        a, b = _Plot2D(), _Plot2D()
        assert Overlay(a, b).embedding_dim == 2

    def test_empty_overlay_has_no_embedding_dim(self):
        assert Overlay().embedding_dim is None

    def test_mismatched_embedding_dim_raises(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            Overlay(_Plot2D(), _Plot3D())

    def test_is_a_layout(self):
        assert isinstance(Overlay(_Plot2D()), Layout)


class TestRow:
    def test_plots_flattens_bare_plots(self):
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        row = Row(a, b, c)
        assert row.plots == [a, b, c]

    def test_plots_flattens_nested_layouts(self):
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        row = Row(Overlay(a, b), c)
        assert row.plots == [a, b, c]

    def test_is_a_layout_not_a_column(self):
        row = Row(_Plot2D())
        assert isinstance(row, Layout)
        assert not isinstance(row, Column)

    def test_items_are_stored_in_order(self):
        a, b = _Plot2D(), _Plot2D()
        row = Row(a, b)
        assert row.items == [a, b]


class TestColumn:
    def test_plots_flattens_bare_plots(self):
        a, b = _Plot2D(), _Plot2D()
        column = Column(a, b)
        assert column.plots == [a, b]

    def test_plots_flattens_nested_layouts(self):
        a, b, c, d = _Plot2D(), _Plot2D(), _Plot2D(), _Plot2D()
        column = Column(Row(a, b), Row(c, d))
        assert column.plots == [a, b, c, d]

    def test_is_a_layout_not_a_row(self):
        column = Column(_Plot2D())
        assert isinstance(column, Layout)
        assert not isinstance(column, Row)


class TestNestedComposition:
    def test_deeply_nested_plots_are_reachable(self):
        a, b, c, d, e = (_Plot2D() for _ in range(5))
        layout = Row(Overlay(a, b), Column(c, Row(d, e)))
        assert layout.plots == [a, b, c, d, e]

    def test_reusing_a_returned_layouts_plots_in_a_new_arrangement(self):
        """Figure(Column(*result.plots)) - a flat list of plots must be a
        valid Column/Row input on its own."""
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        original = Row(a, b, c)
        rearranged = Column(*original.plots)
        assert rearranged.plots == [a, b, c]


def _shape(grid):
    return tuple(len(row) for row in grid)


class TestNormalizeBaseCases:
    def test_bare_plot_is_a_single_cell(self):
        a = _Plot2D()
        grid = normalize(a)
        assert _shape(grid) == (1,)
        assert grid[0][0].plots == [a]

    def test_overlay_is_a_single_cell(self):
        overlay = Overlay(_Plot2D(), _Plot2D())
        grid = normalize(overlay)
        assert grid == [[overlay]]


class TestNormalizeRow:
    def test_row_of_bare_plots_is_one_row_of_cells(self):
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        grid = normalize(Row(a, b, c))
        assert _shape(grid) == (3,)
        assert [ov.plots for ov in grid[0]] == [[a], [b], [c]]

    def test_row_of_overlay_and_plot(self):
        """Figure(Row(Overlay(domain, pred), reference)) from the plan doc."""
        domain, pred, reference = _Plot2D(), _Plot2D(), _Plot2D()
        grid = normalize(Row(Overlay(domain, pred), reference))
        assert _shape(grid) == (2,)
        assert grid[0][0].plots == [domain, pred]
        assert grid[0][1].plots == [reference]


class TestNormalizeColumn:
    def test_2x2_from_two_rows(self):
        """Figure(Column(Row(a, b), Row(c, d))) from the plan doc."""
        a, b, c, d = _Plot2D(), _Plot2D(), _Plot2D(), _Plot2D()
        grid = normalize(Column(Row(a, b), Row(c, d)))
        assert _shape(grid) == (2, 2)
        assert [ov.plots for ov in grid[0]] == [[a], [b]]
        assert [ov.plots for ov in grid[1]] == [[c], [d]]

    def test_narrower_row_is_padded_on_the_right_not_spanned(self):
        """Figure(Column(Row(a, b), c)) from the plan doc: c occupies the
        first cell of its row, the second stays empty."""
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        grid = normalize(Column(Row(a, b), c))
        assert _shape(grid) == (2, 2)
        assert grid[1][0].plots == [c]
        assert grid[1][1].plots == []  # padded, empty Overlay


class TestNormalizeNestedPadding:
    def test_shorter_branch_of_a_row_is_padded_at_the_bottom(self):
        """The Row-nesting-Columns-of-different-heights generalization of
        the Column(Row, Plot) padding rule."""
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        grid = normalize(Row(Column(a, b), c))
        assert _shape(grid) == (2, 2)
        assert grid[0][0].plots == [a]
        assert grid[0][1].plots == [c]
        assert grid[1][0].plots == [b]
        assert grid[1][1].plots == []  # padded

    def test_deeply_nested_grid_stays_rectangular(self):
        a, b, c, d, e = (_Plot2D() for _ in range(5))
        grid = normalize(Row(Overlay(a, b), Column(c, Row(d, e))))
        row_lengths = {len(row) for row in grid}
        assert len(row_lengths) == 1  # rectangular


class TestOverlayMutation:
    def test_remove_drops_a_present_plot_and_returns_true(self):
        a, b = _Plot2D(), _Plot2D()
        overlay = Overlay(a, b)
        assert overlay.remove(a) is True
        assert overlay.plots == [b]

    def test_remove_returns_false_for_an_absent_plot(self):
        overlay = Overlay(_Plot2D())
        assert overlay.remove(_Plot2D()) is False

    def test_replace_swaps_in_place_and_returns_true(self):
        a, b, c = _Plot2D(), _Plot2D(), _Plot2D()
        overlay = Overlay(a, b)
        assert overlay.replace(a, c) is True
        assert overlay.plots == [c, b]

    def test_replace_returns_false_for_an_absent_plot(self):
        overlay = Overlay(_Plot2D())
        assert overlay.replace(_Plot2D(), _Plot2D()) is False

    def test_replace_rejects_a_mismatched_embedding_dim(self):
        a, b = _Plot2D(), _Plot2D()
        overlay = Overlay(a, b)
        with pytest.raises(ValueError, match="embedding_dim"):
            overlay.replace(a, _Plot3D())
        assert overlay.plots == [a, b]  # unchanged on rejection
