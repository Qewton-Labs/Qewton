from __future__ import annotations

from qewton.visualization.plots.base import Plot


class Layout:
    """Base class for anything that places Plots into a Figure. Figure
    normalizes whatever it is given into one of these before drawing.
    """

    @property
    def plots(self) -> list[Plot]:
        """Every Plot this layout contains, flattened, in reading order.

        Returns:
            list[Plot]: The contained plots.
        """
        raise NotImplementedError


class Overlay(Layout):
    """Plots drawn into the same cell, on top of each other.

    Args:
        *plots (Plot): The plots to draw together. All must report the
            same embedding_dim.

    Raises:
        ValueError: If the given plots do not all share one embedding_dim.
    """

    def __init__(self, *plots: Plot):
        dims = {p.embedding_dim for p in plots}
        if len(dims) > 1:
            raise ValueError(
                "Overlay requires every plot to share one embedding_dim, "
                f"got {sorted(dims, key=str)}."
            )
        self._plots = list(plots)

    @property
    def plots(self) -> list[Plot]:
        return list(self._plots)

    @property
    def embedding_dim(self) -> int | None:
        """The embedding_dim shared by every plot in this overlay.

        Returns:
            int | None: The shared embedding_dim, or None if this overlay
                is empty.
        """
        return self._plots[0].embedding_dim if self._plots else None

    def remove(self, plot: Plot) -> bool:
        """Removes `plot` from this overlay, if present.

        Args:
            plot (Plot): The plot to remove.

        Returns:
            bool: True if `plot` was found and removed, False otherwise.
        """
        if plot not in self._plots:
            return False
        self._plots.remove(plot)
        return True

    def replace(self, old: Plot, new: Plot) -> bool:
        """Replaces `old` with `new` in this overlay, if `old` is present.

        Args:
            old (Plot): The plot to replace.
            new (Plot): The plot to put in its place.

        Returns:
            bool: True if `old` was found and replaced, False otherwise.

        Raises:
            ValueError: If `new` does not share the overlay's
                embedding_dim.
        """
        if old not in self._plots:
            return False
        candidate = list(self._plots)
        candidate[candidate.index(old)] = new
        dims = {p.embedding_dim for p in candidate}
        if len(dims) > 1:
            raise ValueError(
                "Overlay requires every plot to share one embedding_dim, "
                f"got {sorted(dims, key=str)}."
            )
        self._plots = candidate
        return True


class _MultiItemLayout(Layout):
    """Shared base for Row and Column: stores items and flattens .plots."""

    def __init__(self, *items: Plot | Layout):
        self.items = list(items)

    @property
    def plots(self) -> list[Plot]:
        flat: list[Plot] = []
        for item in self.items:
            if isinstance(item, Layout):
                flat.extend(item.plots)
            else:
                flat.append(item)
        return flat


class Row(_MultiItemLayout):
    """Places its items side by side, left to right.

    Args:
        *items (Plot | Layout): The plots or layouts to place, in order.
    """


class Column(_MultiItemLayout):
    """Places its items one above another, top to bottom.

    Args:
        *items (Plot | Layout): The plots or layouts to place, in order.
    """


def normalize(item: Plot | Layout) -> list[list[Overlay]]:
    """Normalizes a Plot or Layout into a rectangular grid of Overlays.

    Args:
        item (Plot | Layout): The plot or layout to normalize.

    Returns:
        list[list[Overlay]]: Rows of Overlays. Irregular grids (e.g. a
            Column mixing single plots with multi-column Rows) are padded
            with empty Overlays, not spanned.

    Raises:
        TypeError: If item is neither a Plot nor a Layout.
    """
    if isinstance(item, Overlay):
        return [[item]]
    if isinstance(item, Plot):
        return [[Overlay(item)]]
    if isinstance(item, Row):
        return _compose_row([normalize(child) for child in item.items])
    if isinstance(item, Column):
        return _compose_column([normalize(child) for child in item.items])
    raise TypeError(f"Cannot normalize {type(item).__name__} into a layout grid.")


def _compose_row(grids: list[list[list[Overlay]]]) -> list[list[Overlay]]:
    """Combines sub-grids side by side, row-aligned. A sub-grid shorter
    (fewer rows) than the tallest is padded with empty Overlays at the
    bottom."""
    if not grids:
        return [[]]
    n_rows = max(len(grid) for grid in grids)
    result: list[list[Overlay]] = [[] for _ in range(n_rows)]
    for grid in grids:
        width = len(grid[0]) if grid else 0
        for row_idx in range(n_rows):
            if row_idx < len(grid):
                result[row_idx].extend(grid[row_idx])
            else:
                result[row_idx].extend(Overlay() for _ in range(width))
    return result


def _compose_column(grids: list[list[list[Overlay]]]) -> list[list[Overlay]]:
    """Stacks sub-grids vertically. A row narrower (fewer columns) than
    the widest is padded with empty Overlays on the right."""
    rows: list[list[Overlay]] = [row for grid in grids for row in grid]
    if not rows:
        return [[]]
    n_cols = max(len(row) for row in rows)
    return [row + [Overlay() for _ in range(n_cols - len(row))] for row in rows]
