from __future__ import annotations

from qewton.visualization.axes import POSITION_CHANNELS, FacetAxis, PositionAxis
from qewton.visualization.plots.base import Plot, PointEncoding


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
    """Plots drawn into one cell, in one coordinate system.

    Its dimension is the plots' one. Plots showing values by color at the
    points of their domain are drawn first, the others on top, each group
    in the given order. Facet axes of its plots repeat it within its cell;
    the copies share its position axes, and plots without facet axes are
    drawn in every copy.

    Args:
        *plots: The plots, all of one dimension; those with facet axes all
            with the same.
        x, y, z: The position axes; new linear ones if None.
        title: Title of the cell; None joins the plots' titles.

    Raises:
        ValueError: If the plots differ in dimension, or two plots have
            different facet axes.

    Attributes:
        plots (list of Plot): The plots, in drawing order.
        axes (dict): ``{channel: PositionAxis}`` for its dimension.
        dim (int or None): 2 or 3; None while empty.
        facets (list of FacetAxis): The facet axes of its plots.
        scaled_channels (tuple of str): The channels every plot draws to
            one scale.
        title (str or None): Title of the cell.
    """

    def __init__(
        self,
        *plots: Plot,
        x: PositionAxis | None = None,
        y: PositionAxis | None = None,
        z: PositionAxis | None = None,
        title: str | None = None,
    ) -> None:
        dims = {plot.dim for plot in plots}
        if len(dims) > 1:
            raise ValueError(
                f"An Overlay draws plots of one dimension, got {sorted(dims)}."
            )
        facets = [f for f in (_facets(plot) for plot in plots) if f]
        if any(f != facets[0] for f in facets):
            raise ValueError(
                "Plots of an Overlay with facet axes must all have the same; "
                "plots without are drawn in every facet cell."
            )
        self._plots = sorted(
            plots, key=lambda plot: plot.point_encoding is not PointEncoding.COLOR
        )
        self._axes = {
            "x": x or PositionAxis(),
            "y": y or PositionAxis(),
            "z": z or PositionAxis(),
        }
        self._title = title

    @property
    def plots(self) -> list[Plot]:
        return list(self._plots)

    @property
    def dim(self) -> int | None:
        return self._plots[0].dim if self._plots else None

    @property
    def axes(self) -> dict[str, PositionAxis]:
        return {c: self._axes[c] for c in POSITION_CHANNELS[: self.dim or 0]}

    @property
    def facets(self) -> list[FacetAxis]:
        return next((f for f in map(_facets, self._plots) if f), [])

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        if not self._plots:
            return ()
        common = set.intersection(*(set(p.scaled_channels) for p in self._plots))
        return tuple(c for c in POSITION_CHANNELS if c in common)

    @property
    def title(self) -> str | None:
        if self._title is not None:
            return self._title
        titles = [plot.title for plot in self._plots if plot.title]
        return " / ".join(titles) if titles else None

    def axis_title(self, channel: str) -> str | None:
        """The title of a position axis: the explicit one, else the title
        all plots give it, else None.

        Args:
            channel: ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            str or None: The title.
        """
        axis = self._axes[channel]
        if axis.title is not None:
            return axis.title
        titles = {plot.axis_title(channel) for plot in self._plots}
        return titles.pop() if len(titles) == 1 else None

    def axis_ticks(self, channel: str) -> tuple[list[float], list[str] | None] | None:
        """The ticks of a position axis: the explicit positions, else the
        ticks the plots give it if they agree, else None for automatic
        ones.

        Args:
            channel: ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            tuple or None: Tick positions and their labels (None to label
                them with their values).
        """
        axis = self._axes[channel]
        if axis.ticks is not None:
            return list(axis.ticks), None
        ticks = [t for t in (p.axis_ticks(channel) for p in self._plots) if t is not None]
        if not ticks or any(t != ticks[0] for t in ticks):
            return None
        return ticks[0]


def auto_layout(plots: list[Plot]) -> Overlay | Row:
    """Place plots in one Overlay where they read well together, else
    side by side.

    Plots of one dimension that give their position axes the same titles
    share an Overlay if no two of them show values at the points of the
    domain the same way (see
    :attr:`~qewton.visualization.plots.base.Plot.point_encoding`), e.g.
    arrows of u over a heatmap of p, or lines of one quantity from several
    sources. Otherwise they are placed side by side, e.g. two heatmaps or
    two vector fields of a reference and a prediction; they share their
    position axes there, so their ranges, zoom and camera go together.

    Args:
        plots: The plots, at least one.

    Returns:
        Overlay or Row: An Overlay of all plots, or a Row of them.
    """
    dims = {plot.dim for plot in plots}
    encodings = [p.point_encoding for p in plots if p.point_encoding is not None]
    titles = {
        tuple(plot.axis_title(c) for c in POSITION_CHANNELS[: plot.dim]) for plot in plots
    }
    if len(dims) > 1 or len(titles) > 1:
        return Row(*plots)
    if len(encodings) == len(set(encodings)):
        return Overlay(*plots)
    shared = {channel: PositionAxis() for channel in POSITION_CHANNELS}
    return Row(*(Overlay(plot, **shared) for plot in plots))


def _facets(plot: Plot) -> list[FacetAxis]:
    return [c for c in plot.controls if isinstance(c, FacetAxis)]


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
