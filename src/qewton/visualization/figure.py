"""The Figure, which arranges plots in a grid of cells and draws them."""

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np

from qewton.visualization.axes import (
    Control,
    FixedAxis,
    PositionAxis,
    Selector,
    SliderAxis,
    TimeAxis,
)
from qewton.visualization.layout import Layout, Overlay, Row, normalize
from qewton.visualization.plots.base import Plot
from qewton.visualization.renderers import DEFAULT_RENDERER
from qewton.visualization.renderers.base import Renderer
from qewton.visualization.scales import ColorScale, color_range, log_ticks


@dataclass(eq=False)
class Cell:
    """One drawn copy of an Overlay in the figure grid.

    Args:
        row: Grid row, 1-indexed.
        col: Grid column, 1-indexed.
        overlay: The overlay drawn here.
        facet_states: ``{FacetAxis: state}`` of this copy.

    Attributes:
        dim (int): Number of position components: 2 or 3, or 0 for plots
            that lay out their own axes (e.g. parallel coordinates).
        axes (dict): ``{channel: PositionAxis}`` of the overlay.
        title (str or None): The overlay's title, followed by the facet
            states and the states of fixed axes, e.g. ``"u, t = 0.5"``.
        scaled_channels (tuple of str): The channels drawn to one scale.
        aspect (float or None): Height over width of the plot area of a 2D
            cell drawing x and y to one scale, else None.
        box_aspect (tuple of float or None): Relative lengths of the x, y
            and z sides of a 3D cell's box, else None.
    """

    row: int
    col: int
    overlay: Overlay
    facet_states: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.overlay.dim

    @property
    def axes(self) -> dict[str, PositionAxis]:
        return self.overlay.axes

    @property
    def scaled_channels(self) -> tuple[str, ...]:
        return self.overlay.scaled_channels

    @property
    def aspect(self) -> float | None:
        # The ratio of the spans of y and x (1 while a range is unknown).
        if self.dim != 2 or not {"x", "y"} <= set(self.scaled_channels):
            return None
        spans = _spans(self.axes)
        return spans["y"] / spans["x"] if spans is not None else 1.0

    @property
    def box_aspect(self) -> tuple[float, float, float] | None:
        # The data's spans for the channels drawn to one scale; a channel
        # of its own (e.g. a height over a plane) as long as the widest
        # scaled one; a cube if fewer than two are scaled; None (automatic)
        # while a range is unknown.
        if self.dim != 3:
            return None
        spans = _spans(self.axes)
        if spans is None:
            return None
        scaled = self.scaled_channels
        if len(scaled) < 2:
            return (1.0, 1.0, 1.0)
        widest = max(spans[c] for c in scaled)
        return tuple(spans[c] / widest if c in scaled else 1.0 for c in ("x", "y", "z"))

    @property
    def title(self) -> str | None:
        parts = [self.overlay.title]
        parts += [
            f"{facet.name} = {facet.label(s)}" for facet, s in self.facet_states.items()
        ]
        parts += [
            f"{c.name} = {c.label(c.state)}"
            for plot in self.overlay.plots
            for c in plot.controls
            if isinstance(c, FixedAxis)
        ]
        parts = list(dict.fromkeys(p for p in parts if p))  # each text once, in order
        return ", ".join(parts) if parts else None

    def axis_title(self, channel: str) -> str | None:
        """The title of a position axis; see
        :meth:`~qewton.visualization.layout.Overlay.axis_title`."""
        return self.overlay.axis_title(channel)

    def axis_range(self, channel: str) -> tuple[float, float] | None:
        """The drawn range of a position axis: its scale's range, widened
        on both sides by the overlay's margin of its span (in log space for
        a log scale).

        Args:
            channel: ``"x"``, ``"y"`` or ``"z"``.

        Returns:
            tuple of float or None: ``(lo, hi)``; None for a categorical
                axis, without a range or for an empty one.
        """
        scale = self.axes[channel].scale
        if scale.categorical or scale.range is None:
            return None
        lo, hi = scale.transform(np.asarray(scale.range))
        if not hi > lo:
            return None
        pad = self.overlay.margin * (hi - lo)
        lo, hi = scale.inverse([lo - pad, hi + pad])
        return float(lo), float(hi)

    def axis_ticks(self, channel: str) -> tuple | None:
        """The ticks of a position axis; see
        :meth:`~qewton.visualization.layout.Overlay.axis_ticks`."""
        return self.overlay.axis_ticks(channel)


class Figure:
    """Draws plots, arranged in a layout, with a renderer.

    Each overlay of the layout is one cell of the grid, repeated per state
    of its facet axes. Drawing trains all scales over every state of the
    plots' controls, then draws one artist per Result into its cell, the
    position axes of every cell and one colorbar per ColorScale.

    A control is shown in one of two ways: *connected* to a widget of an
    application, which sets its state and draws again, or *embedded* into
    the output, which then holds every state (:meth:`draw` with
    `embedded`). An embedded Selector draws every candidate and shows the
    chosen one. Per plot at most one control axis and one Selector are
    embedded, since their states would overwrite each other.

    Args:
        plots: A Plot, a Layout, or a list of plots (a Row).
        renderer: The renderer drawing and exporting the figure, used by
            this figure only; a new default renderer if None.
        title: Title of the figure.

    Raises:
        ValueError: With more than one TimeAxis, an overlay with more than
            one facet axis per orientation, or a renderer already drawing
            another figure.

    Attributes:
        renderer (Renderer): The renderer.
        title (str or None): The title.
        plots (list of Plot): Every plot, in reading order.
        cells (list of Cell): The cells of the grid.
        grid_shape (tuple of int): Number of grid rows and columns.
        controls (list of Control): Every control of the plots (control
            axes and selectors), each shared instance once.
        color_scales (list of ColorScale): Every color scale of the plots,
            each shared instance once.
        backend_figure (object): The renderer's figure.
    """

    def __init__(
        self,
        plots: Plot | Layout | list[Plot],
        renderer: Renderer | None = None,
        title: str | None = None,
    ) -> None:
        self.renderer = renderer if renderer is not None else DEFAULT_RENDERER()
        self.title = title
        self._grid = normalize(Row(*plots) if isinstance(plots, list) else plots)
        self.plots = [
            plot for row in self._grid for overlay in row for plot in overlay.plots
        ]
        self._assign_colors()
        self._check_controls()
        self.cells, self.grid_shape = self._layout_cells()
        self._embedded: list[Control] = []
        self._by_visibility: set = set()
        self._keys: set = set()
        self._entries: dict = {}
        self.renderer.setup(self)

    @property
    def backend_figure(self):
        return self.renderer.backend_figure

    # --- structure -----------------------------------------------------------

    def _assign_colors(self) -> None:
        index = 0
        for plot in self.plots:
            plot.color_index = index
            index += plot.n_copies

    def _check_controls(self) -> None:
        times = [c for c in self.controls if isinstance(c, TimeAxis)]
        if len(times) > 1:
            raise ValueError(f"A Figure supports one TimeAxis, got {len(times)}.")
        for row in self._grid:
            for overlay in row:
                orientations = [f.orientation for f in overlay.facets]
                if len(orientations) != len(set(orientations)):
                    raise ValueError(
                        "An overlay takes at most one FacetAxis per orientation."
                    )

    @property
    def controls(self) -> list[Control]:
        return _unique(c for plot in self.plots for c in plot.controls + plot.selectors)

    @property
    def color_scales(self) -> list[ColorScale]:
        return _unique(s for plot in self.plots for s in plot.color_scales)

    def _layout_cells(self) -> tuple[list[Cell], tuple[int, int]]:
        # Every overlay takes a block of cells, one per facet state; rows
        # and columns of the grid are as large as their largest block.
        blocks = [[_facet_block(overlay) for overlay in row] for row in self._grid]
        heights = [max((len(b[0]) for b in row), default=1) for row in blocks]
        widths = [
            max((len(blocks[r][c][1]) for r in range(len(blocks))), default=1)
            for c in range(len(blocks[0]))
        ]
        cells = []
        for r, row in enumerate(self._grid):
            for c, overlay in enumerate(row):
                if not overlay.plots:
                    continue
                row_states, col_states = blocks[r][c]
                for i, row_state in enumerate(row_states):
                    for j, col_state in enumerate(col_states):
                        states = {**row_state, **col_state}
                        row_index, col_index = (
                            sum(heights[:r]) + i + 1,
                            sum(widths[:c]) + j + 1,
                        )
                        cells.append(Cell(row_index, col_index, overlay, states))
        return cells, (sum(heights), sum(widths))

    def plot_sizes(self) -> tuple[list[float], list[float]]:
        """The size of the plot areas, in pixels: the width of every grid
        column and the height of every row. A cell of a fixed aspect fits
        it into the theme's plot box (``plot_width`` x ``plot_height``),
        any other fills the box; a column is as wide as its widest cell, a
        row as tall as its tallest, an empty one takes no space. Follows
        the trained scales.

        Returns:
            tuple: Column widths and row heights.
        """
        theme = self.renderer.theme
        box_width, box_height = theme.plot_width, theme.plot_height
        n_rows, n_cols = self.grid_shape
        widths, heights = [0.0] * n_cols, [0.0] * n_rows
        for cell in self.cells:
            width, height = box_width, box_height
            if cell.aspect is not None:
                if cell.aspect > box_height / box_width:
                    width = box_height / cell.aspect
                else:
                    height = box_width * cell.aspect
            widths[cell.col - 1] = max(widths[cell.col - 1], width)
            heights[cell.row - 1] = max(heights[cell.row - 1], height)
        return widths, heights

    def legends(self) -> list[tuple[dict, Cell]]:
        """The legends of the figure as drawn last: the entries of every
        cell are the labels of its visible Results with a legend entry
        (:attr:`Result.legend_entry
        <qewton.visualization.plots.result.Result.legend_entry>`), each
        once, in drawing order; cells
        of equal entries share one legend beside the last of them, as
        cells of a color scale share its colorbar. Fewer than two entries
        tell nothing apart and get none; nor does a theme without legends.

        Returns:
            list of tuple: Per legend, ``{label: keys}`` (the artists
                drawing every entry, in all its cells) and the cell it
                stands beside.
        """
        if not self.renderer.theme.show_legend:
            return []
        groups: dict = {}
        for index, entries in self._entries.items():
            if len(entries) >= 2:
                groups.setdefault(tuple(entries), []).append(index)
        legends = []
        for labels, indices in groups.items():
            entries = {
                label: [k for i in indices for k in self._entries[i][label]] for label in labels
            }
            legends.append((entries, _last([self.cells[i] for i in indices])))
        return legends

    def cell_at(self, row: int, col: int) -> Cell | None:
        """The cell at a grid position, or None for an empty position.

        Args:
            row: Grid row, 1-indexed.
            col: Grid column, 1-indexed.

        Returns:
            Cell or None: The cell.
        """
        return next((c for c in self.cells if (c.row, c.col) == (row, col)), None)

    def colorbar_cell(self, scale: ColorScale) -> Cell:
        """The cell the colorbar of `scale` stands beside: the last of its
        cells, rightmost, then topmost.

        Args:
            scale: A color scale of the figure.

        Returns:
            Cell: The cell.
        """
        return _last(self.cells_of(scale))

    def colorbar_slot(self, scale: ColorScale) -> int:
        """The place of the colorbar of `scale` beside its cell, counted
        from the plot area: one per plot with a colorbar there, in drawing
        order; the scales of one plot (e.g. of the candidates of a
        Selector) are never shown together and share it.

        Args:
            scale: A color scale of the figure.

        Returns:
            int: The slot, from 0.
        """
        cell = self.colorbar_cell(scale)
        owners = _unique(
            self._owner(s) for s in self.color_scales if self.colorbar_cell(s) is cell
        )
        return next(i for i, owner in enumerate(owners) if owner is self._owner(scale))

    def _owner(self, scale: ColorScale) -> Plot:
        # The first plot drawing with `scale`.
        return next(p for p in self.plots if scale in p.color_scales)

    def cells_of(self, scale: ColorScale) -> list[Cell]:
        """The cells with a plot using `scale`.

        Args:
            scale: A color scale of the figure.

        Returns:
            list of Cell: The cells, in grid order.
        """
        return [
            c for c in self.cells if any(scale in p.color_scales for p in c.overlay.plots)
        ]

    def color_title(self, scale: ColorScale) -> str | None:
        """The colorbar title of `scale`: the color title all plots drawing
        with it share in their current states, else None.

        Args:
            scale: A color scale of the figure.

        Returns:
            str or None: The title.
        """
        titles = {
            plot.color_title for plot in self.plots if scale in plot.active_color_scales
        }
        return titles.pop() if len(titles) == 1 else None

    def color_ticks(self, scale: ColorScale) -> tuple[list[float], list[str]] | None:
        """The colorbar ticks of `scale`, in its transformed values: the
        ticks all plots drawing with it give it (e.g. categories, also
        while it is not shown), else those of a log scale (values at their
        log10), else None for automatic ticks.

        Args:
            scale: A color scale of the figure.

        Returns:
            tuple or None: Positions and labels.
        """
        given = [p.color_ticks(scale) for p in self.plots if scale in p.color_scales]
        if given and given[0] is not None and all(t == given[0] for t in given):
            return given[0]
        if scale.log:
            return log_ticks(*color_range(scale))
        return None

    # --- drawing -------------------------------------------------------------

    def draw(self, embedded: list[Control] = ()):
        """Train the scales and draw every plot, axis, colorbar and legend
        in the current states of the controls; embed the states of
        `embedded`; then lay the figure out, with room for what every
        embedded state shows. Artists of an earlier call are updated, not
        recreated.

        Args:
            embedded: Controls whose states the output holds; facet,
                overlay and fixed axes need no embedding. TimeAxes come
                first, then SliderAxes, then Selectors. A control is
                skipped with a warning if the renderer does not embed its
                kind (:attr:`Renderer.embeddable
                <qewton.visualization.renderers.base.Renderer.embeddable>`),
                if the renderer's limit is reached (``max_embedded``), or
                if it would overwrite an earlier one on a common plot. A
                plot embeds one control axis; next to it one Selector,
                drawn as every candidate of which the chosen one is shown.
                Without one, it embeds several Selectors of different
                channels, each by its data.

        Returns:
            object: The renderer's figure (:attr:`backend_figure`).
        """
        self._embedded, self._by_visibility = self._embeddable(embedded)
        self._train_scales()
        self._entries = {}
        self.redraw()
        for entries, cell in self.legends():
            self.renderer.draw_legend(entries, cell)
        self.renderer.clear_embeddings()
        for control in self._embedded:
            self._embed(control)
        self.renderer.draw_layout()
        return self.backend_figure

    def redraw(self) -> None:
        """Draw every plot, axis and colorbar in the current states of the
        controls, without training the scales again."""
        drawn = self._draw(list(enumerate(self.cells)), self.plots, self.color_scales)
        for key in self._keys - drawn:
            self.renderer.hide(key)
        self._keys |= drawn

    def _draw(self, cells, plots, scales) -> set:
        # Draw `plots` in `cells` (index, cell), their axes and the colorbars
        # of `scales`; returns the keys of the drawn artists.
        keys = set()
        for index, cell in cells:
            originals = {facet: facet.state for facet in cell.facet_states}
            try:
                for facet, state in cell.facet_states.items():
                    facet.state = state
                for plot in cell.overlay.plots:
                    if plot in plots:
                        keys |= self._draw_plot(plot, index, cell)
            finally:
                for facet, state in originals.items():
                    facet.state = state
            self.renderer.draw_axes(cell)
        for scale in scales:
            active = any(scale in p.active_color_scales for p in self.plots)
            self.renderer.draw_colorbar(
                scale,
                self.color_title(scale),
                self.color_ticks(scale),
                self.colorbar_cell(scale),
                self.colorbar_slot(scale),
                visible=active,
            )
        return keys

    def _draw_plot(self, plot: Plot, index: int, cell: Cell) -> set:
        # One artist per copy and choice of the plot's selectors embedded by
        # visibility; only those of the current choice are visible.
        selectors = [s for s in plot.selectors if s in self._by_visibility]
        current = tuple(s.state for s in selectors)
        keys = set()
        try:
            for choice in itertools.product(*(s.states for s in selectors)):
                for selector, state in zip(selectors, choice):
                    selector.state = state
                for copy, result in enumerate(plot.evaluate()):
                    key = (id(plot), index, copy, choice)
                    visible = choice == current
                    self.renderer.draw(key, result, cell, visible=visible)
                    keys.add(key)
                    if visible and result.legend_entry:
                        entries = self._entries.setdefault(index, {})
                        entries.setdefault(result.label, []).append(key)
        finally:
            for selector, state in zip(selectors, current):
                selector.state = state
        return keys

    def _train_scales(self) -> None:
        overlays = _unique(cell.overlay for cell in self.cells)
        for scale in _unique(
            [s for o in overlays for s in (a.scale for a in o.axes.values())]
            + [s for plot in self.plots for s in plot.scales]
        ):
            scale.reset()
        for overlay in overlays:
            axes = {channel: axis.scale for channel, axis in overlay.axes.items()}
            for plot in overlay.plots:
                plot.train_scales(axes)

    # --- embedding -----------------------------------------------------------

    def _embeddable(self, requested) -> tuple[list[Control], set]:
        # The requested controls the renderer embeds, in _EMBEDDINGS order:
        # of a kind it embeds, up to its limit, without those overwriting
        # an earlier one on a common plot; and the selectors among them
        # embedded by visibility.
        kinds = list(_EMBEDDINGS)
        candidates = sorted(
            (c for c in requested if type(c) in _EMBEDDINGS),
            key=lambda c: kinds.index(type(c)),
        )
        accepted, skipped, by_visibility = [], [], set()
        for control in candidates:
            if (
                type(control) not in self.renderer.embeddable
                or len(accepted) == self.renderer.max_embedded
                or self._overwrites(control, accepted)
            ):
                skipped.append(control)
                continue
            accepted.append(control)
            if isinstance(control, Selector) and self._next_to_a_control_axis(
                control, accepted
            ):
                by_visibility.add(control)
        if skipped:
            warnings.warn(
                f"{', '.join(c.name for c in skipped)} cannot be embedded by "
                f"{type(self.renderer).__name__} next to the other controls; only "
                "the current state is shown. Use a connected output (an "
                "application) to change them.",
                stacklevel=3,
            )
        return accepted, by_visibility

    def _next_to_a_control_axis(self, selector: Selector, accepted) -> bool:
        # Whether a plot of `selector` embeds a control axis, which writes
        # all its data.
        return any(
            selector in p.selectors and any(c in p.controls for c in accepted)
            for p in self.plots
        )

    def _overwrites(self, control: Control, accepted) -> bool:
        # Whether embedding `control` next to the accepted controls would
        # overwrite states on a common plot: two control axes, a selector
        # next to a control axis and another selector, or two selectors of
        # one channel.
        for plot in self.plots:
            axes = [c for c in accepted if c in plot.controls]
            selectors = [c for c in accepted if c in plot.selectors]
            if control in plot.controls and axes:
                return True
            if control in plot.selectors and selectors:
                if axes:
                    return True
                channels = set(plot.channels_of(control))
                if any(channels & set(plot.channels_of(s)) for s in selectors):
                    return True
        return False

    def _embed(self, control: Control) -> None:
        # Draw each state of the plots depending on `control` and hand the
        # renderer's snapshots to its embedding.
        plots = [p for p in self.plots if control in p.controls + p.selectors]
        cells = [
            (i, c)
            for i, c in enumerate(self.cells)
            if any(p in plots for p in c.overlay.plots)
        ]
        scales = _unique(s for p in plots for s in p.color_scales)
        original = control.state
        snapshots = []
        try:
            for state in control.states:
                control.state = state
                keys = sorted(self._draw(cells, plots, scales))
                snapshots.append(
                    self.renderer.snapshot(keys, [c for _, c in cells], scales)
                )
        finally:
            control.state = original
            self._draw(cells, plots, scales)
        _EMBEDDINGS[type(control)](self.renderer)(control, snapshots)

    # --- outputs -------------------------------------------------------------

    def show(self) -> None:
        """Draw the figure with every control embedded and display it."""
        self.draw(embedded=self.controls)
        self.renderer.show()

    def save_html(self, path) -> None:
        """Draw the figure with every control embedded and write it as an
        interactive HTML file.

        Args:
            path (str or path-like): Output file path.
        """
        self.draw(embedded=self.controls)
        self.renderer.save_html(path)

    def save_png(self, path, **kwargs) -> None:
        """Draw the figure in its current states and write a PNG image.

        Args:
            path (str or path-like): Output file path.
            **kwargs: Passed on to the renderer's image export.
        """
        self.draw()
        self.renderer.save_image(path, format="png", **kwargs)

    def save_svg(self, path, **kwargs) -> None:
        """Draw the figure in its current states and write an SVG image.

        Args:
            path (str or path-like): Output file path.
            **kwargs: Passed on to the renderer's image export.
        """
        self.draw()
        self.renderer.save_image(path, format="svg", **kwargs)

    def save_gif(self, path, fps: float = 10) -> None:
        """Draw the figure with its TimeAxis embedded and write the
        animation as a looping GIF; other controls stay in their current
        states.

        Args:
            path (str or path-like): Output file path.
            fps: Frames per second of equally spaced time steps; each frame
                is shown in proportion to its time step (see
                :meth:`~qewton.visualization.axes.TimeAxis.frame_weights`).

        Raises:
            ValueError: If the figure has no TimeAxis.
        """
        times = [c for c in self.controls if isinstance(c, TimeAxis)]
        if not times:
            raise ValueError("save_gif needs a TimeAxis in the figure.")
        (time,) = times
        self.draw(embedded=[time])
        self.renderer.save_gif(path, fps=fps, frame_weights=time.frame_weights())


#: How a self-contained output embeds a kind of control: the renderer
#: method embedding its states.
_EMBEDDINGS = {
    TimeAxis: lambda renderer: renderer.animate,
    SliderAxis: lambda renderer: renderer.apply_slider,
    Selector: lambda renderer: renderer.apply_selector,
}


def _facet_block(overlay: Overlay) -> tuple[list[dict], list[dict]]:
    # The facet states of the rows and of the columns of an overlay's block.
    if not overlay.plots:
        return [{}], [{}]
    by_orientation = {f.orientation: f for f in overlay.facets}
    rows, cols = by_orientation.get("row"), by_orientation.get("col")
    row_states = [{rows: s} for s in rows.states] if rows is not None else [{}]
    col_states = [{cols: s} for s in cols.states] if cols is not None else [{}]
    return row_states, col_states


def _unique(items) -> list:
    # Items in order, each instance once.
    found = []
    for item in items:
        if not any(item is other for other in found):
            found.append(item)
    return found


def _spans(axes: dict) -> dict[str, float] | None:
    # The span of every position axis in its transformed values; None
    # while a range is unknown.
    spans = {}
    for channel, axis in axes.items():
        if axis.scale.range is None:
            return None
        lo, hi = axis.scale.transform(np.asarray(axis.scale.range))
        spans[channel] = max(float(hi - lo), 1e-12)
    return spans


def _last(cells: list) -> "Cell":
    # The cell beside which what several cells share stands: the
    # rightmost, then topmost.
    return max(cells, key=lambda c: (c.col, -c.row))
