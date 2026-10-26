"""The Figure, which arranges Plots in a panel grid and draws them."""

import warnings

from qewton.visualization.layout import Layout, Overlay, Row, normalize
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.spec import (
    AxisSpec,
    ControlSpec,
    FacetSpec,
    Scale,
    SliderSpec,
    TimeSpec,
)
from qewton.visualization.renderers.base import Artist, Renderer
from qewton.visualization.themes.base import Theme
from qewton.visualization.renderers import DEFAULT_RENDERER
from qewton.visualization.themes import DEFAULT_THEME


class Figure:
    """Draws one or more Plots into a single backend figure.

    Plots are arranged in a grid of panels (see
    :class:`~qewton.visualization.layout.Overlay`,
    :class:`~qewton.visualization.layout.Row` and
    :class:`~qewton.visualization.layout.Column`). A panel is split into a
    facet grid when one of its plots declares a
    :class:`~qewton.visualization.plots.spec.FacetSpec`, and the figure is
    animated when a plot declares a
    :class:`~qewton.visualization.plots.spec.TimeSpec`.

    Args:
        plots: A single Plot, a Layout (Overlay/Row/Column, arbitrarily
            nested), a list of Plots (equivalent to ``Row(*plots)``), or
            None to add plots later via :meth:`add_plot`.
        renderer: The renderer used to draw and export the figure.
        theme: The theme applied to every plot of this figure.
        title (str, optional): Figure title.

    Attributes:
        renderer (Renderer): The renderer used to draw and export the
            figure.
        theme (Theme): The theme applied to every plot of this figure.
        title (str | None): Figure title.
        panels (list[list[Overlay]]): The panel grid, row by row.
        plots (list[Plot]): Every plot of this figure, in the order they
            were added.
        controls (list[ControlSpec]): The distinct controls of all plots,
            except FacetSpecs.
        selector_specs (list[SelectorSpec]): The distinct SelectorSpecs
            embedded in all plots.
        artists (dict): The Artist per plot and facet cell, keyed by plot,
            then by ``(row_index, col_index)``; ``(None, None)`` without
            facets.
        backend_figure (object): The renderer's figure object, e.g. a
            Plotly figure.
    """

    def __init__(
        self,
        plots: Plot | Layout | list[Plot] | None = None,
        renderer: Renderer = DEFAULT_RENDERER,
        theme: Theme = DEFAULT_THEME,
        title=None,
    ):
        self.renderer = renderer
        self.theme = theme
        self.title = title
        self.panels: list[list[Overlay]] = []
        self.plots = []
        self.controls = []
        self.selector_specs = []

        if plots is not None:
            if isinstance(plots, list):
                plots = Row(*plots)
            self.panels = normalize(plots)
            for row in self.panels:
                for overlay in row:
                    for plot in overlay.plots:
                        self._register_plot(plot)

        # One artist per (plot, cell) - the non-faceted case degenerates to a
        # single {(None, None): artist} entry per plot.
        self.artists: dict[Plot, dict[tuple, Artist]] = {}
        self.backend_figure = renderer.setup(self)

    def add_plot(self, plot: Plot):
        """Add `plot` as a new panel in a new row of the panel grid.

        The plot gets this figure's theme and a ``color_index`` (its
        position among all plots of the figure), and its controls and
        SelectorSpecs are registered.

        Args:
            plot: The plot to add.
        """
        self.panels.append([Overlay(plot)])
        self._register_plot(plot)
        self._sync_backend_figure()

    def _sync_backend_figure(self):
        """Rebuild backend_figure and drop every cached Artist.

        The backend figure fixes the grid shape and cell types when built,
        so it is rebuilt whenever the panel grid changes; the next draw()
        recreates the artists.
        """
        self.backend_figure = self.renderer.setup(self)
        self.artists = {}

    def _register_plot(self, plot: Plot):
        """Apply this Figure's theme and collect `plot`'s controls and
        SelectorSpecs."""
        plot.theme = self.theme
        plot.color_index = len(self.plots)
        self.plots.append(plot)
        for spec in plot.controls:
            if isinstance(spec, ControlSpec) and not isinstance(spec, FacetSpec):
                if spec not in self.controls:
                    self.controls.append(spec)
        for spec in plot.selector_specs:
            if spec not in self.selector_specs:
                self.selector_specs.append(spec)

    def _rebuild_controls_and_selector_specs(self):
        """Recompute self.controls/self.selector_specs from self.plots."""
        self.controls = []
        self.selector_specs = []
        for plot in self.plots:
            for spec in plot.controls:
                if isinstance(spec, ControlSpec) and not isinstance(spec, FacetSpec):
                    if spec not in self.controls:
                        self.controls.append(spec)
            for spec in plot.selector_specs:
                if spec not in self.selector_specs:
                    self.selector_specs.append(spec)

    def remove_plot(self, plot: Plot):
        """Remove `plot` from this figure's panel grid.

        Args:
            plot: The plot to remove.

        Raises:
            ValueError: If `plot` is not part of this figure.
        """
        for row in self.panels:
            for overlay in row:
                if overlay.remove(plot):
                    self.plots.remove(plot)
                    self._rebuild_controls_and_selector_specs()
                    self._sync_backend_figure()
                    return
        raise ValueError(f"{plot} is not part of this Figure.")

    def replace_plot(self, old: Plot, new: Plot):
        """Replace `old` with `new` in the same panel position.

        `new` gets this figure's theme and keeps `old`'s ``color_index``.

        Args:
            old: The plot of this figure to replace.
            new: The plot to put in its place.

        Raises:
            ValueError: If `old` is not part of this figure, or if `new`
                does not share the ``embedding_dim`` of `old`'s Overlay.
        """
        for row in self.panels:
            for overlay in row:
                if old in overlay.plots:
                    overlay.replace(old, new)
                    self.plots[self.plots.index(old)] = new
                    new.theme = self.theme
                    new.color_index = old.color_index
                    self._rebuild_controls_and_selector_specs()
                    self._sync_backend_figure()
                    return
        raise ValueError(f"{old} is not part of this Figure.")

    @staticmethod
    def facet_specs(plot: Plot) -> dict[str, FacetSpec]:
        """Collect the FacetSpecs of `plot` by orientation.

        Args:
            plot: The plot whose controls are searched.

        Returns:
            The FacetSpec per orientation (``"row"`` and/or ``"col"``).

        Raises:
            AssertionError: If `plot` has more than one FacetSpec for one
                orientation.
        """
        specs = [s for s in plot.controls if isinstance(s, FacetSpec)]
        by_orientation = {s.orientation: s for s in specs}
        assert len(by_orientation) == len(
            specs
        ), "At most one FacetSpec per orientation (row/col) allowed per plot."
        return by_orientation

    def _facet_extent(self) -> tuple[int, int]:
        """(facet_rows, facet_cols): the largest facet grid of any plot.
        Every panel's block of cells has this size."""
        facet_rows, facet_cols = 1, 1
        for plot in self.plots:
            facets = self.facet_specs(plot)
            if "row" in facets:
                facet_rows = max(facet_rows, len(facets["row"].values))
            if "col" in facets:
                facet_cols = max(facet_cols, len(facets["col"].values))
        return facet_rows, facet_cols

    def grid_shape(self) -> tuple[int, int]:
        """Compute the size of the figure's cell grid.

        Every panel gets a block of the largest facet grid of any plot. A
        plot without a FacetSpec on an orientation draws into the first
        row or column of its block only.

        Returns:
            ``(n_rows, n_cols)``: panel rows times facet rows, panel
            columns times facet columns.
        """
        facet_rows, facet_cols = self._facet_extent()
        panel_rows = len(self.panels) if self.panels else 1
        panel_cols = max((len(row) for row in self.panels), default=1)
        return panel_rows * facet_rows, panel_cols * facet_cols

    def cell_titles(self, n_rows: int, n_cols: int) -> list[str]:
        """Compute the subplot title of every grid cell.

        A cell's title joins the distinct, non-None titles of its panel's
        plots, followed by the facet labels of that cell if any. Cells no
        plot draws into get ``""``.

        Args:
            n_rows: Number of grid rows, as returned by :meth:`grid_shape`.
            n_cols: Number of grid columns, as returned by
                :meth:`grid_shape`.

        Returns:
            One title per cell, in row-major order.
        """
        titles = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                row_off = panel_row * facet_rows
                col_off = panel_col * facet_cols
                panel_title = ", ".join(
                    dict.fromkeys(p.title for p in overlay.plots if p.title is not None)
                )
                cells_reached = set()
                for plot in overlay.plots:
                    facets = self.facet_specs(plot)
                    row_indices = (
                        range(len(facets["row"].values)) if "row" in facets else [0]
                    )
                    col_indices = (
                        range(len(facets["col"].values)) if "col" in facets else [0]
                    )
                    cells_reached.update((r, c) for r in row_indices for c in col_indices)
                for row_idx, col_idx in cells_reached or {(0, 0)}:
                    titles[row_off + row_idx][col_off + col_idx] = panel_title
                for plot in overlay.plots:
                    facets = self.facet_specs(plot)
                    row_facet, col_facet = facets.get("row"), facets.get("col")
                    row_indices = range(len(row_facet.values)) if row_facet else [0]
                    col_indices = range(len(col_facet.values)) if col_facet else [0]
                    for row_idx in row_indices:
                        for col_idx in col_indices:
                            facet_parts = [
                                facet.labels[idx]
                                for facet, idx in (
                                    (row_facet, row_idx),
                                    (col_facet, col_idx),
                                )
                                if facet is not None and facet.labels is not None
                            ]
                            if not facet_parts:
                                continue
                            cell = (row_off + row_idx, col_off + col_idx)
                            facet_title = ", ".join(facet_parts)
                            existing = titles[cell[0]][cell[1]]
                            titles[cell[0]][cell[1]] = (
                                f"{existing}, {facet_title}" if existing else facet_title
                            )
        return [title for row in titles for title in row]

    def cell_y_titles(self, n_rows: int, n_cols: int) -> list[str | None]:
        """Compute the y-axis title override of every grid cell.

        Considers the plots of the cell's panel whose ``y`` is an
        :class:`~qewton.visualization.plots.spec.AxisSpec`.

        Args:
            n_rows: Number of grid rows, as returned by :meth:`grid_shape`.
            n_cols: Number of grid columns, as returned by
                :meth:`grid_shape`.

        Returns:
            One entry per cell, in row-major order: the shared
            ``y.math_name`` if all these plots agree on it, ``""`` if they
            differ, and None if the panel has no such plot (the artists'
            own titles are kept).
        """
        titles: list[str | None] = [None] * (n_rows * n_cols)
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                names = {
                    p.y.math_name
                    for p in overlay.plots
                    if isinstance(getattr(p, "y", None), AxisSpec)
                }
                if not names:
                    continue
                title = names.pop() if len(names) == 1 else ""
                row_off = panel_row * facet_rows
                col_off = panel_col * facet_cols
                for dr in range(facet_rows):
                    for dc in range(facet_cols):
                        titles[(row_off + dr) * n_cols + (col_off + dc)] = title
        return titles

    def cell_dimensions(self, n_rows: int, n_cols: int) -> list[list[int]]:
        """Compute the embedding dimension of every grid cell.

        Args:
            n_rows: Number of grid rows, as returned by :meth:`grid_shape`.
            n_cols: Number of grid columns, as returned by
                :meth:`grid_shape`.

        Returns:
            The ``embedding_dim`` (2 or 3) of the plots drawing into each
            cell, row by row; 2 for cells no plot draws into.
        """
        dims = [[2] * n_cols for _ in range(n_rows)]
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                row_off = panel_row * facet_rows
                col_off = panel_col * facet_cols
                for plot in overlay.plots:
                    facets = self.facet_specs(plot)
                    row_indices = (
                        range(len(facets["row"].values)) if "row" in facets else [0]
                    )
                    col_indices = (
                        range(len(facets["col"].values)) if "col" in facets else [0]
                    )
                    for row_idx in row_indices:
                        for col_idx in col_indices:
                            dims[row_off + row_idx][
                                col_off + col_idx
                            ] = plot.embedding_dim
        return dims

    def cell_spans(self, n_rows: int, n_cols: int) -> list[list[tuple[int, int] | None]]:
        """Compute the row and column span of every grid cell.

        A panel without a FacetSpec on an orientation, in a figure whose
        blocks are faceted on that orientation, spans its whole block on
        that orientation. All other cells span ``(1, 1)``.

        Args:
            n_rows: Number of grid rows, as returned by :meth:`grid_shape`.
            n_cols: Number of grid columns, as returned by
                :meth:`grid_shape`.

        Returns:
            ``(rowspan, colspan)`` per cell, row by row; None for a cell
            covered by a spanning neighbor.
        """
        spans: list[list[tuple[int, int] | None]] = [
            [(1, 1) for _ in range(n_cols)] for _ in range(n_rows)
        ]
        absorbed = set()
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                if not overlay.plots:
                    continue
                row_off = panel_row * facet_rows
                col_off = panel_col * facet_cols
                has_row_facet = any("row" in self.facet_specs(p) for p in overlay.plots)
                has_col_facet = any("col" in self.facet_specs(p) for p in overlay.plots)
                rowspan = 1 if has_row_facet else facet_rows
                colspan = 1 if has_col_facet else facet_cols
                if rowspan == 1 and colspan == 1:
                    continue
                for plot in overlay.plots:
                    facets = self.facet_specs(plot)
                    row_indices = (
                        range(len(facets["row"].values)) if "row" in facets else [0]
                    )
                    col_indices = (
                        range(len(facets["col"].values)) if "col" in facets else [0]
                    )
                    for row_idx in row_indices:
                        for col_idx in col_indices:
                            origin = (row_off + row_idx, col_off + col_idx)
                            spans[origin[0]][origin[1]] = (rowspan, colspan)
                            for dr in range(rowspan):
                                for dc in range(colspan):
                                    if (dr, dc) != (0, 0):
                                        absorbed.add((origin[0] + dr, origin[1] + dc))
        for r, c in absorbed:
            spans[r][c] = None
        return spans

    def _animation_spec(self) -> TimeSpec | None:
        """The one TimeSpec of the whole Figure, or None. Animation frames
        are figure-wide, so at most one TimeSpec is supported."""
        specs = [s for s in self.controls if isinstance(s, TimeSpec)]
        assert len(specs) <= 1, "At most one TimeSpec per Figure is supported."
        return specs[0] if specs else None

    def _scales_in_use(self) -> list[Scale]:
        seen = []
        for plot in self.plots:
            spec = getattr(plot, "color", None)
            scale = getattr(spec, "scale", None) if spec is not None else None
            if scale is not None and scale not in seen:
                seen.append(scale)
        return seen

    def _plot_cells(self, plot: Plot) -> list[tuple[int, int]]:
        """Every 1-indexed (row, col) grid cell `plot` draws into."""
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                if plot not in overlay.plots:
                    continue
                row_off, col_off = panel_row * facet_rows, panel_col * facet_cols
                facets = self.facet_specs(plot)
                row_indices = range(len(facets["row"].values)) if "row" in facets else [0]
                col_indices = range(len(facets["col"].values)) if "col" in facets else [0]
                return [
                    (row_off + r + 1, col_off + c + 1)
                    for r in row_indices
                    for c in col_indices
                ]
        return []

    def _assign_colorbar_cells(self) -> None:
        """Set each in-use Scale's `colorbar_cell` to the rightmost grid
        cell of the plots using it, so a shared Scale gets one colorbar
        after all of them. No-op for a single-cell figure."""
        if self.grid_shape() == (1, 1):
            return
        for scale in self._scales_in_use():
            cells = [
                cell
                for plot in self.plots
                if getattr(plot, "color", None) is not None and plot.color.scale is scale
                for cell in self._plot_cells(plot)
            ]
            if cells:
                scale.colorbar_cell = max(cells, key=lambda cell: cell[1])

    def draw(self):
        """Draw every plot into the backend figure.

        Shared color scales are trained on all plots first, then each plot
        is drawn into its cells, and with a TimeSpec the animation frames
        are added. Artists from an earlier call are updated, not recreated.

        Returns:
            object: The renderer's figure object (:attr:`backend_figure`).
        """
        # Pass 1: reset and train shared scales before anything is drawn, so
        # no colorbar claim from a previous draw() remains.
        for scale in self._scales_in_use():
            scale.reset()
        for plot in self.plots:
            values = plot.color_values()
            if values is not None:
                plot.color.scale.observe(values)
        self._assign_colorbar_cells()

        # Pass 2: draw with the trained scales. In a grid, Plotly requires
        # row and col together, so a non-faceted orientation uses cell 1 of
        # the plot's block instead of None.
        is_grid = self.grid_shape() != (1, 1)
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                block_offset = (panel_row * facet_rows, panel_col * facet_cols)
                for plot in overlay.plots:
                    self._draw_plot(plot, is_grid, block_offset)

        # Reconcile y-axis titles per panel after all artists set theirs: a
        # single artist can't see its Overlay siblings (see cell_y_titles()).
        n_rows, n_cols = self.grid_shape()
        self.renderer.reconcile_y_axis_titles(self, self.backend_figure, n_rows, n_cols)

        # Pass 3 (only with a TimeSpec): the renderer adds one frame per
        # animated state on top of the drawn figure.
        spec = self._animation_spec()
        if spec is not None:
            self.renderer.animate(self, self.backend_figure, spec)

        return self.backend_figure

    def _draw_plot(
        self, plot: Plot, is_grid: bool, block_offset: tuple[int, int] = (0, 0)
    ):
        """Draw one plot into every grid cell of its FacetSpec(s), offset by
        `block_offset` (its panel's top-left cell). Each FacetSpec's state is
        set per cell and restored afterwards."""
        facets = self.facet_specs(plot)
        row_spec = facets.get("row")
        col_spec = facets.get("col")
        row_values = row_spec.values if row_spec is not None else [None]
        col_values = col_spec.values if col_spec is not None else [None]
        row_off, col_off = block_offset

        cell_artists = self.artists.setdefault(plot, {})
        original_row_state = row_spec.state if row_spec is not None else None
        original_col_state = col_spec.state if col_spec is not None else None
        try:
            for row_idx, row_value in enumerate(row_values):
                if row_spec is not None:
                    row_spec.state = row_value
                for col_idx, col_value in enumerate(col_values):
                    if col_spec is not None:
                        col_spec.state = col_value

                    key = (
                        row_idx if row_spec is not None else None,
                        col_idx if col_spec is not None else None,
                    )
                    if is_grid:
                        row = row_off + (row_idx if row_spec is not None else 0) + 1
                        col = col_off + (col_idx if col_spec is not None else 0) + 1
                    else:
                        row = col = None

                    artist = cell_artists.get(key)
                    if artist is None:
                        artist = plot.create_artist(
                            self.backend_figure, self.renderer, row=row, col=col
                        )
                        cell_artists[key] = artist
                    else:
                        artist.update(self.backend_figure, plot, row=row, col=col)
        finally:
            if row_spec is not None:
                row_spec.state = original_row_state
            if col_spec is not None:
                col_spec.state = original_col_state

    def _apply_selectors(self):
        """Bake every SelectorSpec into the backend figure as a dropdown.
        Called by show()/save_html() only, not by draw(), since
        DashApplication provides its own dropdowns."""
        for spec in self.selector_specs:
            self.renderer.apply_selector(self, self.backend_figure, spec)

    def _apply_sliders(self):
        """Bake a single SliderSpec into the backend figure (see
        Renderer.apply_slider()). With several SliderSpecs, only the initial
        state is shown and a warning is issued. Called by show()/save_html()
        only, like _apply_selectors().
        """
        sliders = [c for c in self.controls if isinstance(c, SliderSpec)]
        if len(sliders) == 1:
            self.renderer.apply_slider(self, self.backend_figure, sliders[0])
        elif len(sliders) > 1:
            warnings.warn(
                f"Figure has {len(sliders)} SliderSpecs "
                f"({', '.join(s.name for s in sliders)}) - show()/save_html() "
                "can only bake a single slider into a static figure, so only "
                "the initial state is shown. Use DashApplication for "
                "interactive sliders.",
                stacklevel=3,
            )

    def show(self):
        """Draw the figure and display it, e.g. in a browser tab.

        One SliderSpec and all SelectorSpecs are baked into the figure as
        static widgets; with several SliderSpecs, only their initial state
        is shown and a warning is issued.
        """
        self.draw()
        self._apply_selectors()
        self._apply_sliders()
        self.renderer.show(self.backend_figure)

    def save_html(self, path):
        """Draw the figure and write it to an interactive HTML file.

        Widgets are baked in as for :meth:`show`.

        Args:
            path (str | os.PathLike): The output file.
        """
        self.draw()
        self._apply_selectors()
        self._apply_sliders()
        self.renderer.save_html(self.backend_figure, path)

    def save_gif(self, path, fps=10):
        """Draw the figure and write its animation to an animated GIF.

        Requires a TimeSpec in the figure: each animation frame is
        rasterized and becomes one GIF frame.

        Args:
            path (str | os.PathLike): The output file.
            fps (float): Frame rate for equally spaced time steps. With
                non-uniform time steps, each frame is shown in proportion to
                its time step (see
                :meth:`~qewton.visualization.plots.spec.TimeSpec.frame_weights`).
                Defaults to 10.
        """
        self.draw()
        spec = self._animation_spec()
        self.renderer.save_gif(
            self.backend_figure,
            path,
            fps=fps,
            frame_weights=spec.frame_weights() if spec is not None else None,
        )

    def save_png(self, path, **kwargs):
        """Draw the figure and write its current state to a PNG file.

        Args:
            path (str | os.PathLike): The output file.
            **kwargs: Forwarded to the renderer's ``save_image``.
        """
        self.draw()
        self.renderer.save_image(self.backend_figure, path, format="png", **kwargs)

    def save_svg(self, path, **kwargs):
        """Draw the figure and write its current state to an SVG file.

        Args:
            path (str | os.PathLike): The output file.
            **kwargs: Forwarded to the renderer's ``save_image``.
        """
        self.draw()
        self.renderer.save_image(self.backend_figure, path, format="svg", **kwargs)
