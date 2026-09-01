from qewton.visualization.layout import Layout, Overlay, Row, normalize
from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.spec import ControlSpec, FacetSpec, Scale, TimeSpec
from qewton.visualization.renderers.base import Artist, Renderer
from qewton.visualization.themes.base import Theme
from qewton.visualization.renderers import DEFAULT_RENDERER
from qewton.visualization.themes import DEFAULT_THEME


class Figure:
    """Draws one or more Plots into a single backend figure. Plots are
    arranged in a grid of panels (see qewton.visualization.layout: Overlay,
    Row, Column), each panel further split into a facet grid when any of
    its plots declares a FacetSpec, and animated when any plot declares a
    TimeSpec.

    Args:
        plots: A single Plot, a Layout (Overlay/Row/Column, arbitrarily
            nested), a list of Plots (equivalent to Row(*plots)), or None
            to add plots later via `add_plot()`.
        renderer: The Renderer backend used to draw and export the figure.
        theme: The Theme applied to every plot that doesn't set its own.
        title: Optional figure title.
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
        self.variable_specs = []

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
        """Adds `plot` to this Figure as a new panel (a new row of the
        panel grid, one cell wide), applying the Figure's theme unless the
        plot already has its own, assigning it a color_index (its position
        among every plot in this Figure, in add order), and registering
        its controls and any VariableSpecs it embeds."""
        self.panels.append([Overlay(plot)])
        self._register_plot(plot)
        self._sync_backend_figure()

    def _sync_backend_figure(self):
        """Rebuilds backend_figure and drops every cached Artist.

        backend_figure is a Plotly object that bakes in a structural
        decision - plain canvas vs. a make_subplots() grid of a specific
        shape and per-cell type (2D/3D) - at the moment it's built (see
        PlotlyRenderer.setup()). add_plot()/remove_plot()/replace_plot()
        can each change grid_shape() or a cell's embedding_dim after the
        fact, so every one of them must rebuild rather than mutate the
        existing backend_figure in place. Cached artists are tied to
        positions in the old backend_figure, so they're dropped too - the
        next draw() recreates them against the new one.
        """
        self.backend_figure = self.renderer.setup(self)
        self.artists = {}

    def _register_plot(self, plot: Plot):
        """Applies this Figure's theme and collects `plot`'s controls and
        VariableSpecs - the bookkeeping shared by add_plot() and the
        Layout-driven registration in __init__()."""
        plot.theme = self.theme
        plot.color_index = len(self.plots)
        self.plots.append(plot)
        for spec in plot.controls:
            if isinstance(spec, ControlSpec) and not isinstance(spec, FacetSpec):
                if spec not in self.controls:
                    self.controls.append(spec)
        for spec in plot.variable_specs:
            if spec not in self.variable_specs:
                self.variable_specs.append(spec)

    def _rebuild_controls_and_variable_specs(self):
        """Recomputes self.controls/self.variable_specs from self.plots -
        called after remove_plot()/replace_plot(), since either can orphan
        a control or VariableSpec only the removed/replaced plot
        referenced."""
        self.controls = []
        self.variable_specs = []
        for plot in self.plots:
            for spec in plot.controls:
                if isinstance(spec, ControlSpec) and not isinstance(spec, FacetSpec):
                    if spec not in self.controls:
                        self.controls.append(spec)
            for spec in plot.variable_specs:
                if spec not in self.variable_specs:
                    self.variable_specs.append(spec)

    def remove_plot(self, plot: Plot):
        """Removes `plot` from wherever it is in this Figure's panel grid.

        Args:
            plot (Plot): The plot to remove.

        Raises:
            ValueError: If `plot` is not part of this Figure.
        """
        for row in self.panels:
            for overlay in row:
                if overlay.remove(plot):
                    self.plots.remove(plot)
                    self._rebuild_controls_and_variable_specs()
                    self._sync_backend_figure()
                    return
        raise ValueError(f"{plot} is not part of this Figure.")

    def replace_plot(self, old: Plot, new: Plot):
        """Replaces `old` with `new` in the same panel position, keeping
        `old`'s color_index.

        Args:
            old (Plot): The plot currently in this Figure to replace.
            new (Plot): The plot to put in its place.

        Raises:
            ValueError: If `old` is not part of this Figure, or if `new`
                does not share `old`'s Overlay's embedding_dim.
        """
        for row in self.panels:
            for overlay in row:
                if old in overlay.plots:
                    overlay.replace(old, new)
                    self.plots[self.plots.index(old)] = new
                    new.theme = self.theme
                    new.color_index = old.color_index
                    self._rebuild_controls_and_variable_specs()
                    self._sync_backend_figure()
                    return
        raise ValueError(f"{old} is not part of this Figure.")

    @staticmethod
    def facet_specs(plot: Plot) -> dict[str, FacetSpec]:
        specs = [s for s in plot.controls if isinstance(s, FacetSpec)]
        by_orientation = {s.orientation: s for s in specs}
        assert len(by_orientation) == len(
            specs
        ), "At most one FacetSpec per orientation (row/col) allowed per plot."
        return by_orientation

    def _facet_extent(self) -> tuple[int, int]:
        """(facet_rows, facet_cols): the largest facet grid any single plot
        in this Figure declares. Every panel's block is sized to this,
        uniformly - see figure_plan.md §5 (panels and facets multiply)."""
        facet_rows, facet_cols = 1, 1
        for plot in self.plots:
            facets = self.facet_specs(plot)
            if "row" in facets:
                facet_rows = max(facet_rows, len(facets["row"].values))
            if "col" in facets:
                facet_cols = max(facet_cols, len(facets["col"].values))
        return facet_rows, facet_cols

    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) the whole Figure needs: panel_rows * facet_rows,
        panel_cols * facet_cols. A plot with no FacetSpec on an orientation
        draws into just the first row/col of its own block, not every one
        (matches Plot.create_artist's row=None/col=None meaning "cell 1"),
        and a panel with no FacetSpec at all still gets a full-size block -
        it just uses that block's first cell (see figure_plan.md §5)."""
        facet_rows, facet_cols = self._facet_extent()
        panel_rows = len(self.panels) if self.panels else 1
        panel_cols = max((len(row) for row in self.panels), default=1)
        return panel_rows * facet_rows, panel_cols * facet_cols

    def cell_titles(self, n_rows: int, n_cols: int) -> list[str]:
        """Row-major subplot title per grid cell (Plotly's make_subplots(
        subplot_titles=...) order): each panel's own plot title(s) (every
        distinct, non-None Plot.title in that cell's Overlay), with its
        facet's label appended when both are present. Only set on the
        cells the panel's plots actually draw into (its first cell alone,
        for a panel with no FacetSpec sharing a block with a faceted one -
        see figure_plan.md §5) - "" everywhere else."""
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

    def cell_dimensions(self, n_rows: int, n_cols: int) -> list[list[int]]:
        """Embedding dimension (2 or 3) per grid cell, inferred from
        whichever plot(s) draw into each cell - renderer-agnostic; translating
        this into a specific backend's subplot-type vocabulary (Plotly's
        "xy"/"scene", or whatever an eventual second renderer uses) is that
        renderer's job, not Figure's. Cells no plot reaches keep dimension 2,
        the harmless default."""
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
                            dims[row_off + row_idx][col_off + col_idx] = (
                                plot.embedding_dim
                            )
        return dims

    def cell_spans(
        self, n_rows: int, n_cols: int
    ) -> list[list[tuple[int, int] | None]]:
        """(rowspan, colspan) per grid cell for a backend's subplot-grid
        vocabulary (Plotly's make_subplots(specs=...) rowspan/colspan) -
        (1, 1) everywhere, except a panel that has no FacetSpec on an
        orientation while sharing a block sized by a sibling panel's
        FacetSpec on that same orientation (figure_plan.md §5): its cell
        spans the block's full extent on that orientation, rather than
        sitting narrow in the block's first cell, and every other cell it
        absorbs is None (a backend's "no subplot here, spanned by a
        neighbor" marker). Scoped to that one case - the unrelated
        padded-but-unspanned cells from Row/Column nesting (§9) are left
        alone, each still its own (1, 1) cell."""
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
                has_row_facet = any(
                    "row" in self.facet_specs(p) for p in overlay.plots
                )
                has_col_facet = any(
                    "col" in self.facet_specs(p) for p in overlay.plots
                )
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
        """The one TimeSpec in use across the whole Figure, or None. Unlike
        FacetSpec (one per orientation, per plot), animation frames are
        figure-wide - Plotly plays one shared frame sequence, so at most one
        distinct TimeSpec across every plot is supported."""
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

    def draw(self):
        # Pass 1: reset and train shared scales before anything is drawn -
        # otherwise the colorbar claim from a previous draw() lingers and the
        # second render loses its colorbar.
        for scale in self._scales_in_use():
            scale.reset()
        for plot in self.plots:
            values = plot.color_values()
            if values is not None:
                plot.color.scale.observe(values)

        # Pass 2: draw with the now-trained scales. Plotly (and presumably
        # any renderer with a comparable subplot mechanism) requires row/col
        # together or not at all - self.backend_figure is a subplot grid the
        # moment ANY plot facets on either orientation, or there is more
        # than one panel, so every plot's non-faceted orientation must
        # default to cell 1 of its own block, not None, once that's true.
        is_grid = self.grid_shape() != (1, 1)
        facet_rows, facet_cols = self._facet_extent()
        for panel_row, row in enumerate(self.panels):
            for panel_col, overlay in enumerate(row):
                block_offset = (panel_row * facet_rows, panel_col * facet_cols)
                for plot in overlay.plots:
                    self._draw_plot(plot, is_grid, block_offset)

        # Pass 3 (only with a TimeSpec present): materialize one frame per
        # animated state on top of the now-drawn figure. Frames/play-pause UI
        # are backend vocabulary (Plotly specifically wants them up front,
        # unlike the on-demand create()/update() cycle everything else uses),
        # so building them is the renderer's job, not Figure's - the same
        # split as Renderer.setup() owning subplot-grid construction.
        spec = self._animation_spec()
        if spec is not None:
            self.renderer.animate(self, self.backend_figure, spec)

        return self.backend_figure

    def _draw_plot(
        self, plot: Plot, is_grid: bool, block_offset: tuple[int, int] = (0, 0)
    ):
        """Draws one plot into every grid cell its FacetSpec(s) call for,
        offset by `block_offset` (this plot's panel's top-left corner) -
        exactly one cell, at block_offset, if it has none. The renderer
        sets `spec.state` per cell and calls plot.evaluate() again through
        create_artist()/update() - the same cycle a Dash slider callback
        triggers - so Plot itself needs no facet awareness."""
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

                    key = (row_idx if row_spec is not None else None,
                           col_idx if col_spec is not None else None)
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
                        artist.update(self.backend_figure, plot)
        finally:
            if row_spec is not None:
                row_spec.state = original_row_state
            if col_spec is not None:
                col_spec.state = original_col_state

    def _apply_variable_selectors(self):
        """Static-export counterpart to DashApplication's dropdown widget -
        only called by show()/save_html()/save_png()/save_svg(), not draw()
        itself, since a Dash app already handles VariableSpec server-side
        and would otherwise get a redundant, non-functional copy baked into
        its figure too."""
        for spec in self.variable_specs:
            self.renderer.apply_variable_selector(self, self.backend_figure, spec)

    def show(self):
        """Draws the figure and displays it (e.g. opening a browser tab)."""
        self.draw()
        self._apply_variable_selectors()
        self.renderer.show(self.backend_figure)

    def save_html(self, path):
        """Draws the figure and writes it to `path` as an interactive HTML file."""
        self.draw()
        self._apply_variable_selectors()
        self.renderer.save_html(self.backend_figure, path)

    def save_gif(self, path, fps=10):
        """Draws the figure and writes it to `path` as an animated GIF.

        Only meaningful with a TimeSpec control somewhere in the Figure -
        `draw()` must populate `backend_figure.frames`, which is then
        rasterized frame by frame and assembled into the GIF.
        """
        self.draw()
        self.renderer.save_gif(self.backend_figure, path, fps=fps)

    def save_png(self, path, **kwargs):
        """Draws the figure and writes its current state to `path` as a PNG."""
        self.draw()
        self.renderer.save_image(self.backend_figure, path, format="png", **kwargs)

    def save_svg(self, path, **kwargs):
        """Draws the figure and writes its current state to `path` as an SVG."""
        self.draw()
        self.renderer.save_image(self.backend_figure, path, format="svg", **kwargs)
