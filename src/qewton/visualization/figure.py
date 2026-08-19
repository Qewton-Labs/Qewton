from qewton.visualization.plots.base import Plot
from qewton.visualization.plots.spec import ControlSpec, FacetSpec, Scale, TimeSpec
from qewton.visualization.renderers.base import Artist, Renderer
from qewton.visualization.themes.base import Theme
from qewton.visualization.renderers import DEFAULT_RENDERER
from qewton.visualization.themes import DEFAULT_THEME


class Figure:
    def __init__(
        self,
        plots: Plot | list[Plot] | None = None,
        renderer: Renderer = DEFAULT_RENDERER,
        theme: Theme = DEFAULT_THEME,
        title=None,
    ):
        self.renderer = renderer
        self.theme = theme
        self.title = title
        self.plots = []
        self.controls = []

        if plots is not None:
            if isinstance(plots, list):
                for plot in plots:
                    assert isinstance(plot, Plot)
                    self.add_plot(plot)
            else:
                assert isinstance(plots, Plot)
                self.add_plot(plots)

        # One artist per (plot, cell) - the non-faceted case degenerates to a
        # single {(None, None): artist} entry per plot.
        self.artists: dict[Plot, dict[tuple, Artist]] = {}
        self.legend = None
        self.backend_figure = renderer.setup(self)

    def add_plot(self, plot: Plot):
        plot.theme = self.theme
        self.plots.append(plot)
        for spec in plot.controls:
            if isinstance(spec, ControlSpec) and not isinstance(spec, FacetSpec):
                if spec not in self.controls:
                    self.controls.append(spec)

    @staticmethod
    def facet_specs(plot: Plot) -> dict[str, FacetSpec]:
        specs = [s for s in plot.controls if isinstance(s, FacetSpec)]
        by_orientation = {s.orientation: s for s in specs}
        assert len(by_orientation) == len(
            specs
        ), "At most one FacetSpec per orientation (row/col) allowed per plot."
        return by_orientation

    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) the whole Figure needs - the largest facet grid
        any single plot declares. A plot with no FacetSpec on an orientation
        draws into just the first row/col of that orientation, not every one
        (matches Plot.create_artist's row=None/col=None meaning "cell 1")."""
        n_rows, n_cols = 1, 1
        for plot in self.plots:
            facets = self.facet_specs(plot)
            if "row" in facets:
                n_rows = max(n_rows, len(facets["row"].values))
            if "col" in facets:
                n_cols = max(n_cols, len(facets["col"].values))
        return n_rows, n_cols

    def cell_dimensions(self, n_rows: int, n_cols: int) -> list[list[int]]:
        """Embedding dimension (2 or 3) per grid cell, inferred from
        whichever plot(s) draw into each cell - renderer-agnostic; translating
        this into a specific backend's subplot-type vocabulary (Plotly's
        "xy"/"scene", or whatever an eventual second renderer uses) is that
        renderer's job, not Figure's. Cells no plot reaches keep dimension 2,
        the harmless default."""
        dims = [[2] * n_cols for _ in range(n_rows)]
        for plot in self.plots:
            facets = self.facet_specs(plot)
            row_indices = range(len(facets["row"].values)) if "row" in facets else [0]
            col_indices = range(len(facets["col"].values)) if "col" in facets else [0]
            for row_idx in row_indices:
                for col_idx in col_indices:
                    dims[row_idx][col_idx] = plot.embedding_dim
        return dims

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
        # moment ANY plot facets on either orientation, so every plot's
        # non-faceted orientation must default to cell 1, not None, once
        # that's true.
        is_grid = self.grid_shape() != (1, 1)
        for plot in self.plots:
            self._draw_plot(plot, is_grid)

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

    def _draw_plot(self, plot: Plot, is_grid: bool):
        """Draws one plot into every grid cell its FacetSpec(s) call for -
        exactly one cell, (None, None), if it has none. The renderer sets
        `spec.state` per cell and calls plot.evaluate() again through
        create_artist()/update() - the same cycle a Dash slider callback
        triggers - so Plot itself needs no facet awareness."""
        facets = self.facet_specs(plot)
        row_spec = facets.get("row")
        col_spec = facets.get("col")
        row_values = row_spec.values if row_spec is not None else [None]
        col_values = col_spec.values if col_spec is not None else [None]

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
                        row = row_idx + 1 if row_spec is not None else 1
                        col = col_idx + 1 if col_spec is not None else 1
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

    def show(self):
        self.draw()
        self.renderer.show(self.backend_figure)

    def save_html(self, path):
        self.draw()
        self.renderer.save_html(self.backend_figure, path)

    def save_gif(self, path, fps=10):
        """Only meaningful with a TimeSpec control somewhere in the Figure -
        draw() must have populated backend_figure.frames, which
        renderer.save_gif() then rasterizes and assembles."""
        self.draw()
        self.renderer.save_gif(self.backend_figure, path, fps=fps)
