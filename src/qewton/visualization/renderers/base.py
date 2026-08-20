class Renderer:
    """Base class for a backend that turns Plots into a concrete figure
    (e.g. Plotly). Owns everything specific to that backend's vocabulary:
    building the figure/subplot grid, applying a Theme, animation, and
    export - Plot and Figure themselves stay renderer-agnostic."""

    @staticmethod
    def setup(figure):
        """Builds and returns a new, empty backend figure for `figure`,
        already sized to its facet grid and themed."""
        raise NotImplementedError()

    @staticmethod
    def animate(figure, backend_figure, spec):
        """Materializes one frame per `spec.values` on top of an already-drawn
        backend_figure, plus whatever play/pause/scrub UI this backend uses.
        Only called when a TimeSpec is present. Backends without a
        frame-based animation model can leave this unimplemented."""
        raise NotImplementedError()

    @staticmethod
    def apply_variable_selector(figure, backend_figure, spec):
        """Adds interactive UI to an already-drawn static backend_figure
        letting it switch which candidate a VariableSpec currently selects,
        without needing a running Dash app. Only called for the static
        export/display path (Figure.show()/save_html()/save_png()/
        save_svg()) - Dash's own callback loop already handles VariableSpec
        server-side and has no use for this. Backends without a way to
        embed this in a static figure can leave it unimplemented."""
        raise NotImplementedError()

    @staticmethod
    def show(backend_figure):
        """Displays the figure (e.g. opening a browser tab)."""
        raise NotImplementedError()

    @staticmethod
    def save_html(backend_figure, path):
        """Writes the figure to `path` as an interactive HTML file."""
        raise NotImplementedError()

    @staticmethod
    def save_gif(backend_figure, path, fps=10):
        """Renders each animation frame (see `animate()`) to a static image
        and assembles them into a looping GIF. Requires `backend_figure` to
        already carry frames, i.e. `Figure.draw()` must have run with a
        TimeSpec present. An implementation may depend on an optional
        rasterization library (e.g. Plotly needs 'kaleido'), imported
        lazily rather than as a hard dependency of this package."""
        raise NotImplementedError()

    @staticmethod
    def save_image(backend_figure, path, format, **kwargs):
        """Renders the figure's current state to a single static image.
        Same optional-dependency convention as `save_gif()`."""
        raise NotImplementedError()


class Artist:
    """Base class for a renderer-specific object that draws one Plot's
    evaluated result into a backend figure and keeps it in sync across
    redraws."""

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Evaluates `plot` and adds its trace(s) to `backend_figure`,
        returning the new Artist instance. `row`/`col` place it in a facet
        grid cell, or are None for a non-faceted figure."""
        raise NotImplementedError

    def update(self, backend_figure, plot):
        """Re-evaluates `plot` in its current control state and refreshes
        this artist's existing trace(s) in place."""
        raise NotImplementedError

    def remove(self, backend_figure):
        """Removes this artist's trace(s) from `backend_figure`."""
        raise NotImplementedError
