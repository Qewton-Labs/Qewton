"""Backend-agnostic base classes for renderers and artists."""


class Renderer:
    """Backend that turns the plots of a Figure into a concrete figure.

    A renderer owns everything specific to one backend's vocabulary:
    building the figure and its subplot grid, applying a
    :class:`~qewton.visualization.themes.base.Theme`, animation, static
    interactive controls, and export.
    :class:`~qewton.visualization.plots.base.Plot` and
    :class:`~qewton.visualization.figure.Figure` stay renderer-agnostic.

    Notes:
        Subclasses must implement :meth:`setup`, :meth:`show` and
        :meth:`save_html`. They may implement :meth:`reconcile_y_axis_titles`
        (default: no-op), :meth:`animate`, :meth:`apply_selector`,
        :meth:`apply_slider`, :meth:`save_gif` and :meth:`save_image`
        (default: raise :class:`NotImplementedError`) when the backend supports
        the feature. A renderer also exposes one
        :class:`~qewton.visualization.renderers.base.Artist` subclass per plot
        family as a class attribute (e.g. ``renderer.HeatmapArtist``), which
        the plots look up in ``create_artist()``.
    """

    @staticmethod
    def setup(figure):
        """Build a new, empty backend figure for `figure`.

        Args:
            figure (Figure): The :class:`~qewton.visualization.figure.Figure`
                to build a backend figure for.

        Returns:
            object: An empty backend figure, sized to `figure`'s facet grid and
                themed.
        """
        raise NotImplementedError()

    @staticmethod
    def reconcile_y_axis_titles(figure, backend_figure, n_rows, n_cols):
        """Set y-axis titles that depend on Figure-level context.

        Called once after every plot has drawn its traces. Lets the backend
        override a per-artist y-axis title with
        :meth:`~qewton.visualization.figure.Figure.cell_y_titles`, which a
        single artist cannot compute on its own. The default is a no-op.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (object): The backend figure to modify in place.
            n_rows (int): Number of rows of the figure's grid.
            n_cols (int): Number of columns of the figure's grid.
        """

    @staticmethod
    def animate(figure, backend_figure, spec):
        """Add one animation frame per value of a TimeSpec.

        Called by :meth:`~qewton.visualization.figure.Figure.draw` only when
        a :class:`~qewton.visualization.plots.spec.TimeSpec` is present.
        Adds the frames plus the backend's play/pause/scrub UI on top of the
        already-drawn `backend_figure`.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (object): The drawn backend figure to modify in
                place.
            spec (TimeSpec): The time control whose ``values`` become frames.

        Returns:
            object: `backend_figure`.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_selector(figure, backend_figure, spec):
        """Add static UI that switches the candidate of a SelectorSpec.

        Bakes every candidate of `spec` into the already-drawn
        `backend_figure`, so it can switch between them without a running
        Dash app. Called only by
        :meth:`~qewton.visualization.figure.Figure.show` and
        :meth:`~qewton.visualization.figure.Figure.save_html`; a
        :class:`~qewton.visualization.applications.dash_app.DashApplication`
        handles selectors server-side instead.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (object): The drawn backend figure to modify in
                place.
            spec (SelectorSpec): The selector to bake.

        Returns:
            object: `backend_figure`.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_slider(figure, backend_figure, spec):
        """Add static UI that steps through every state of a SliderSpec.

        The slider counterpart to :meth:`apply_selector`, called on the
        same static display/export path.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (object): The drawn backend figure to modify in
                place.
            spec (SliderSpec): The slider to bake.

        Returns:
            object: `backend_figure`.
        """
        raise NotImplementedError()

    @staticmethod
    def show(backend_figure):
        """Display the figure (e.g. in a browser tab or notebook cell).

        Args:
            backend_figure (object): The drawn backend figure.
        """
        raise NotImplementedError()

    @staticmethod
    def save_html(backend_figure, path):
        """Write the figure to `path` as an interactive HTML file.

        Args:
            backend_figure (object): The drawn backend figure.
            path (str or path-like): Output file path.
        """
        raise NotImplementedError()

    @staticmethod
    def save_gif(backend_figure, path, fps=10, frame_weights=None):
        """Render the animation frames into a looping GIF.

        Frame ``i`` is shown for ``frame_weights[i] / fps`` seconds, or
        ``1 / fps`` seconds without weights (see
        :meth:`~qewton.visualization.plots.spec.TimeSpec.frame_weights`).
        `backend_figure` must already carry frames, i.e. it was drawn with a
        TimeSpec present (see :meth:`animate`). Implementations may import
        an optional rasterization library lazily.

        Args:
            backend_figure (object): The drawn backend figure, with animation
                frames.
            path (str or path-like): Output file path.
            fps (float, optional): Frames per second for equally weighted
                frames.
            frame_weights (list of float, optional): Relative display duration
                per frame.
        """
        raise NotImplementedError()

    @staticmethod
    def save_image(backend_figure, path, format, **kwargs):
        """Render the figure's current state to a single static image.

        Implementations may import an optional rasterization library
        lazily.

        Args:
            backend_figure (object): The drawn backend figure.
            path (str or path-like): Output file path.
            format (str): Image format, e.g. ``"png"`` or ``"svg"``.
            **kwargs: Passed on to the backend's image export.
        """
        raise NotImplementedError()


class Artist:
    """Renderer-specific object that draws one Plot into a backend figure.

    An artist draws the evaluated result of one
    :class:`~qewton.visualization.plots.base.Plot` (in one facet cell) and
    keeps its traces in sync across redraws.

    Notes:
        Subclasses must implement :meth:`create` and :meth:`update`.
        :meth:`update` must fully refresh every trace the artist owns from the
        plot's current control state; animation frames, static sliders and
        dropdowns are built by replaying it (see
        :meth:`~qewton.visualization.renderers.base.Renderer.animate`).
    """

    @classmethod
    def create(cls, backend_figure, plot, row=None, col=None):
        """Evaluate `plot` and add its traces to `backend_figure`.

        Args:
            backend_figure (object): The backend figure to draw into.
            plot (Plot): The plot to draw, in its current control state.
            row (int, optional): 1-based facet grid row, or None for a
                non-faceted figure.
            col (int, optional): 1-based facet grid column, or None for a
                non-faceted figure.

        Returns:
            Artist: The new artist, owning the added traces.
        """
        raise NotImplementedError

    def update(self, backend_figure, plot, row=None, col=None):
        """Re-evaluate `plot` and refresh this artist's traces in place.

        Args:
            backend_figure (object): The backend figure the artist drew into.
            plot (Plot): The plot to draw, in its current control state.
            row (int, optional): The same facet grid row passed to
                :meth:`create`, for artists that also refresh per-cell layout
                such as axis titles.
            col (int, optional): The same facet grid column passed to
                :meth:`create`, for artists that also refresh per-cell layout
                such as axis titles.
        """
        raise NotImplementedError
