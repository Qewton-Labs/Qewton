from qewton.visualization.themes import DEFAULT_THEME
from qewton.visualization.themes.base import Theme


class Renderer:
    """Draws one Figure in one plotting framework, in one theme.

    A renderer holds the backend figure and the artists drawing the
    figure's Results, one artist class per Result class and number of
    position components (:attr:`artists`); it draws what belongs to a cell
    or the whole figure itself: axes and colorbars. Self-contained outputs
    embed states of controls: the figure draws each state, the renderer
    takes a :meth:`snapshot` of what was drawn and writes the snapshots
    into the backend figure (:meth:`animate`, :meth:`apply_slider`,
    :meth:`apply_selector`).

    Args:
        theme: Colors, fonts and sizes of everything drawn.

    Attributes:
        artists (dict): ``{(Result class, dim): Artist class}`` (class
            attribute).
        embeddable (tuple of type): Kinds of control a self-contained
            output embeds (class attribute); the Figure leaves the others
            in their current state, with a warning.
        max_embedded (int or None): Most controls embedded per output
            (class attribute); None for no limit.
        theme (Theme): The theme.
        figure (Figure or None): The figure drawn; None before
            :meth:`setup`.
        backend_figure (object): The framework's figure; None before
            :meth:`setup`.

    Notes:
        Subclasses fill :attr:`artists` and implement every method that
        raises NotImplementedError.
    """

    artists: dict = {}
    embeddable: tuple[type, ...] = ()
    max_embedded: int | None = None

    def __init__(self, theme: Theme = DEFAULT_THEME) -> None:
        self.theme = theme
        self.backend_figure = None
        self.figure = None
        self._drawn: dict = {}

    def artist(self, result_class: type, dim: int) -> type["Artist"]:
        """The artist class drawing a Result class in `dim` dimensions.

        Args:
            result_class: A Result class.
            dim: Number of position components, 2 or 3.

        Returns:
            type: The Artist class.

        Raises:
            ValueError: If the renderer cannot draw `result_class` in `dim`
                dimensions.
        """
        key = (result_class, dim)
        if key not in self.artists:
            raise ValueError(
                f"{type(self).__name__} cannot draw a {result_class.__name__} in {dim}D."
            )
        return self.artists[key]

    def setup(self, figure) -> None:
        """Bind the renderer to `figure` and build an empty backend figure
        with its cell grid.

        Args:
            figure (Figure): The figure to draw.

        Raises:
            ValueError: If the renderer already draws another figure.
        """
        if self.figure is not None and self.figure is not figure:
            raise ValueError("A renderer draws one figure; create a renderer per figure.")
        self.figure = figure
        self._drawn = {}
        self.backend_figure = self.create_backend_figure(figure)

    def create_backend_figure(self, figure):
        """An empty backend figure with the figure's cell grid, title and
        theme.

        Args:
            figure (Figure): The figure.

        Returns:
            object: The backend figure.
        """
        raise NotImplementedError

    def draw(self, key, result, cell, visible: bool = True) -> None:
        """Draw `result` with the artist of `key`: create it on first use,
        update it afterwards.

        Args:
            key: Identifies one drawn copy in one cell across states.
            result (Result): The Result of the current state.
            cell (Cell): The cell drawn into.
            visible: Whether the artist is shown.
        """
        artist = self._drawn.get(key)
        if artist is None:
            artist_class = self.artist(type(result), cell.dim)
            artist = artist_class.create(self.backend_figure, result, cell, self.theme)
            self._drawn[key] = artist
        else:
            artist.update(self.backend_figure, result, cell, self.theme)
        artist.set_visible(self.backend_figure, visible)

    def hide(self, key) -> None:
        """Hide the artist of `key`.

        Args:
            key: A key passed to :meth:`draw` before.
        """
        self._drawn[key].set_visible(self.backend_figure, False)

    def artist_of(self, key) -> "Artist":
        """The artist of `key`.

        Args:
            key: A key passed to :meth:`draw` before.

        Returns:
            Artist: The artist.
        """
        return self._drawn[key]

    def draw_axes(self, cell) -> None:
        """Apply a cell's title and position axes: titles, scales (log,
        trained range, categorical), ticks and equal aspect.

        Args:
            cell (Cell): The cell.
        """
        raise NotImplementedError

    def draw_colorbar(self, scale, title, ticks, cell, slot: int, visible: bool = True) -> None:
        """Draw or update the one colorbar of a ColorScale, beside a cell's
        plot area; it is drawn in the scale's transformed values (log10 for
        a log scale).

        Args:
            scale (ColorScale): The trained color scale.
            title (str or None): Title of the colorbar.
            ticks (tuple or None): Tick positions in transformed values
                and their labels; None for automatic ticks.
            cell (Cell): The cell it stands beside
                (:meth:`Figure.colorbar_cell
                <qewton.visualization.figure.Figure.colorbar_cell>`).
            slot: Its place beside the cell, from the plot area
                (:meth:`Figure.colorbar_slot
                <qewton.visualization.figure.Figure.colorbar_slot>`).
            visible: Whether the colorbar is shown.
        """
        raise NotImplementedError

    def draw_legend(self, entries: dict, cell) -> None:
        """Draw one legend of the figure, beside a cell's plot area and
        its colorbars: one entry per label (as written by the renderer's
        text), for the artists drawing it.

        Args:
            entries: ``{label: keys}``, in order (see :meth:`Figure.legends
                <qewton.visualization.figure.Figure.legends>`).
            cell (Cell): The cell it stands beside.
        """
        raise NotImplementedError

    def draw_layout(self) -> None:
        """Size the backend figure and place its cells, titles, colorbars
        and embedded controls, once everything is drawn in every embedded
        state: every plot area gets its size from :meth:`Figure.plot_sizes
        <qewton.visualization.figure.Figure.plot_sizes>`, and nothing
        overlaps it.
        """
        raise NotImplementedError

    def clear_embeddings(self) -> None:
        """Remove every embedded control from the backend figure."""
        raise NotImplementedError

    def snapshot(self, keys, cells, scales) -> object:
        """What is drawn now by the artists of `keys`, the axes of `cells`
        and the colorbars of `scales`; one per embedded state.

        Args:
            keys (list): Artist keys.
            cells (list of Cell): Cells whose axes depend on the state.
            scales (list of ColorScale): Scales whose colorbars depend on
                the state.

        Returns:
            object: A snapshot for :meth:`animate`, :meth:`apply_slider`
                and :meth:`apply_selector`.
        """
        raise NotImplementedError

    def animate(self, time_axis, snapshots) -> None:
        """Embed the states of a TimeAxis as an animation.

        Args:
            time_axis (TimeAxis): The time axis.
            snapshots (list): One :meth:`snapshot` per state.
        """
        raise NotImplementedError

    def apply_slider(self, slider, snapshots) -> None:
        """Embed the states of a SliderAxis behind a slider.

        Args:
            slider (SliderAxis): The slider axis.
            snapshots (list): One :meth:`snapshot` per state.
        """
        raise NotImplementedError

    def apply_selector(self, selector, snapshots) -> None:
        """Embed the choices of a Selector behind a dropdown.

        Args:
            selector (Selector): The selector.
            snapshots (list): One :meth:`snapshot` per choice.
        """
        raise NotImplementedError

    def show(self) -> None:
        """Display the backend figure."""
        raise NotImplementedError

    def save_html(self, path) -> None:
        """Write the backend figure as an interactive HTML file.

        Args:
            path (str or path-like): Output file path.
        """
        raise NotImplementedError

    def save_gif(self, path, fps: float, frame_weights: list[float]) -> None:
        """Write the embedded animation as a looping GIF.

        Args:
            path (str or path-like): Output file path.
            fps: Frames per second of frames with weight 1.
            frame_weights: Relative display time per frame.
        """
        raise NotImplementedError

    def save_image(self, path, format, **kwargs) -> None:
        """Write the backend figure's current state as a static image.

        Args:
            path (str or path-like): Output file path.
            format (str): Image format, e.g. ``"png"`` or ``"svg"``.
            **kwargs: Passed on to the framework's image export.
        """
        raise NotImplementedError


class Artist:
    """Draws one Result into one cell of a backend figure and keeps it up
    to date. It sees only Results, cells and the theme, never plots.

    Notes:
        Subclasses implement :meth:`create`, :meth:`update` and
        :meth:`set_visible`. :meth:`update` must refresh everything that
        depends on the state, since outputs embed states by redrawing.
    """

    @classmethod
    def create(cls, backend_figure, result, cell, theme) -> "Artist":
        """Draw `result` into `cell`.

        Args:
            backend_figure (object): The backend figure.
            result (Result): The Result to draw.
            cell (Cell): The cell to draw into.
            theme (Theme): The renderer's theme.

        Returns:
            Artist: The new artist.
        """
        raise NotImplementedError

    def update(self, backend_figure, result, cell, theme) -> None:
        """Redraw with `result`, in place.

        Args:
            backend_figure (object): The backend figure.
            result (Result): The Result of the current state.
            cell (Cell): The cell drawn into.
            theme (Theme): The renderer's theme.
        """
        raise NotImplementedError

    def set_visible(self, backend_figure, visible: bool) -> None:
        """Show or hide what the artist draws.

        Args:
            backend_figure (object): The backend figure.
            visible: Whether it is shown.
        """
        raise NotImplementedError


def notebook_html(element: str) -> str:
    """HTML showing a figure's element in a notebook at its own size and
    resolution: in a box that scrolls horizontally where the figure is
    wider than the output, as notebooks scale images down to fit it.

    Args:
        element: The figure's HTML element.

    Returns:
        str: The HTML.
    """
    return (
        '<div class="qewton-figure" style="overflow-x: auto; max-width: 100%;">'
        "<style>.qewton-figure img { max-width: none !important; }</style>"
        f"{element}</div>"
    )
