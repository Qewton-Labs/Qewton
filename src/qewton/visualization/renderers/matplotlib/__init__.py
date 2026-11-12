"""Matplotlib renderer: :class:`MatplotlibRenderer` and its artists."""

import base64
import contextlib
import dataclasses
import io
import shutil

import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure as MatplotlibFigure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import MaxNLocator

from qewton.visualization.axes import SliderAxis, TimeAxis
from qewton.visualization.plots.result import (
    ArrowResult,
    BarResult,
    HeatmapResult,
    ImageResult,
    LineResult,
    MarkerResult,
    SurfaceResult,
)
from qewton.visualization.renderers.base import Renderer, notebook_html
from qewton.visualization.renderers import space
from qewton.visualization.renderers.matplotlib.primitives import (
    Arrow2DArtist,
    Arrow3DArtist,
    BarArtist,
    FlatMeshArtist,
    HeatmapArtist,
    ImageArtist,
    Line2DArtist,
    Line3DArtist,
    MarkerArtist,
    COLORBAR_LABEL,
    SurfaceArtist,
    axes_of,
    cell_label,
)
from qewton.visualization.renderers.matplotlib.style import (
    PIXELS_PER_INCH,
    color,
    colormap,
    norm,
    points,
    rc_params,
)
from qewton.visualization.renderers.matplotlib.text import as_mathtext
from qewton.visualization.scales import color_range
from qewton.visualization.themes import DEFAULT_THEME
from qewton.visualization.themes.base import Theme


#: Time each state of an embedded slider is shown when playing, in ms.
_SLIDER_INTERVAL = 500

#: Zoom of a 3D cell's box, small enough for its axis titles to fit.
_SCENE_ZOOM = 0.85


class MatplotlibRenderer(Renderer):
    """Draws a Figure with Matplotlib as a ``matplotlib.figure.Figure``,
    without pyplot: drawing changes no global state and opens no window.

    Text is written as mathtext, which renders the core's ``$...$`` math
    in Computer Modern without a LaTeX installation; the theme applies to
    every figure the renderer draws or writes, never globally.

    A self-contained output embeds one TimeAxis or SliderAxis as frames:
    the snapshot of each state is drawn into the existing artists, and the
    title names the state (HTML: Matplotlib's JavaScript player; GIF).
    Other controls keep their current state, with the Figure's warning.

    Args:
        theme: Colors, fonts and sizes of everything drawn.
        usetex: Whether LaTeX renders all text, instead of mathtext.
        dpi: Resolution of the figure and of raster images.

    Raises:
        RuntimeError: If `usetex` is set and no ``latex`` is installed.

    Notes:
        Not drawn: parallel coordinates and graph layouts.
    """

    artists = {
        (HeatmapResult, 2): HeatmapArtist,
        (SurfaceResult, 2): FlatMeshArtist,
        (SurfaceResult, 3): SurfaceArtist,
        (LineResult, 2): Line2DArtist,
        (LineResult, 3): Line3DArtist,
        (MarkerResult, 2): MarkerArtist,
        (MarkerResult, 3): MarkerArtist,
        (BarResult, 2): BarArtist,
        (ImageResult, 2): ImageArtist,
        (ArrowResult, 2): Arrow2DArtist,
        (ArrowResult, 3): Arrow3DArtist,
    }
    embeddable = (TimeAxis, SliderAxis)
    max_embedded = 1

    def __init__(
        self, theme: Theme = DEFAULT_THEME, usetex: bool = False, dpi: int = 100
    ) -> None:
        if usetex and shutil.which("latex") is None:
            raise RuntimeError(
                "usetex=True needs a LaTeX installation; no `latex` found."
            )
        super().__init__(theme)
        self.usetex = usetex
        self.dpi = dpi
        self._colorbars: dict = {}
        self._colorbar_room: dict = {}
        self._legends: list = []
        self._shown: dict = {}
        self._axes_settings: dict = {}
        self._colorbar_settings: dict = {}
        self._embedded = None

    def style(self) -> contextlib.AbstractContextManager:
        """The theme as a ``matplotlib.rc_context``. Fonts and math are
        resolved when the figure is rendered, so every method drawing or
        writing the figure runs in it.

        Returns:
            contextlib.AbstractContextManager: The context.
        """
        return matplotlib.rc_context(rc_params(self.theme, self.usetex))

    def text(self, text: str | None) -> str | None:
        """A text of the core as this renderer writes it into titles,
        axes, legends and colorbars.

        Args:
            text: A text, possibly with ``$...$`` parts.

        Returns:
            str or None: The text, as plain text if mathtext cannot render
                it (with a warning); unchanged with LaTeX.
        """
        return text if self.usetex else as_mathtext(text)

    def setup(self, figure) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.setup`."""
        self._colorbars, self._shown = {}, {}
        self._colorbar_room, self._legends = {}, []
        self._axes_settings, self._colorbar_settings = {}, {}
        self._embedded = None
        super().setup(figure)

    def draw(self, key, result, cell, visible: bool = True) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw`;
        the legend entry is written as :meth:`text`."""
        result = dataclasses.replace(result, label=self.text(result.label))
        with self.style():
            super().draw(key, result, cell, visible)
        self._shown[key] = (result, cell, visible)

    def hide(self, key) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.hide`."""
        super().hide(key)
        result, cell, _ = self._shown[key]
        self._shown[key] = (result, cell, False)

    # --- figure, axes, colorbars --------------------------------------------

    def create_backend_figure(self, figure) -> MatplotlibFigure:
        """See :meth:`Renderer.create_backend_figure
        <qewton.visualization.renderers.base.Renderer.create_backend_figure>`.

        Cells sharing a position axis with an earlier 2D cell share its
        Matplotlib axis, so they zoom together."""
        n_rows, n_cols = figure.grid_shape
        box = (self.theme.plot_width * n_cols, self.theme.plot_height * n_rows)
        with self.style():
            backend_figure = MatplotlibFigure(
                figsize=tuple(b / PIXELS_PER_INCH for b in box),
                dpi=self.dpi,
                layout=ConstrainedLayoutEngine(
                    w_pad=space.GAP / PIXELS_PER_INCH, h_pad=space.GAP / PIXELS_PER_INCH
                ),
            )
            grid = backend_figure.add_gridspec(n_rows, n_cols)
            for cell in figure.cells:
                shared = {}
                if cell.dim == 2:
                    for channel in ("x", "y"):
                        first = _first_cell_with(figure, cell.axes[channel], channel)
                        if first is not cell:
                            shared[f"share{channel}"] = axes_of(backend_figure, first)
                axes = backend_figure.add_subplot(
                    grid[cell.row - 1, cell.col - 1],
                    projection="3d" if cell.dim == 3 else None,
                    label=cell_label(cell),
                    **shared,
                )
                if cell.dim == 3:
                    _style_scene(axes, self.theme)
                elif self.theme.show_zero_line:
                    _draw_zero_lines(axes, self.theme)
            if figure.title is not None:
                backend_figure.suptitle(self.text(figure.title))
        return backend_figure

    def draw_axes(self, cell) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_axes`."""
        settings = dict(
            title=self.text(cell.title) or "",
            channels={
                channel: self._axis_settings(cell, channel, axis.scale)
                for channel, axis in cell.axes.items()
            },
            aspect=_aspect(cell),
        )
        self._axes_settings[(cell.row, cell.col)] = settings
        self._apply_axes(cell, settings)

    def _axis_settings(self, cell, channel: str, scale) -> dict:
        # Title, scale, range and ticks of one position axis.
        limits = cell.axis_range(channel)
        ticks = cell.axis_ticks(channel)
        if ticks is not None:
            positions, labels = ticks
            labels = [self.text(str(t)) for t in labels] if labels is not None else None
            ticks = (list(positions), labels)
        return dict(
            label=self.text(cell.axis_title(channel)) or "",
            scale="log" if scale.log else "linear",
            limits=limits,
            ticks=ticks,
            integer=scale.categorical,
        )

    def _apply_axes(self, cell, settings: dict) -> None:
        # Setting the scale resets the ticks to the scale's automatic ones.
        with self.style():
            axes = axes_of(self.backend_figure, cell)
            axes.set_title(settings["title"])
            for channel, axis in settings["channels"].items():
                getattr(axes, f"set_{channel}label")(axis["label"])
                getattr(axes, f"set_{channel}scale")(axis["scale"])
                if axis["limits"] is not None:
                    getattr(axes, f"set_{channel}lim")(*axis["limits"])
                if axis["ticks"] is not None:
                    getattr(axes, f"set_{channel}ticks")(*axis["ticks"])
                elif axis["integer"]:
                    locator = MaxNLocator(integer=True)
                    getattr(axes, f"{channel}axis").set_major_locator(locator)
            if cell.dim == 2:
                axes.set_aspect(settings["aspect"])
            else:
                axes.set_box_aspect(settings["aspect"], zoom=_SCENE_ZOOM)

    def draw_legend(self, entries: dict, cell) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_legend`.

        Drawn by :meth:`draw_layout`, which knows its room: right of the
        colorbars of its cell, with the symbol of the first artist of every
        label that has one (meshes draw none, see ``_has_legend_entry``)."""
        if not any(e.keys() == entries.keys() and c is cell for e, c in self._legends):
            self._legends.append((dict(entries), cell))

    def _create_legend(self, entries: dict, cell, offset: float, widths: list[float]) -> None:
        # A legend of the cell's Axes, `offset` pixels right of its plot area.
        handles, texts = [], []
        for label, keys in entries.items():
            text = self.text(label)
            symbols = (
                o for key in keys for o in self.artist_of(key).objects if o.get_label() == text
            )
            handle = next(symbols, None)
            if handle is not None:
                handles.append(handle)
                texts.append(text)
        axes = axes_of(self.backend_figure, cell)
        if axes.get_legend() is not None:
            axes.get_legend().remove()
        if handles:
            axes.legend(
                handles,
                texts,
                loc="upper left",
                bbox_to_anchor=(1 + offset / widths[cell.col - 1], 1.0),
                borderaxespad=0.0,
            )

    def draw_colorbar(self, scale, title, ticks, cell, slot: int, visible: bool = True) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_colorbar`.

        The colorbar is created by :meth:`draw_layout`, which knows its
        room; until then its settings are kept."""
        space.note_colorbar(
            self._colorbar_room,
            id(scale),
            cell,
            slot,
            title,
            ticks[1] if ticks is not None else space.tick_labels(*color_range(scale)),
        )
        if ticks is not None:
            positions, labels = ticks
            ticks = (list(positions), [self.text(t) for t in labels])
        settings = dict(
            norm=norm(scale),
            cmap=colormap(scale.colorscale(self.theme)),
            title=self.text(title) or "",
            ticks=ticks,
            visible=visible,
        )
        self._colorbar_settings[id(scale)] = settings
        if id(scale) in self._colorbars:
            self._apply_colorbar(id(scale), settings)

    def _apply_colorbar(self, scale_id: int, settings: dict) -> None:
        # A new norm resets the ticks to automatic ones.
        with self.style():
            colorbar = self._colorbars[scale_id]
            colorbar.update_normal(ScalarMappable(settings["norm"], settings["cmap"]))
            colorbar.set_label(settings["title"])
            colorbar.outline.set_visible(self.theme.axis_box)
            if settings["ticks"] is not None:
                positions, labels = settings["ticks"]
                colorbar.set_ticks(positions, labels=labels)
            colorbar.ax.set_visible(settings["visible"])

    def draw_layout(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_layout`.

        The figure is sized from the plot sizes and the estimated room
        beside them (:mod:`~qewton.visualization.renderers.space`); the
        grid's columns and rows get the plot sizes as ratios, and
        constrained layout places everything when rendering, so a plot
        area may miss its size by what the estimate misses. A colorbar is
        an Axes inside its cell's, right of its plot area, so it is as tall
        whatever its size. The figure title stands centered over the plot
        areas.
        """
        backend_figure, figure, theme = self.backend_figure, self.figure, self.theme
        legends = [(list(entries), cell) for entries, cell in self._legends]
        room = space.grid(figure, self._colorbar_room, theme, legends)
        columns = [l + w + r for l, w, r in zip(room["left"], room["widths"], room["right"])]
        rows = [t + h + b for t, h, b in zip(room["top"], room["heights"], room["bottom"])]
        # Constrained layout pads the figure's edges itself: no margin.
        title = space.title_height(figure, theme)
        width, height = sum(columns), title + sum(rows)
        backend_figure.set_size_inches(width / PIXELS_PER_INCH, height / PIXELS_PER_INCH)
        grid = axes_of(backend_figure, figure.cells[0]).get_subplotspec().get_gridspec()
        # A ratio of 0 is no grid; an empty column or row stays negligible.
        grid.set_width_ratios([max(w, 1e-3) for w in room["widths"]])
        grid.set_height_ratios([max(h, 1e-3) for h in room["heights"]])
        with self.style():
            for key, offset in room["colorbars"].items():
                if key not in self._colorbars:
                    self._create_colorbar(key, offset, room["widths"])
            for (entries, cell), offset in zip(self._legends, room["legends"]):
                self._create_legend(entries, cell, offset, room["widths"])
            if backend_figure.get_suptitle():
                start, end = room["left"][0], sum(columns) - room["right"][-1]
                backend_figure.suptitle(backend_figure.get_suptitle(), x=(start + end) / 2 / width)

    def _create_colorbar(self, key, offset: float, widths: list[float]) -> None:
        # Axes inside the cell's, `offset` pixels right of its plot area and
        # as tall; constrained layout keeps room for them as for labels.
        cell = self._colorbar_room[key]["cell"]
        width = widths[cell.col - 1]
        bar = axes_of(self.backend_figure, cell).inset_axes(
            [1 + offset / width, 0.0, space.COLORBAR_THICKNESS / width, 1.0],
            label=COLORBAR_LABEL,
        )
        settings = self._colorbar_settings[key]
        mappable = ScalarMappable(settings["norm"], settings["cmap"])
        self._colorbars[key] = self.backend_figure.colorbar(mappable, cax=bar)
        self._apply_colorbar(key, settings)

    # --- embedded controls ---------------------------------------------------

    def clear_embeddings(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.clear_embeddings`."""
        self._embedded = None

    def snapshot(self, keys, cells, scales) -> tuple[dict, dict, dict]:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.snapshot`.

        Returns:
            tuple: ``({key: (result, cell, visible)}, {(row, col): (cell,
                axes settings)}, {id(scale): colorbar settings})``, as
                drawn last.
        """
        return (
            {key: self._shown[key] for key in keys},
            {(c.row, c.col): (c, self._axes_settings[(c.row, c.col)]) for c in cells},
            {id(s): self._colorbar_settings[id(s)] for s in scales},
        )

    def animate(self, time_axis, snapshots) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.animate`.

        The states become frames shown for the time axis's duration each
        (frame weights apply to GIFs only), playing in a loop."""
        self._embedded = (time_axis, snapshots, time_axis.duration, "loop")

    def apply_slider(self, slider, snapshots) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.apply_slider`.

        The states become frames, chosen with the frame slider."""
        self._embedded = (slider, snapshots, _SLIDER_INTERVAL, "once")

    def _apply_frame(self, index: int) -> None:
        # Draw the snapshot of one state of the embedded control into the
        # existing artists, axes and colorbars; the title names the state.
        control, snapshots, _, _ = self._embedded
        artists, cells, colorbars = snapshots[index]
        with self.style():
            for key, (result, cell, visible) in artists.items():
                artist = self.artist_of(key)
                artist.update(self.backend_figure, result, cell, self.theme)
                artist.set_visible(self.backend_figure, visible)
        for cell, settings in cells.values():
            self._apply_axes(cell, settings)
        for scale_id, settings in colorbars.items():
            self._apply_colorbar(scale_id, settings)
        self._set_title(f"{control.name} = {control.label(control.states[index])}")

    @contextlib.contextmanager
    def _frames(self):
        # Yields the frame indices of the embedded control; restores its
        # current state and the title afterwards.
        control, snapshots, _, _ = self._embedded
        try:
            yield range(len(snapshots))
        finally:
            self._apply_frame(control.states.index(control.state))
            self._set_title(None)

    def _set_title(self, state: str | None) -> None:
        # The figure's title, followed by the state of an embedded control.
        parts = [self.text(self.figure.title), self.text(state)]
        title = ", ".join(p for p in parts if p)
        with self.style():
            self.backend_figure.suptitle(title)

    def _animation(self) -> FuncAnimation:
        _, snapshots, interval, _ = self._embedded
        return FuncAnimation(
            self.backend_figure,
            self._apply_frame,
            frames=len(snapshots),
            interval=interval,
        )

    # --- outputs -------------------------------------------------------------

    def show(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.show`.

        A notebook gets the figure as :meth:`html` at its own size,
        scrolling where it is wider than the output
        (:func:`~qewton.visualization.renderers.base.notebook_html`);
        elsewhere it opens in a window of Matplotlib's backend, animated if
        a control is embedded."""
        if _in_notebook():
            from IPython.display import HTML, display

            display(HTML(notebook_html(self.html(full_html=False))))
            return
        from matplotlib.backends import backend_registry

        module = backend_registry.load_backend_module(matplotlib.get_backend())
        manager = module.FigureCanvas.new_manager(self.backend_figure, id(self))
        with self.style():
            animation = self._animation() if self._embedded is not None else None
            manager.show()
            module.FigureManager.start_main_loop()
        del animation

    def html(self, full_html: bool = True) -> str:
        """The figure as HTML: Matplotlib's JavaScript player of the frames
        if a control is embedded, else an inline SVG image.

        Args:
            full_html: A whole page, else an element to embed.

        Returns:
            str: The HTML.
        """
        if self._embedded is not None:
            mode = self._embedded[3]
            with self._frames(), self.style():
                element = self._animation().to_jshtml(default_mode=mode)
        else:
            buffer = io.BytesIO()
            self.save_image(buffer, "svg")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            element = f'<img src="data:image/svg+xml;base64,{encoded}"/>'
        if not full_html:
            return element
        return (
            '<!DOCTYPE html>\n<html><head><meta charset="utf-8"></head>'
            f"<body>{element}</body></html>\n"
        )

    def save_html(self, path) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.save_html`;
        the page needs no scripts."""
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.html())

    def save_image(self, path, format, **kwargs) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.save_image`;
        e.g. ``"png"``, ``"svg"`` or ``"pdf"``, with ``savefig`` arguments."""
        with self.style():
            self.backend_figure.savefig(path, format=format, **kwargs)

    def save_gif(self, path, fps: float = 10, frame_weights=None) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.save_gif`.

        Raises:
            ValueError: If no control is embedded.
        """
        from PIL import Image

        if self._embedded is None:
            raise ValueError("The figure has no embedded control to write as frames.")
        images = []
        with self._frames() as indices:
            for index in indices:
                self._apply_frame(index)
                buffer = io.BytesIO()
                self.save_image(buffer, "png")
                images.append(Image.open(buffer))
        weights = frame_weights or [1.0] * len(images)
        durations = [int(1000 / fps * w) for w in weights]
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=durations, loop=0
        )


def _first_cell_with(figure, axis, channel: str):
    # The first cell of the figure drawing `axis` as `channel`.
    return next(c for c in figure.cells if c.axes.get(channel) is axis)


def _style_scene(axes, theme) -> None:
    # The panes of a 3D cell in the background color, gridded as the theme
    # says; 2D cells take both from the rcParams.
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color(color(theme.background_color))
    axes.grid(theme.show_grid)


def _draw_zero_lines(axes, theme) -> None:
    # Lines at 0 below the data, in the grid color; outside the drawn
    # range they are not seen.
    for draw in (axes.axhline, axes.axvline):
        draw(0.0, color=color(theme.grid_color), linewidth=points(1.0), zorder=0.5)


def _aspect(cell):
    # Equal units in a 2D cell of a fixed aspect, else free; the box of a
    # 3D cell (None: automatic).
    if cell.dim == 2:
        return "equal" if cell.aspect is not None else "auto"
    return cell.box_aspect


def _in_notebook() -> bool:
    # Whether code runs in a Jupyter kernel (notebook, JupyterLab, VS Code).
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config
