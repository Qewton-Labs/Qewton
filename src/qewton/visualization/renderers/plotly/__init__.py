"""Plotly renderer: :class:`PlotlyRenderer` and its artists."""

import dataclasses
import io

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qewton.visualization.axes import Selector, SliderAxis, TimeAxis
from qewton.visualization.plots.result import (
    ArrowResult,
    BarResult,
    GraphLayoutResult,
    HeatmapResult,
    ImageResult,
    LineResult,
    MarkerResult,
    ParallelCoordinatesResult,
    SurfaceResult,
)
from qewton.visualization.renderers.base import Renderer, notebook_html
from qewton.visualization.renderers.plotly import camera
from qewton.visualization.renderers.plotly.graph import NodeLinkArtist
from qewton.visualization.renderers.plotly.text import as_markup, as_math
from qewton.visualization.themes import DEFAULT_THEME
from qewton.visualization.themes.base import Theme
from qewton.visualization.renderers.plotly.primitives import (
    Arrow2DArtist,
    Arrow3DArtist,
    BarArtist,
    FlatMeshArtist,
    HeatmapArtist,
    ImageArtist,
    ParallelCoordinatesArtist,
    Line2DArtist,
    Line3DArtist,
    Marker2DArtist,
    Marker3DArtist,
    SurfaceArtist,
)
from qewton.visualization.renderers import space
from qewton.visualization.renderers.plotly import embedding, layout
from qewton.visualization.scales import color_range

#: Distance of a scene's camera on each axis: far enough that the box and
#: its tick labels fit the scene's domain (Plotly's own 1.25 clips them).
_SCENE_EYE = 1.6

#: make_subplots() subplot type per number of position components.
_SUBPLOT_TYPES = {0: "domain", 2: "xy", 3: "scene"}

#: Per number of position components: a trace drawing nothing but the
#: colorbar of its one value in such a cell, and the property holding its
#: colors. A cell laying out its own axes takes only domain traces, hence
#: parallel coordinates with one hidden axis.
_CARRIERS = {
    0: (lambda value: go.Parcoords(dimensions=[dict(values=value, visible=False)]), "line"),
    2: (
        lambda value: go.Scatter(
            x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip"
        ),
        "marker",
    ),
    3: (
        lambda value: go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", showlegend=False, hoverinfo="skip"
        ),
        "marker",
    ),
}


class PlotlyRenderer(Renderer):
    """Draws a Figure as a Plotly ``go.Figure``.

    Every cell is a subplot of ``make_subplots``: ``"xy"`` in 2D,
    ``"scene"`` in 3D, ``"domain"`` without position axes. Embedded controls become frames (TimeAxis),
    ``method="update"`` slider steps (SliderAxis) and dropdown buttons
    (Selector); each state sets only the trace properties and axes that
    differ between the states.

    Names of Variables (``$...$`` in the core) are rendered by MathJax:
    every text containing one is written as one math text, and
    :meth:`show` and :meth:`save_html` load MathJax. Slider labels,
    dropdown buttons and the axis titles of 3D scenes, where Plotly renders
    no math, get Plotly's own text markup (italics, sub- and superscripts)
    instead, as does every text without MathJax.

    Args:
        theme: As for :class:`~qewton.visualization.renderers.base.Renderer`.
        mathjax: Whether math is rendered by MathJax (loaded from a CDN);
            False writes every text in Plotly's markup and shows figures
            the usual Plotly way, e.g. offline.

    Attributes:
        mathjax (bool): Whether math is rendered by MathJax.
    """

    artists = {
        (HeatmapResult, 2): HeatmapArtist,
        (SurfaceResult, 2): FlatMeshArtist,
        (SurfaceResult, 3): SurfaceArtist,
        (LineResult, 2): Line2DArtist,
        (LineResult, 3): Line3DArtist,
        (MarkerResult, 2): Marker2DArtist,
        (MarkerResult, 3): Marker3DArtist,
        (BarResult, 2): BarArtist,
        (ImageResult, 2): ImageArtist,
        (ArrowResult, 2): Arrow2DArtist,
        (ArrowResult, 3): Arrow3DArtist,
        (ParallelCoordinatesResult, 0): ParallelCoordinatesArtist,
        (GraphLayoutResult, 2): NodeLinkArtist,
    }

    embeddable = (TimeAxis, SliderAxis, Selector)

    def __init__(self, theme: Theme = DEFAULT_THEME, mathjax: bool = True) -> None:
        super().__init__(theme)
        self.mathjax = mathjax
        self._colorbars: dict = {}
        self._legends: list = []
        self._cells: dict = {}

    def setup(self, figure) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.setup`."""
        self._colorbars, self._legends, self._cells = {}, [], {}
        super().setup(figure)

    def text(self, text: str | None) -> str | None:
        """A text of the core as this renderer writes it into titles,
        axes, legends and colorbars.

        Args:
            text: A text, possibly with ``$...$`` parts.

        Returns:
            str or None: One math text with MathJax, else Plotly markup.
        """
        return as_math(text) if self.mathjax else as_markup(text)

    def draw(self, key, result, cell, visible: bool = True) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw`;
        the legend entry is written as :meth:`text`, shown once the figure
        draws its legend (:meth:`draw_legend`)."""
        super().draw(key, dataclasses.replace(result, label=self.text(result.label)), cell, visible)
        self._cells[key] = cell

    def _traces_of(self, cell) -> list:
        # The traces of the artists drawn into a cell.
        data = self.backend_figure.data
        return [
            data[i] for key, c in self._cells.items() if c is cell
            for i in self.artist_of(key).trace_indices
        ]

    def draw_legend(self, entries: dict, cell) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_legend`.

        One Plotly legend (``legend``, ``legend2``, ...) per legend of the
        figure: the traces of a label form a ``legendgroup``, so a click
        toggles every copy, and the first trace of its first artist shows
        the entry. Artists draw no entry of their own and updates keep
        these, so they are set once. Placed by :meth:`draw_layout`."""
        texts = [self.text(label) for label in entries]
        names = [n for n, t, c in self._legends if t == texts and c is cell]
        name = names[0] if names else _legend_name(len(self._legends))
        if not names:
            self._legends.append((name, texts, cell))
            self.backend_figure.update_layout({name: _legend_style(self.theme)})
        data = self.backend_figure.data
        for text, keys in zip(texts, entries.values()):
            for k, key in enumerate(keys):
                indices = self.artist_of(key).trace_indices
                for index in indices:
                    data[index].update(
                        legend=name,
                        legendgroup=f"{name}-{text}",
                        showlegend=k == 0 and index == indices[0],
                    )

    # --- figure, axes, colorbars --------------------------------------------

    def create_backend_figure(self, figure) -> go.Figure:
        """See :meth:`Renderer.create_backend_figure
        <qewton.visualization.renderers.base.Renderer.create_backend_figure>`."""
        n_rows, n_cols = figure.grid_shape
        specs = [
            [
                {"type": _SUBPLOT_TYPES[cell.dim] if cell else "xy"}
                for cell in (figure.cell_at(row, col) for col in range(1, n_cols + 1))
            ]
            for row in range(1, n_rows + 1)
        ]
        backend_figure = make_subplots(rows=n_rows, cols=n_cols, specs=specs)
        backend_figure.update_layout(uirevision=True, title=self.text(figure.title))
        for cell in figure.cells:
            if cell.title:
                # Placed by draw_layout().
                backend_figure.add_annotation(
                    name=_title_name(cell),
                    text=self.text(cell.title),
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                )
        _apply_theme(backend_figure, self.theme)
        # Once: an embedded state or a redraw leaves the viewer's camera.
        backend_figure.update_scenes(camera_eye=dict(x=_SCENE_EYE, y=_SCENE_EYE, z=_SCENE_EYE))
        return backend_figure

    def draw_layout(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_layout`.

        The figure is sized in pixels and has fixed margins of 0; every
        cell, cell title, colorbar and embedded control is placed in paper
        coordinates (see :mod:`~qewton.visualization.renderers.plotly.layout`):
        sliders below the cells, an animation's buttons left of its slider,
        dropdowns right of the cells. Equal units never widen a range: the
        plot area has the aspect of its ranges.
        """
        backend_figure = self.backend_figure
        sliders = list(backend_figure.layout.sliders)
        menus = list(backend_figure.layout.updatemenus)
        dropdowns = [m for m in menus if m.type != "buttons"]
        placed = layout.place(
            self.figure,
            self._colorbars,
            self.theme,
            legends=[(texts, cell) for _, texts, cell in self._legends],
            sliders=[s.name.startswith("time-") for s in sliders],
            dropdowns=[[b.label for b in m.buttons] for m in dropdowns],
        )
        width, height = placed["size"]
        title_x, title_y = placed["title"]
        backend_figure.update_layout(
            width=round(width),
            height=round(height),
            # Plotly would widen margins for what it draws outside the plot
            # areas, shrinking the paper under the places computed here.
            margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
            title=dict(x=title_x, y=title_y, xanchor="center", yanchor="middle"),
        )
        for cell in self.figure.cells:
            x0, x1, y0, y1 = placed["cells"][(cell.row, cell.col)]
            if cell.dim == 3:
                backend_figure.update_scenes(
                    domain=dict(x=[x0, x1], y=[y0, y1]), row=cell.row, col=cell.col
                )
            elif cell.dim == 2:
                backend_figure.update_xaxes(domain=[x0, x1], row=cell.row, col=cell.col)
                backend_figure.update_yaxes(domain=[y0, y1], row=cell.row, col=cell.col)
            else:  # its traces lay out their own axes
                for trace in self._traces_of(cell):
                    trace.domain = dict(x=[x0, x1], y=[y0, y1])
            if cell.title:
                x, y = placed["titles"][(cell.row, cell.col)]
                backend_figure.update_annotations(dict(x=x, y=y), selector=dict(name=_title_name(cell)))
        for key, (x, y, length) in placed["colorbars"].items():
            place = dict(x=x, xanchor="left", y=y, yanchor="middle", len=length, lenmode="fraction")
            _, colored = _CARRIERS[self._colorbars[key]["cell"].dim]
            backend_figure.update_traces({colored: dict(colorbar=place)}, selector=dict(uid=key))
        for (name, _, _), (x, y) in zip(self._legends, placed["legends"]):
            backend_figure.update_layout({name: dict(x=x, xanchor="left", y=y, yanchor="top")})
        for slider, (x, y, length) in zip(sliders, placed["sliders"]):
            slider.update(x=x, xanchor="left", y=y, yanchor="top", len=length)
        for menu in menus:
            if menu.type == "buttons":
                x, y = placed["buttons"]
                menu.update(x=x, xanchor="left", y=y, yanchor="top", direction="left")
        for menu, (x, y) in zip(dropdowns, placed["dropdowns"]):
            menu.update(x=x, xanchor="left", y=y, yanchor="top")
        backend_figure.update_layout(sliders=sliders, updatemenus=menus)

    def draw_axes(self, cell) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_axes`."""
        if not cell.axes:
            return  # e.g. parallel coordinates, which lay out their own axes
        backend_figure = self.backend_figure
        # Scenes render no MathJax: their titles are written in markup.
        text = as_markup if cell.dim == 3 else self.text
        axes = {channel: _axis(cell, channel, axis, text) for channel, axis in cell.axes.items()}
        if cell.dim == 3:
            backend_figure.update_scenes(
                row=cell.row,
                col=cell.col,
                **{f"{c}axis": a for c, a in axes.items()},
                **_scene_aspect(cell),
            )
            return
        for channel in ("x", "y"):
            first = self._first_cell_with(cell.axes[channel], channel)
            if first is not cell:
                # Zooming one cell zooms every cell sharing the axis.
                axes[channel]["matches"] = _axis_reference(backend_figure, first, channel)
        backend_figure.update_xaxes(row=cell.row, col=cell.col, **axes["x"])
        backend_figure.update_yaxes(row=cell.row, col=cell.col, **axes["y"])
        # Equal units: draw_layout() sizes the plot area to the aspect of
        # the ranges. A cell of its own also keeps them on zooming
        # (scaleanchor, shrinking the plot area rather than widening a
        # range); Plotly draws the ticks of matched axes with a scaleanchor
        # displaced, so cells sharing an axis rely on their sizes.
        shared = any(
            other is not cell and other.axes.get(c) is cell.axes[c]
            for other in self.figure.cells
            for c in ("x", "y")
        )
        if cell.aspect is not None and not shared:
            backend_figure.update_xaxes(row=cell.row, col=cell.col, constrain="domain")
            backend_figure.update_yaxes(
                row=cell.row,
                col=cell.col,
                scaleanchor=_axis_reference(backend_figure, cell, "x"),
                scaleratio=1,
                constrain="domain",
            )

    def _first_cell_with(self, axis, channel: str):
        # The first cell of the figure drawing `axis` as `channel`.
        return next(c for c in self.figure.cells if c.axes.get(channel) is axis)

    def linked_scenes(self) -> list[list[str]]:
        """The 3D cells sharing their position axes, whose cameras move
        together: groups of scene names, e.g. ``[["scene", "scene2"]]``.

        Returns:
            list of list of str: Every group of at least two scenes.
        """
        groups: dict[int, list[str]] = {}
        for cell in self.figure.cells:
            if cell.dim == 3:
                scene = self.backend_figure.get_subplot(cell.row, cell.col).plotly_name
                groups.setdefault(id(self._first_cell_with(cell.axes["x"], "x")), []).append(scene)
        return [names for names in groups.values() if len(names) > 1]

    def draw_colorbar(self, scale, title, ticks, cell, slot: int, visible: bool = True) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_colorbar`.

        The colorbar is carried by a one-value trace drawing nothing else
        (see ``_CARRIERS``), identified by the scale's ``uid``.
        """
        backend_figure = self.backend_figure
        # What it shows in any state drawn, for the room draw_layout()
        # keeps for it.
        space.note_colorbar(
            self._colorbars,
            _colorbar_uid(scale),
            cell,
            slot,
            title,
            ticks[1] if ticks is not None else space.tick_labels(*color_range(scale)),
        )
        lo, hi = color_range(scale)
        colorbar = _colorbar_style(self.theme, self.text(title))
        if ticks is not None:
            positions, labels = ticks
            colorbar.update(
                tickvals=list(positions), ticktext=[self.text(t) for t in labels]
            )
        else:
            colorbar.update(tickvals=None, ticktext=None)
        create, colored = _CARRIERS[cell.dim]
        colors = {
            colored: dict(
                color=[lo],
                colorscale=scale.colorscale(self.theme),
                cmin=lo,
                cmax=hi,
                showscale=True,
                colorbar=colorbar,
            )
        }
        index = self._colorbar_index(scale)
        if index is not None:
            backend_figure.data[index].update(colors, visible=visible)
            return
        carrier = create([lo]).update(colors, uid=_colorbar_uid(scale), visible=visible)
        backend_figure.add_trace(carrier, row=cell.row, col=cell.col)

    def colorbar(self, scale):
        """The colorbar drawn for `scale`.

        Args:
            scale (ColorScale): A color scale of the figure.

        Returns:
            The Plotly colorbar, e.g. with ``x``, ``y``, ``len``,
            ``tickvals`` and ``ticktext``.
        """
        _, colored = _CARRIERS[self._colorbars[_colorbar_uid(scale)]["cell"].dim]
        return self.backend_figure.data[self._colorbar_index(scale)][colored].colorbar

    def _colorbar_index(self, scale) -> int | None:
        uid = _colorbar_uid(scale)
        return next(
            (i for i, t in enumerate(self.backend_figure.data) if t.uid == uid), None
        )

    # --- embedded controls ---------------------------------------------------

    def clear_embeddings(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.clear_embeddings`."""
        self.backend_figure.frames = ()
        self.backend_figure.layout.sliders = ()
        self.backend_figure.layout.updatemenus = ()

    def snapshot(self, keys, cells, scales) -> tuple[dict, dict]:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.snapshot`.

        Returns:
            tuple: ``({trace index: flat properties}, flat axis layout)``,
                with nested properties as dotted paths.
        """
        data = self.backend_figure.data
        indices = [i for key in keys for i in self.artist_of(key).trace_indices]
        indices += [self._colorbar_index(scale) for scale in scales]
        traces = {i: embedding.flatten(data[i].to_plotly_json()) for i in indices}
        layout = self.backend_figure.layout.to_plotly_json()
        axes = {}
        for cell in cells:
            for name in _axis_names(self.backend_figure, cell):
                axes.update(embedding.flatten(layout.get(name, {}), prefix=name))
        return traces, axes

    def animate(self, time_axis, snapshots) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.animate`.

        Frames update only the trace properties and axes that differ
        between the states. With non-uniform frame weights each frame gets
        its own duration and playback always starts at the first frame.
        """
        backend_figure = self.backend_figure
        states = time_axis.states
        indices, keys, layout_keys = embedding.changes(snapshots)
        types = {i: backend_figure.data[i].type for i in indices}
        backend_figure.frames = [
            go.Frame(
                data=[
                    dict(embedding.nest({k: traces[i].get(k) for k in keys[i]}), type=types[i])
                    for i in indices
                ],
                traces=indices,
                layout=embedding.nest({k: axes.get(k) for k in layout_keys}),
                name=str(state),
            )
            for state, (traces, axes) in zip(states, snapshots)
        ]
        weights = time_axis.frame_weights()
        uniform = bool(np.allclose(weights, weights[0]))
        play_frame = (
            dict(duration=time_axis.duration, redraw=True)
            if uniform
            else [dict(duration=time_axis.duration * w, redraw=True) for w in weights]
        )
        # No transitions: Plotly would interpolate the coordinates of traces
        # between frames, e.g. move triangles from one value bin's trace to
        # another's.
        still = dict(duration=0)
        play = dict(frame=play_frame, fromcurrent=uniform, transition=still)
        pause = dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=still)
        step = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=still)
        backend_figure.update_layout(
            updatemenus=list(backend_figure.layout.updatemenus)
            + [
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play", method="animate", args=[None, play]),
                        dict(label="Pause", method="animate", args=[[None], pause]),
                    ],
                )
            ],
            sliders=list(backend_figure.layout.sliders)
            + [
                dict(
                    name=f"time-{id(time_axis)}",
                    active=time_axis.state,
                    transition=still,
                    currentvalue=dict(prefix=as_markup(f"{time_axis.name} = ")),
                    steps=[
                        dict(
                            method="animate",
                            args=[[str(s)], step],
                            label=as_markup(time_axis.label(s)),
                        )
                        for s in states
                    ],
                )
            ],
        )

    def apply_slider(self, slider, snapshots) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.apply_slider`.

        Each step updates only the trace properties and axes that differ
        between the states.
        """
        steps = [
            dict(label=as_markup(slider.label(state)), method="update", args=args)
            for state, args in zip(slider.states, embedding.update_args(snapshots))
        ]
        existing = list(self.backend_figure.layout.sliders)
        self.backend_figure.update_layout(
            sliders=existing
            + [
                dict(
                    name=f"slider-{id(slider)}",
                    active=slider.state,
                    steps=steps,
                    currentvalue=dict(prefix=as_markup(f"{slider.name} = ")),
                )
            ]
        )

    def apply_selector(self, selector, snapshots) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.apply_selector`.

        Each choice updates only the trace properties and axes that differ
        between the choices: with every candidate drawn, their visibility.
        """
        buttons = [
            dict(label=as_markup(selector.label(state)), method="update", args=args)
            for state, args in zip(selector.states, embedding.update_args(snapshots))
        ]
        existing = list(self.backend_figure.layout.updatemenus)
        self.backend_figure.update_layout(
            updatemenus=existing
            + [
                dict(
                    name=f"selector-{id(selector)}",
                    buttons=buttons,
                    direction="down",
                    active=selector.state,
                    showactive=True,
                )
            ]
        )

    # --- outputs -------------------------------------------------------------

    def show(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.show`.

        A notebook gets the figure as :meth:`html` (which loads plotly.js
        and, with MathJax, MathJax from a CDN, as notebook frontends load no
        MathJax for Plotly's own output) at its own size, scrolling where it
        is wider than the output (:func:`~qewton.visualization.renderers.base.notebook_html`);
        elsewhere Plotly's renderer shows it. An animation starts with its
        play button, not on load. Linked 3D cells move their cameras
        together.
        """
        if _in_notebook():
            from IPython.display import HTML, display

            display(HTML(notebook_html(self.html(full_html=False))))
            return
        groups = self.linked_scenes()
        script = dict(post_script=camera.page_script(groups)) if groups else {}
        mathjax = dict(include_mathjax="cdn") if self.mathjax else {}
        self.backend_figure.show(auto_play=False, **mathjax, **script)

    def html(self, full_html: bool = True) -> str:
        """The figure as HTML that loads plotly.js and, with MathJax,
        MathJax from a CDN. An animation starts with its play button, not on
        load: playing on load would use Plotly's default transitions, which
        interpolate trace coordinates between frames. Linked 3D cells move
        their cameras together.

        Args:
            full_html: A whole page, else an element to embed.

        Returns:
            str: The HTML.
        """
        groups = self.linked_scenes()
        return self.backend_figure.to_html(
            full_html=full_html,
            include_plotlyjs="cdn",
            include_mathjax="cdn" if self.mathjax else False,
            auto_play=False,
            post_script=camera.page_script(groups) if groups else None,
        )

    def save_html(self, path) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.save_html`;
        the page loads plotly.js and MathJax from a CDN."""
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.html())

    def save_image(self, path, format, **kwargs) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.save_image`.

        Raises:
            ImportError: If the optional ``kaleido`` package is missing.
        """
        try:
            import kaleido  # noqa: F401 - presence check; write_image() uses it
        except ImportError as e:
            raise ImportError(
                "Saving images requires the optional 'kaleido' package: "
                "pip install kaleido"
            ) from e
        self.backend_figure.write_image(path, format=format, **kwargs)

    def frame_figures(self) -> list[go.Figure]:
        """One static figure per animation frame: the backend figure with
        the frame's changes applied, without frames, sliders and buttons.

        Returns:
            list of plotly.graph_objects.Figure: The frames' figures.

        Raises:
            ValueError: If the figure has no animation frames.
        """
        base = self.backend_figure
        if not base.frames:
            raise ValueError("The figure has no animation frames; it needs a TimeAxis.")
        figures = []
        for frame in base.frames:
            figure = go.Figure(base)
            figure.frames = ()
            figure.layout.sliders, figure.layout.updatemenus = (), ()
            for index, data in zip(frame.traces, frame.data):
                figure.data[index].update(data.to_plotly_json())
            if frame.layout is not None:
                figure.update_layout(frame.layout.to_plotly_json())
            figures.append(figure)
        return figures

    def save_gif(self, path, fps: float = 10, frame_weights=None) -> None:
        """Rasterize every animation frame and write a looping GIF.

        Args:
            path (str or path-like): Output file path.
            fps: Frames per second of equally weighted frames.
            frame_weights (list of float, optional): Relative display
                duration per frame.

        Raises:
            ImportError: If ``kaleido`` or ``Pillow`` is missing.
            ValueError: If the figure has no animation frames.
        """
        try:
            import kaleido  # noqa: F401 - presence check
            from PIL import Image
        except ImportError as e:
            raise ImportError("save_gif() requires 'kaleido' and 'Pillow'.") from e
        images = [
            Image.open(io.BytesIO(figure.to_image(format="png")))
            for figure in self.frame_figures()
        ]
        durations = [int(1000 / fps * w) for w in (frame_weights or [1.0] * len(images))]
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=durations, loop=0
        )


def _axis_reference(backend_figure, cell, channel: str) -> str:
    # The reference of a cell's axis, e.g. "x2".
    subplot = backend_figure.get_subplot(cell.row, cell.col)
    axis = subplot.xaxis if channel == "x" else subplot.yaxis
    return axis.plotly_name.replace("axis", "")


def _scene_aspect(cell) -> dict:
    # The scene settings of a 3D cell's box aspect.
    box = cell.box_aspect
    if box is None:
        return dict(aspectmode="auto")
    return dict(aspectmode="manual", aspectratio=dict(zip("xyz", box)))


def _axis(cell, channel: str, axis, text) -> dict:
    # Plotly axis settings of a position axis: title (written by `text`),
    # type, range, ticks.
    scale = axis.scale
    settings = dict(title=dict(text=text(cell.axis_title(channel))))
    settings["type"] = (
        "category" if scale.categorical else ("log" if scale.log else "linear")
    )
    if scale.range is not None and not scale.categorical:
        drawn = cell.axis_range(channel)
        settings["range"] = (
            [float(v) for v in scale.transform(np.asarray(drawn))] if drawn else None
        )
    ticks = cell.axis_ticks(channel)
    if ticks is not None:
        positions, labels = ticks
        settings["tickvals"] = list(positions)
        settings["ticktext"] = list(labels) if labels is not None else None
    return settings


def _title_name(cell) -> str:
    return f"cell-title-{cell.row}-{cell.col}"


def _colorbar_uid(scale) -> str:
    return f"colorbar-{id(scale)}"


def _axis_names(backend_figure, cell) -> list[str]:
    # Layout names of a cell's axes, e.g. ["xaxis2", "yaxis2"] or ["scene"];
    # none for a cell without position axes.
    if not cell.axes:
        return []
    subplot = backend_figure.get_subplot(cell.row, cell.col)
    if cell.dim == 3:
        return [subplot.plotly_name]
    return [subplot.xaxis.plotly_name, subplot.yaxis.plotly_name]


def _apply_theme(backend_figure, theme) -> None:
    # Figure-wide theme chrome: background, fonts, legend, axis lines; the
    # annotations are the cell titles, the only ones when it is applied.
    backend_figure.update_annotations(font=dict(size=theme.font_size_axes))
    backend_figure.update_layout(
        paper_bgcolor=theme.background_color,
        plot_bgcolor=theme.background_color,
        font=dict(
            family=theme.font_family, size=theme.font_size_labels, color=theme.text_color
        ),
        title_font=dict(size=theme.font_size_title, color=theme.text_color),
        showlegend=theme.show_legend,
        legend=_legend_style(theme),
    )
    axis = dict(
        showgrid=theme.show_grid,
        gridcolor=theme.grid_color,
        zeroline=theme.show_zero_line,
        zerolinecolor=theme.grid_color,
        showline=theme.show_axis_line,
        linecolor=theme.line_color,
        mirror=theme.axis_box,
        ticks="outside" if theme.show_axis_line else "",
        tickcolor=theme.line_color,
        tickfont=dict(size=theme.font_size_labels, color=theme.text_color),
        title_font=dict(size=theme.font_size_axes, color=theme.text_color),
    )
    backend_figure.update_xaxes(**axis)
    backend_figure.update_yaxes(**axis)
    scene_axis = dict(axis, showbackground=True, backgroundcolor=theme.background_color)
    backend_figure.update_scenes(
        bgcolor=theme.background_color,
        xaxis=scene_axis,
        yaxis=scene_axis,
        zaxis=scene_axis,
    )


def _colorbar_style(theme, title: str | None) -> dict:
    # The look of every colorbar: its title beside it, a fixed thickness,
    # no padding within its length (it is as tall as the plot area).
    return dict(
        title=dict(text=title, side="right", font=dict(size=theme.font_size_axes)),
        thickness=space.COLORBAR_THICKNESS,
        thicknessmode="pixels",
        tickfont=dict(size=theme.font_size_labels),
        ypad=0,
        outlinewidth=1 if theme.axis_box else 0,
        outlinecolor=theme.line_color,
    )


def _legend_name(index: int) -> str:
    # Plotly's name of the index-th legend: "legend", "legend2", ...
    return "legend" if index == 0 else f"legend{index + 1}"


def _legend_style(theme) -> dict:
    # The look of every legend.
    return dict(
        bordercolor=theme.line_color,
        borderwidth=1 if theme.legend_box else 0,
        bgcolor=theme.background_color,
        font=dict(size=theme.font_size_labels),
    )


def _in_notebook() -> bool:
    # Whether code runs in a Jupyter kernel (notebook, JupyterLab, VS Code).
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config
