"""Plotly renderer: :class:`PlotlyRenderer` and its artists."""

import dataclasses
import io

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

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
from qewton.visualization.renderers.base import Renderer
from qewton.visualization.renderers.plotly import camera
from qewton.visualization.renderers.plotly.graph import NodeLinkArtist
from qewton.visualization.scales import log_ticks
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
from qewton.visualization.renderers.plotly.traces import (
    color_range,
    colorbar_position,
    colorscale,
)

#: make_subplots() subplot type per number of position components.
_SUBPLOT_TYPES = {0: "domain", 2: "xy", 3: "scene"}


class PlotlyRenderer(Renderer):
    """Draws a Figure as a Plotly ``go.Figure``.

    Every cell is a subplot of ``make_subplots``: ``"xy"`` in 2D,
    ``"scene"`` in 3D, ``"domain"`` without position axes. Embedded controls become frames (TimeAxis),
    ``method="update"`` slider steps (SliderAxis) and dropdown buttons
    (Selector); each state sets only the trace properties and axes that
    differ between the states.

    Names of Variables (``$...$`` in the core) are rendered by MathJax:
    every text containing one is written as one math text, and
    :meth:`show` and :meth:`save_html` load MathJax. Slider labels and
    dropdown buttons, where Plotly renders no math, get Plotly's own text
    markup (italics, sub- and superscripts) instead, as does every text
    without MathJax.

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

    def __init__(self, theme: Theme = DEFAULT_THEME, mathjax: bool = True) -> None:
        super().__init__(theme)
        self.mathjax = mathjax

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
        the legend entry is written as :meth:`text`."""
        super().draw(key, dataclasses.replace(result, label=self.text(result.label)), cell, visible)

    # --- figure, axes, colorbars --------------------------------------------

    def create_backend_figure(self, figure) -> go.Figure:
        """See :meth:`Renderer.create_backend_figure
        <qewton.visualization.renderers.base.Renderer.create_backend_figure>`."""
        n_rows, n_cols = figure.grid_shape
        specs, titles = [], []
        for row in range(1, n_rows + 1):
            specs.append([])
            for col in range(1, n_cols + 1):
                cell = figure.cell_at(row, col)
                specs[-1].append({"type": _SUBPLOT_TYPES[cell.dim] if cell else "xy"})
                titles.append(self.text(cell.title if cell else None) or "")
        backend_figure = make_subplots(
            rows=n_rows,
            cols=n_cols,
            specs=specs,
            subplot_titles=titles if any(titles) else None,
        )
        backend_figure.update_layout(uirevision=True, title=self.text(figure.title))
        _apply_theme(backend_figure, self.theme)
        return backend_figure

    def draw_axes(self, cell) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_axes`."""
        if not cell.axes:
            return  # e.g. parallel coordinates, which lay out their own axes
        backend_figure = self.backend_figure
        axes = {
            channel: _axis(cell, channel, axis, self.text)
            for channel, axis in cell.axes.items()
        }
        if cell.dim == 3:
            backend_figure.update_scenes(
                row=cell.row,
                col=cell.col,
                **{f"{c}axis": a for c, a in axes.items()},
                **_scene_aspect(cell),
            )
            return
        matched = False
        for channel in ("x", "y"):
            first = self._first_cell_with(cell.axes[channel], channel)
            if first is not cell:
                # Zooming one cell zooms every cell sharing the axis.
                axes[channel]["matches"] = _axis_reference(backend_figure, first, channel)
                matched = True
        backend_figure.update_xaxes(row=cell.row, col=cell.col, **axes["x"])
        backend_figure.update_yaxes(row=cell.row, col=cell.col, **axes["y"])
        # Plotly forbids an axis in both a matches and a scaleanchor
        # constraint: a matched cell takes its scale from the first one.
        if {"x", "y"} <= set(cell.scaled_channels) and not matched:
            backend_figure.update_yaxes(
                row=cell.row,
                col=cell.col,
                scaleanchor=_axis_reference(backend_figure, cell, "x"),
                scaleratio=1,
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

    def draw_colorbar(self, scale, title, cells, visible: bool = True) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.draw_colorbar`.

        The colorbar is an invisible one-point trace carrying the
        continuous colorbar, identified by the scale's ``uid``.
        """
        backend_figure = self.backend_figure
        cell = max(cells, key=lambda c: (c.col, -c.row))
        lo, hi = color_range(scale)
        colorbar = dict(
            colorbar_position(backend_figure, cell.row, cell.col),
            title=dict(text=self.text(title)),
        )
        if scale.log:
            tickvals, ticktext = log_ticks(lo, hi)
            colorbar.update(tickvals=tickvals, ticktext=ticktext)
        marker = dict(
            color=[lo],
            colorscale=colorscale(scale, self.theme),
            cmin=lo,
            cmax=hi,
            showscale=True,
            colorbar=colorbar,
        )
        index = self._colorbar_index(scale)
        if index is not None:
            backend_figure.data[index].update(marker=marker, visible=visible)
            return
        trace_type = go.Scatter3d if cell.dim == 3 else go.Scatter
        coordinates = {c: [None] for c in ("x", "y", "z")[: cell.dim]}
        carrier = trace_type(
            **coordinates,
            mode="markers",
            marker=marker,
            uid=_colorbar_uid(scale),
            visible=visible,
            showlegend=False,
            hoverinfo="skip",
        )
        backend_figure.add_trace(carrier, row=cell.row, col=cell.col)

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
        traces = {i: _flatten(data[i].to_plotly_json()) for i in indices}
        layout = self.backend_figure.layout.to_plotly_json()
        axes = {}
        for cell in cells:
            for name in _axis_names(self.backend_figure, cell):
                axes.update(_flatten(layout.get(name, {}), prefix=name))
        return traces, axes

    def animate(self, time_axis, snapshots) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.animate`.

        Frames update only the trace properties and axes that differ
        between the states. With non-uniform frame weights each frame gets
        its own duration and playback always starts at the first frame.
        """
        backend_figure = self.backend_figure
        states = time_axis.states
        indices, keys, layout_keys = _changes(snapshots)
        types = {i: backend_figure.data[i].type for i in indices}
        backend_figure.frames = [
            go.Frame(
                data=[
                    dict(_nest({k: traces[i].get(k) for k in keys[i]}), type=types[i])
                    for i in indices
                ],
                traces=indices,
                layout=_nest({k: axes.get(k) for k in layout_keys}),
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
            for state, args in zip(slider.states, _update_args(snapshots))
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
                    y=-0.15 * len(existing),
                    yanchor="top",
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
            for state, args in zip(selector.states, _update_args(snapshots))
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
                    x=1.01,
                    xanchor="left",
                    y=1.0 - 0.12 * len(existing),
                    yanchor="top",
                )
            ]
        )

    # --- outputs -------------------------------------------------------------

    def show(self) -> None:
        """See :meth:`~qewton.visualization.renderers.base.Renderer.show`.

        With MathJax, a notebook gets the figure as HTML that loads
        plotly.js and MathJax (from a CDN) before drawing, since notebook
        frontends do not load MathJax for Plotly's own output; elsewhere
        Plotly's renderer is given MathJax. An animation starts with its
        play button, not on load. Linked 3D cells move their cameras
        together, except in Plotly's own notebook output (without MathJax),
        which runs no scripts.
        """
        groups = self.linked_scenes()
        script = dict(post_script=camera.page_script(groups)) if groups else {}
        if not self.mathjax:
            self.backend_figure.show(auto_play=False, **script)
        elif _in_notebook():
            from IPython.display import HTML, display

            display(HTML(self.html(full_html=False)))
        else:
            self.backend_figure.show(include_mathjax="cdn", auto_play=False, **script)

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
    # Aspect of a 3D cell: the data's for the channels drawn to one scale;
    # a channel of its own (e.g. a height over a plane) as tall as the
    # widest scaled one.
    scaled = cell.scaled_channels
    if len(scaled) == 3:
        return dict(aspectmode="data")
    spans = {}
    for channel, axis in cell.axes.items():
        if axis.scale.range is None:
            return dict(aspectmode="auto")
        lo, hi = axis.scale.transform(np.asarray(axis.scale.range))
        spans[channel] = max(float(hi - lo), 1e-12)
    if len(scaled) < 2:
        return dict(aspectmode="cube")
    widest = max(spans[c] for c in scaled)
    ratio = {c: spans[c] / widest if c in scaled else 1.0 for c in spans}
    return dict(aspectmode="manual", aspectratio=ratio)


def _axis(cell, channel: str, axis, text) -> dict:
    # Plotly axis settings of a position axis: title (written by `text`),
    # type, range, ticks.
    scale = axis.scale
    settings = dict(title=dict(text=text(cell.axis_title(channel))))
    settings["type"] = (
        "category" if scale.categorical else ("log" if scale.log else "linear")
    )
    if scale.range is not None and not scale.categorical:
        lo, hi = scale.transform(np.asarray(scale.range))
        settings["range"] = [float(lo), float(hi)] if hi > lo else None
    ticks = cell.axis_ticks(channel)
    if ticks is not None:
        positions, labels = ticks
        settings["tickvals"] = list(positions)
        settings["ticktext"] = list(labels) if labels is not None else None
    return settings


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


def _flatten(properties: dict, prefix: str = "") -> dict:
    # Nested properties as {"marker.color": ...}; the binding of a trace to
    # its subplot is left out, since it never changes.
    flat = {}
    for key, value in properties.items():
        if not prefix and key in ("type", "uid", "xaxis", "yaxis", "scene"):
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _nest(flat: dict) -> dict:
    # Inverse of _flatten.
    nested: dict = {}
    for path, value in flat.items():
        *parents, leaf = path.split(".")
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return nested


def _same(a, b) -> bool:
    encode = PlotlyJSONEncoder().encode
    return encode(a) == encode(b)


def _varying(snapshots: list[dict]) -> list[str]:
    # The keys whose values differ between the flat snapshots.
    keys = list(dict.fromkeys(k for s in snapshots for k in s))
    first = snapshots[0]
    return [
        k for k in keys if any(not _same(s.get(k), first.get(k)) for s in snapshots[1:])
    ]


def _changes(snapshots) -> tuple[list[int], dict[int, list[str]], list[str]]:
    # The traces with properties differing between states, those
    # properties per trace, and the differing axis layout keys.
    trace_keys = {
        i: _varying([traces[i] for traces, _ in snapshots]) for i in snapshots[0][0]
    }
    indices = [i for i, keys in trace_keys.items() if keys]
    layout_keys = _varying([axes for _, axes in snapshots])
    return indices, {i: trace_keys[i] for i in indices}, layout_keys


def _update_args(snapshots) -> list[list]:
    # Arguments of a method="update" call per state, setting the properties
    # that differ between states (one value per changed trace and key).
    indices, keys, layout_keys = _changes(snapshots)
    union = list(dict.fromkeys(k for i in indices for k in keys[i]))
    return [
        [
            {k: [traces[i].get(k) for i in indices] for k in union},
            {k: axes.get(k) for k in layout_keys},
            indices,
        ]
        for traces, axes in snapshots
    ]


def _apply_theme(backend_figure, theme) -> None:
    # Figure-wide theme chrome: background, fonts, legend, axis lines.
    backend_figure.update_layout(
        paper_bgcolor=theme.background_color,
        plot_bgcolor=theme.background_color,
        font=dict(
            family=theme.font_family, size=theme.font_size_labels, color=theme.text_color
        ),
        title_font=dict(size=theme.font_size_title, color=theme.text_color),
        showlegend=theme.show_legend,
        legend=dict(
            bordercolor=theme.line_color,
            borderwidth=1 if theme.show_axis_line else 0,
            bgcolor=theme.background_color,
        ),
    )
    axis = dict(
        showgrid=theme.show_grid,
        gridcolor=theme.grid_color,
        zerolinecolor=theme.grid_color,
        showline=theme.show_axis_line,
        linecolor=theme.line_color,
        mirror=theme.axis_box,
        ticks="outside" if theme.show_axis_line else "",
        tickcolor=theme.line_color,
        tickfont=dict(size=theme.font_size_axes, color=theme.text_color),
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


def _in_notebook() -> bool:
    # Whether code runs in a Jupyter kernel (notebook, JupyterLab, VS Code).
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config
