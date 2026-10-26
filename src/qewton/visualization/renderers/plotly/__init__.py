"""Plotly renderer: :class:`PlotlyRenderer` and its artists."""

import warnings

from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qewton.visualization.plots.spec import FacetSpec, SelectorSpec


def _axis_layout_updates(backend_figure, trace) -> dict:
    # Current title/type of the x/y axes `trace` is bound to, as a flat
    # relayout dict ({"xaxis3.title.text": ..., "xaxis3.type": ...}) for the
    # layout half of a method="update" button. The axis is read from the
    # trace's own "x"/"x2"/... binding, so no row/col is needed.
    updates = {}
    if not hasattr(trace, "xaxis"):
        # A 3D trace (Cone, Mesh3d, Surface, Scatter3d, ...) lives in a
        # `scene`, whose axes no Artist.update() retitles - nothing to capture.
        return updates
    for ref, prefix in ((trace.xaxis, "xaxis"), (trace.yaxis, "yaxis")):
        # A trace outside a make_subplots() grid has xaxis/yaxis None and is
        # bound to the implicit "xaxis"/"yaxis".
        axis_name = prefix if ref is None else f"{prefix}{ref[1:]}"
        axis = getattr(backend_figure.layout, axis_name, None)
        if axis is None:
            continue
        if axis.title is not None:
            updates[f"{axis_name}.title.text"] = axis.title.text
        updates[f"{axis_name}.type"] = axis.type
    return updates


from qewton.visualization.renderers.base import Renderer
from qewton.visualization.renderers.plotly.curve import LineArtist, PathArtist
from qewton.visualization.renderers.plotly.geometry import (
    GeometryArtist,
    GeometryArtist2D,
)
from qewton.visualization.renderers.plotly.graph import NodeLinkArtist
from qewton.visualization.renderers.plotly.grid import (
    HeatmapArtist,
    ParametricHeatmapArtist,
    ImageArtist,
    ParametricSurfaceArtist,
    SurfaceArtist,
)
from qewton.visualization.renderers.plotly.mesh import (
    CurveFieldArtist,
    FilledMeshArtist,
    SurfaceMeshArtist,
)
from qewton.visualization.renderers.plotly.points import (
    PointCloud2DArtist,
    PointCloud3DArtist,
)
from qewton.visualization.renderers.plotly.table import ParallelCoordinatesArtist
from qewton.visualization.renderers.plotly.tabular import BarArtist, ScatterArtist
from qewton.visualization.renderers.plotly.vector import (
    ArrowField2DArtist,
    ArrowField3DArtist,
)


class PlotlyRenderer(Renderer):
    """Renderer that draws a Figure as a Plotly ``go.Figure``.

    Besides implementing the
    :class:`~qewton.visualization.renderers.base.Renderer` interface, the
    class collects every Plotly artist as a class attribute (e.g.
    ``renderer.SurfaceMeshArtist``), so plots reach them off the renderer
    instance in ``create_artist()`` without importing the artist modules.

    Static controls (:meth:`animate`, :meth:`apply_slider`,
    :meth:`apply_selector`) are baked by replaying
    :meth:`~qewton.visualization.renderers.base.Artist.update` for every
    state and capturing the traces listed in each artist's
    ``trace_indices``.
    """

    ImageArtist = ImageArtist
    HeatmapArtist = HeatmapArtist
    ParametricHeatmapArtist = ParametricHeatmapArtist
    SurfaceArtist = SurfaceArtist
    ParametricSurfaceArtist = ParametricSurfaceArtist
    SurfaceMeshArtist = SurfaceMeshArtist
    FilledMeshArtist = FilledMeshArtist
    CurveFieldArtist = CurveFieldArtist
    GeometryArtist = GeometryArtist
    GeometryArtist2D = GeometryArtist2D
    LineArtist = LineArtist
    PathArtist = PathArtist
    ScatterArtist = ScatterArtist
    BarArtist = BarArtist
    PointCloud2DArtist = PointCloud2DArtist
    PointCloud3DArtist = PointCloud3DArtist
    ParallelCoordinatesArtist = ParallelCoordinatesArtist
    ArrowField2DArtist = ArrowField2DArtist
    ArrowField3DArtist = ArrowField3DArtist
    NodeLinkArtist = NodeLinkArtist

    #: make_subplots() subplot type per embedding dimension reported by
    #: Figure.cell_dimensions() (None: non-spatial plots).
    _SUBPLOT_TYPE_BY_DIM = {2: "xy", 3: "scene", None: "domain"}

    @staticmethod
    def setup(figure):
        """Build an empty, themed ``go.Figure`` for `figure`.

        A 1x1 grid gets a plain ``go.Figure``; larger grids use
        ``make_subplots()`` with one subplot type per cell (``"xy"``,
        ``"scene"`` or ``"domain"``), row/column spans and cell titles. The
        figure title is ``figure.title`` only, never a plot title.

        Args:
            figure (Figure): The :class:`~qewton.visualization.figure.Figure`
                to build for.

        Returns:
            plotly.graph_objects.Figure: The empty figure.
        """
        n_rows, n_cols = figure.grid_shape()
        if n_rows == 1 and n_cols == 1:
            # No make_subplots() for a single cell: avoids its extra layout
            # (subplot titles, spacing defaults).
            fig = go.Figure()
        else:
            dims = figure.cell_dimensions(n_rows, n_cols)
            spans = figure.cell_spans(n_rows, n_cols)
            specs = [
                [
                    (
                        None
                        if spans[r][c] is None
                        else {
                            "type": PlotlyRenderer._SUBPLOT_TYPE_BY_DIM[dims[r][c]],
                            **({"rowspan": spans[r][c][0]} if spans[r][c][0] > 1 else {}),
                            **({"colspan": spans[r][c][1]} if spans[r][c][1] > 1 else {}),
                        }
                    )
                    for c in range(n_cols)
                ]
                for r in range(n_rows)
            ]
            titles = figure.cell_titles(n_rows, n_cols)
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                specs=specs,
                subplot_titles=titles if any(titles) else None,
            )
        # Plot titles are shown as subplot titles (cell_titles() above).
        fig.update_layout(uirevision=True, title=figure.title)
        PlotlyRenderer._apply_theme(fig, figure.theme)
        return fig

    @staticmethod
    def _apply_theme(fig, theme):
        # Figure-wide theme chrome (background, fonts, grid, axis lines).
        # Artists only add other axis keys (title, type) on top, so the call
        # order does not matter. Data-driven styling stays in the artists.
        fig.update_layout(
            paper_bgcolor=theme.background_color,
            plot_bgcolor=theme.background_color,
            font=dict(
                family=theme.font_family,
                size=theme.font_size_labels,
                color=theme.text_color,
            ),
            title_font=dict(size=theme.font_size_title, color=theme.text_color),
            showlegend=theme.show_legend,
            legend=dict(
                bordercolor=theme.line_color,
                borderwidth=1 if theme.show_axis_line else 0,
                bgcolor=theme.background_color,
            ),
        )
        axis_kwargs = dict(
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
        fig.update_xaxes(**axis_kwargs)
        fig.update_yaxes(**axis_kwargs)
        # Scene (3D) axes additionally have a background plane.
        scene_axis_kwargs = dict(
            axis_kwargs, showbackground=True, backgroundcolor=theme.background_color
        )
        fig.update_scenes(
            bgcolor=theme.background_color,
            xaxis=scene_axis_kwargs,
            yaxis=scene_axis_kwargs,
            zaxis=scene_axis_kwargs,
        )

    @staticmethod
    def reconcile_y_axis_titles(figure, backend_figure, n_rows, n_cols):
        """Apply :meth:`~qewton.visualization.figure.Figure.cell_y_titles`.

        Cells whose title is None keep the title their artists set.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (plotly.graph_objects.Figure): The drawn figure,
                modified in place.
            n_rows (int): Number of grid rows.
            n_cols (int): Number of grid columns.
        """
        titles = figure.cell_y_titles(n_rows, n_cols)
        is_grid = (n_rows, n_cols) != (1, 1)
        for idx, title in enumerate(titles):
            if title is None:
                continue
            if is_grid:
                row, col = divmod(idx, n_cols)
                backend_figure.update_yaxes(title=title, row=row + 1, col=col + 1)
            else:
                backend_figure.update_yaxes(title=title)

    @staticmethod
    def animate(figure, backend_figure, spec):
        """Add one ``go.Frame`` per value of `spec`, plus play/pause/slider UI.

        Each frame is captured by setting ``spec.state`` to the value,
        replaying :meth:`~qewton.visualization.renderers.base.Artist.update`
        for every artist whose plot is controlled by `spec`, and
        snapshotting the traces in its ``trace_indices``. Afterwards `spec`
        and the live traces are back at their initial state. With
        non-uniform frame weights
        (:meth:`~qewton.visualization.plots.spec.TimeSpec.frame_weights`),
        each frame gets its own duration and playback always starts from
        the first frame.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (plotly.graph_objects.Figure): The drawn figure,
                modified in place.
            spec (TimeSpec): The time control to animate.

        Returns:
            plotly.graph_objects.Figure: `backend_figure`.

        Raises:
            NotImplementedError: If a plot controlled by `spec` also has a
                :class:`~qewton.visualization.plots.spec.FacetSpec`.
        """
        animated: list[tuple] = []
        for plot, cells in figure.artists.items():
            if spec not in plot.controls:
                continue
            if any(isinstance(c, FacetSpec) for c in plot.controls):
                raise NotImplementedError(
                    f"{type(plot).__name__} combines a TimeSpec with a FacetSpec - "
                    "animating a faceted plot is not supported yet."
                )
            animated.extend((plot, artist) for artist in cells.values())

        if not animated:
            return backend_figure

        original_state = spec.state
        frames = []
        for value in spec.values:
            spec.state = value
            frame_data, frame_traces = [], []
            for plot, artist in animated:
                artist.update(backend_figure, plot)
                for idx in artist.trace_indices:
                    frame_data.append(backend_figure.data[idx].to_plotly_json())
                    frame_traces.append(idx)
            frames.append(go.Frame(data=frame_data, traces=frame_traces, name=str(value)))
        spec.state = original_state
        for plot, artist in animated:
            artist.update(
                backend_figure, plot
            )  # leave the live traces at the initial state

        backend_figure.frames = frames
        # Plotly indexes a per-frame duration list by position in the
        # played sequence, which "fromcurrent" shortens, so non-uniform
        # durations disable "fromcurrent".
        weights = spec.frame_weights()
        uniform = all(w == weights[0] for w in weights)
        play_frame = (
            dict(duration=spec.duration, redraw=True)
            if uniform
            else [dict(duration=spec.duration * w, redraw=True) for w in weights]
        )
        backend_figure.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=play_frame,
                                    fromcurrent=uniform,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    currentvalue=dict(prefix=f"{spec.name} = "),
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [str(value)],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                ),
                            ],
                            label=spec.label(value),
                        )
                        for value in spec.values
                    ],
                )
            ],
        )
        return backend_figure

    @staticmethod
    def _capture_states(backend_figure, affected, spec, states) -> list[list]:
        """Args of one method="update" call per state of `spec`.

        Replays Artist.update() for each state and captures the affected
        traces (and the titles/types of their 2D axes). Restores `spec` and
        the live traces afterwards.
        """
        original_state = spec.state
        trace_indices = [idx for _, artist in affected for idx in artist.trace_indices]
        captured = []
        try:
            for state in states:
                spec.state = state
                trace_jsons = []
                layout_updates: dict = {}
                for plot, artist in affected:
                    artist.update(backend_figure, plot)
                    for idx in artist.trace_indices:
                        trace = backend_figure.data[idx]
                        trace_jsons.append(trace.to_plotly_json())
                        # update() may retitle/retype the trace's axes,
                        # which live in the layout, not the trace.
                        layout_updates.update(_axis_layout_updates(backend_figure, trace))
                # One value per trace for every key, in trace_indices order;
                # traces can differ in type (e.g. arrow bins plus a colorbar
                # carrier), so a key a trace lacks gets None.
                keys = {key for j in trace_jsons for key in j} - {"type", "uid"}
                per_key_values = {key: [j.get(key) for j in trace_jsons] for key in keys}
                captured.append([per_key_values, layout_updates, trace_indices])
        finally:
            spec.state = original_state
            for plot, artist in affected:
                artist.update(backend_figure, plot)  # back to the initial state
        return captured

    @staticmethod
    def apply_slider(figure, backend_figure, spec):
        """Add a Plotly slider stepping through every state of `spec`.

        Every state from ``spec.minimum`` to ``spec.maximum`` (step
        ``spec.step``) is evaluated up front and embedded as one
        method="update" slider step, so the figure grows with the number of
        steps. No frames are used, so the slider coexists with a TimeSpec
        slider from :meth:`animate`; it is placed below existing sliders.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (plotly.graph_objects.Figure): The drawn figure,
                modified in place.
            spec (SliderSpec): The slider to bake.

        Returns:
            plotly.graph_objects.Figure: `backend_figure`, unchanged if no plot
                is controlled by `spec` or if any such plot is faceted (then
                with a warning).
        """
        affected = [
            (plot, artist)
            for plot, cells in figure.artists.items()
            if spec in plot.controls
            for artist in cells.values()
        ]
        if not affected:
            return backend_figure
        if any(isinstance(c, FacetSpec) for plot, _ in affected for c in plot.controls):
            warnings.warn(
                f"SliderSpec '{spec.name}' controls a faceted plot - its slider "
                "can't be baked into a static figure, so only the initial state "
                "is shown. Use DashApplication for an interactive slider.",
                stacklevel=4,
            )
            return backend_figure

        states = list(range(spec.minimum, spec.maximum + 1, spec.step))
        captured = PlotlyRenderer._capture_states(backend_figure, affected, spec, states)
        steps = [
            dict(label=spec.label(state), method="update", args=args)
            for state, args in zip(states, captured)
        ]
        active = states.index(spec.state) if spec.state in states else 0

        # Below any slider already present (e.g. from animate()).
        existing = list(backend_figure.layout.sliders)
        backend_figure.update_layout(
            sliders=existing
            + [
                dict(
                    name=f"slider_{id(spec)}",
                    active=active,
                    steps=steps,
                    currentvalue=dict(prefix=f"{spec.name} = "),
                    y=-0.15 * len(existing),
                    yanchor="top",
                )
            ]
        )
        return backend_figure

    @staticmethod
    def apply_selector(figure, backend_figure, spec):
        """Add a Plotly dropdown switching the candidate of `spec`.

        Every candidate is evaluated up front (same replay-and-capture as
        :meth:`animate`) and embedded as one method="update" button. Menus
        of other selectors are kept and all selector menus are stacked
        vertically right of the plot; non-selector menus (e.g. the
        animation buttons) are kept unchanged.

        Args:
            figure (Figure): The figure that was drawn.
            backend_figure (plotly.graph_objects.Figure): The drawn figure,
                modified in place.
            spec (SelectorSpec): The selector to bake.

        Returns:
            plotly.graph_objects.Figure: `backend_figure`.

        Raises:
            NotImplementedError: If a plot controlled by `spec` has more than
                one :class:`~qewton.visualization.plots.spec.SelectorSpec`.
        """
        affected = [
            (plot, artist)
            for plot, cells in figure.artists.items()
            if spec in plot.selector_specs
            for artist in cells.values()
        ]
        if not affected:
            return backend_figure

        for plot, _ in affected:
            if len(plot.selector_specs) > 1:
                raise NotImplementedError(
                    f"{type(plot).__name__} has {len(plot.selector_specs)} "
                    "SelectorSpecs feeding into one evaluate() call, so their "
                    "dropdowns can't be baked as independent static buttons - "
                    "each one's choices would silently ignore the others' "
                    "current selection except at the combination the figure "
                    "started at. Figure.show()/save_html()/save_png()/"
                    "save_svg() only support a plot with at most one "
                    "SelectorSpec; use "
                    "qewton.visualization.applications.dash_app."
                    "DashApplication for a plot with more than one, which "
                    "resolves them all correctly via a real callback."
                )

        buttons = [
            dict(
                label=SelectorSpec.candidate_name(candidate),
                method="update",
                args=args,
            )
            for candidate, args in zip(
                spec.candidates,
                PlotlyRenderer._capture_states(
                    backend_figure, affected, spec, spec.candidates
                ),
            )
        ]

        # Selector menus are tagged by name, so only those are replaced.
        menu_name = f"selector_{id(spec)}"

        # Keep any non-selector menus (e.g. animation controls) and
        # rebuild selector menus with explicit positions so they do
        # not overlap.
        existing_selectors = []
        kept = []
        for menu in backend_figure.layout.updatemenus:
            menu_name_value = getattr(menu, "name", None)
            if menu_name_value is None and isinstance(menu, dict):
                menu_name_value = menu.get("name")

            to_plotly_json = getattr(menu, "to_plotly_json", None)
            if callable(to_plotly_json):
                menu_dict = to_plotly_json()
            elif isinstance(menu, dict):
                menu_dict = dict(menu)
            else:
                menu_dict = {}
            if isinstance(menu_name_value, str) and menu_name_value.startswith(
                "selector_"
            ):
                existing_selectors.append(menu_dict)
            else:
                kept.append(menu_dict)

        # Replace/create this selector's menu content before laying all
        # variable selectors out in a clean vertical stack.
        existing_selectors = [m for m in existing_selectors if m.get("name") != menu_name]
        existing_selectors.append(
            dict(name=menu_name, buttons=buttons, direction="down", showactive=True)
        )

        x = 1.01
        y_start = 1.0
        y_step = 0.12
        laid_out_selectors = []
        for idx, menu in enumerate(existing_selectors):
            laid_out_selectors.append(
                {
                    **menu,
                    "x": x,
                    "xanchor": "left",
                    "y": y_start - idx * y_step,
                    "yanchor": "top",
                }
            )

        backend_figure.update_layout(updatemenus=kept + laid_out_selectors)
        return backend_figure

    @staticmethod
    def _pinned_size_config(backend_figure) -> dict | None:
        """``dict(responsive=False)`` if width and height are set, else None.

        Plotly's responsive mode would otherwise stretch a pinned-size
        figure (e.g. from NodeLinkArtist) to its container.
        """
        if backend_figure.layout.width is not None and backend_figure.layout.height is not None:
            return dict(responsive=False)
        return None

    @staticmethod
    def show(backend_figure):
        """Display the figure with Plotly's default renderer.

        MathJax is loaded from a CDN for the ``$...$`` titles. A figure with
        explicit width and height is shown non-responsive, at that size.

        Args:
            backend_figure (plotly.graph_objects.Figure): The drawn figure.
        """
        # plotly.js does not load MathJax itself; mimebundle-based renderers
        # (e.g. "vscode") ignore include_mathjax.
        config = PlotlyRenderer._pinned_size_config(backend_figure)
        backend_figure.show(include_mathjax="cdn", config=config)

    @staticmethod
    def save_html(backend_figure, path):
        """Write the figure to `path` as an interactive HTML file.

        Same MathJax and sizing behavior as :meth:`show`.

        Args:
            backend_figure (plotly.graph_objects.Figure): The drawn figure.
            path (str or path-like): Output file path.
        """
        config = PlotlyRenderer._pinned_size_config(backend_figure)
        backend_figure.write_html(path, include_mathjax="cdn", config=config)

    @staticmethod
    def save_gif(backend_figure, path, fps=10, frame_weights=None):
        """Rasterize each animation frame and write a looping GIF.

        Frames are rendered to PNG with the optional ``kaleido`` package
        and assembled with the optional ``Pillow`` package, both imported
        lazily. The play/pause buttons and sliders are removed from the
        rendered frames; `backend_figure` itself is not modified.

        Args:
            backend_figure (plotly.graph_objects.Figure): The drawn figure,
                with animation frames.
            path (str or path-like): Output file path.
            fps (float, optional): Frames per second for equally weighted
                frames.
            frame_weights (list of float, optional): Relative display duration
                per frame.

        Raises:
            ImportError: If ``kaleido`` or ``Pillow`` is not installed.
            ValueError: If `backend_figure` has no animation frames.
        """
        try:
            import kaleido  # noqa: F401 - presence check; Figure.to_image() uses it internally
        except ImportError as e:
            raise ImportError(
                "save_gif() requires the optional 'kaleido' package. "
                "Install via: pip install kaleido"
            ) from e
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "save_gif() requires the optional 'Pillow' package. "
                "Install via: pip install Pillow"
            ) from e
        import io

        if not backend_figure.frames:
            raise ValueError(
                "backend_figure has no animation frames - draw a Figure with "
                "a TimeSpec control before calling save_gif()."
            )

        # Separate per-frame figures without play/pause/slider UI, so the
        # live backend_figure stays untouched.
        static_layout = go.Layout(backend_figure.layout)
        static_layout.updatemenus = ()
        static_layout.sliders = ()

        images = [
            Image.open(
                io.BytesIO(
                    go.Figure(data=frame.data, layout=static_layout).to_image(
                        format="png"
                    )
                )
            )
            for frame in backend_figure.frames
        ]

        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=[
                int(1000 / fps * w)
                for w in (frame_weights or [1.0] * len(images))
            ],
            loop=0,
        )

    @staticmethod
    def save_image(backend_figure, path, format, **kwargs):
        """Write the figure's current state as a static image.

        Uses the optional ``kaleido`` package, imported lazily.

        Args:
            backend_figure (plotly.graph_objects.Figure): The drawn figure.
            path (str or path-like): Output file path.
            format (str): Image format, e.g. ``"png"`` or ``"svg"``.
            **kwargs: Passed on to ``write_image()``.

        Raises:
            ImportError: If ``kaleido`` is not installed.
        """
        try:
            import kaleido  # noqa: F401 - presence check
        except ImportError as e:
            raise ImportError(
                "save_png()/save_svg() require the optional 'kaleido' package. "
                "Install via: pip install kaleido"
            ) from e
        backend_figure.write_image(path, format=format, **kwargs)
