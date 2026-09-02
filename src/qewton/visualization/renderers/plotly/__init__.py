from plotly import graph_objects as go
from plotly.subplots import make_subplots

from qewton.visualization.plots.spec import FacetSpec
from qewton.visualization.renderers.base import Renderer
from qewton.visualization.renderers.plotly.curve import LineArtist, PathArtist
from qewton.visualization.renderers.plotly.geometry import GeometryArtist, GeometryArtist2D
from qewton.visualization.renderers.plotly.graph import NodeLinkArtist
from qewton.visualization.renderers.plotly.grid import (
    HeatmapArtist,
    ImageArtist,
    ParametricSurfaceArtist,
    SurfaceArtist,
)
from qewton.visualization.renderers.plotly.mesh import FilledMeshArtist, SurfaceMeshArtist
from qewton.visualization.renderers.plotly.points import PointCloud2DArtist, PointCloud3DArtist
from qewton.visualization.renderers.plotly.table import ParallelCoordinatesArtist
from qewton.visualization.renderers.plotly.tabular import BarArtist, ScatterArtist
from qewton.visualization.renderers.plotly.vector import ArrowField2DArtist, ArrowField3DArtist


class PlotlyRenderer(Renderer):
    """Aggregation namespace: renderer.SurfaceMeshArtist.create(...) etc.

    The artists themselves live at module level (geometry.py, grid.py, mesh.py,
    vector.py) - this class only collects them so plots can reach them off the
    renderer instance without knowing which module each one lives in.
    """

    ImageArtist = ImageArtist
    HeatmapArtist = HeatmapArtist
    SurfaceArtist = SurfaceArtist
    ParametricSurfaceArtist = ParametricSurfaceArtist
    SurfaceMeshArtist = SurfaceMeshArtist
    FilledMeshArtist = FilledMeshArtist
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

    #: Plotly's make_subplots `specs` vocabulary for each embedding dimension
    #: Figure.cell_dimensions() reports. Plotly-specific translation lives
    #: here, not on Plot/Figure, which only know "2 or 3 spatial dimensions,
    #: or None for non-spatial (ParallelCoordinatesPlot, GraphPlot)".
    _SUBPLOT_TYPE_BY_DIM = {2: "xy", 3: "scene", None: "domain"}

    @staticmethod
    def setup(figure):
        n_rows, n_cols = figure.grid_shape()
        if n_rows == 1 and n_cols == 1:
            # Unchanged path for the common non-faceted case - avoids
            # make_subplots' extra layout machinery (subplot titles, spacing
            # defaults) changing anything about today's single-canvas output.
            fig = go.Figure()
        else:
            dims = figure.cell_dimensions(n_rows, n_cols)
            spans = figure.cell_spans(n_rows, n_cols)
            specs = [
                [
                    None
                    if spans[r][c] is None
                    else {
                        "type": PlotlyRenderer._SUBPLOT_TYPE_BY_DIM[dims[r][c]],
                        **(
                            {"rowspan": spans[r][c][0]}
                            if spans[r][c][0] > 1
                            else {}
                        ),
                        **(
                            {"colspan": spans[r][c][1]}
                            if spans[r][c][1] > 1
                            else {}
                        ),
                    }
                    for c in range(n_cols)
                ]
                for r in range(n_rows)
            ]
            titles = figure.cell_titles(n_rows, n_cols)
            fig = make_subplots(
                rows=n_rows, cols=n_cols, specs=specs,
                subplot_titles=titles if any(titles) else None,
            )
        # The whole-figure title is Figure.title alone, set once here - never
        # any individual Plot.title (each panel's own title already shows as
        # a subplot annotation via cell_titles()/make_subplots above). With
        # no Figure.title, no title is shown at all rather than falling back
        # to some arbitrary plot's title.
        fig.update_layout(uirevision=True, title=figure.title)
        PlotlyRenderer._apply_theme(fig, figure.theme)
        return fig

    @staticmethod
    def _apply_theme(fig, theme):
        """Figure-wide chrome (background, fonts, gridlines) applied once
        here rather than per-artist - every artist's own update_xaxes/
        update_scenes calls only ever ADD keys (title, type=log, ...) on top
        of what's set here, they never overwrite it, so call order with this
        doesn't matter. Per-trace/data-driven styling (marker colors,
        surface fills, ...) stays in the artists - this is only the parts of
        a theme that are true of the whole figure regardless of what's
        plotted in it."""
        fig.update_layout(
            paper_bgcolor=theme.background_color,
            plot_bgcolor=theme.background_color,
            font=dict(family=theme.font_family, size=theme.font_size_labels, color=theme.text_color),
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
        # Scene (3D) axes are a different schema - they additionally have
        # their own background plane (backgroundcolor/showbackground), which
        # 2D axes don't, so this can't just reuse axis_kwargs as-is.
        scene_axis_kwargs = dict(axis_kwargs, showbackground=True, backgroundcolor=theme.background_color)
        fig.update_scenes(
            bgcolor=theme.background_color,
            xaxis=scene_axis_kwargs,
            yaxis=scene_axis_kwargs,
            zaxis=scene_axis_kwargs,
        )

    @staticmethod
    def reconcile_y_axis_titles(figure, backend_figure, n_rows, n_cols):
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
        """Materializes one go.Frame per spec.values on top of the
        already-drawn backend_figure, by replaying Artist.update() at each
        state and snapshotting the resulting trace - no per-Artist animation
        code needed, since every artist already knows how to fully refresh
        its trace(s) for the plot's current control state via update().

        Animating a faceted plot (TimeSpec combined with FacetSpec on the
        same plot) is deliberately rejected rather than silently producing
        wrong frames: update() has no per-cell state to replay outside
        Figure._draw_plot()'s own row/col loop, so calling it here would use
        whichever facet state happened to be left over after the normal
        draw() pass for every cell alike.
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
                frame_data.append(backend_figure.data[artist.figure_idx].to_plotly_json())
                frame_traces.append(artist.figure_idx)
            frames.append(go.Frame(data=frame_data, traces=frame_traces, name=str(value)))
        spec.state = original_state
        for plot, artist in animated:
            artist.update(backend_figure, plot)  # leave the live traces at the initial state

        backend_figure.frames = frames
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
                                    frame=dict(duration=spec.duration, redraw=True),
                                    fromcurrent=True,
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
                            label=str(value),
                        )
                        for value in spec.values
                    ]
                )
            ],
        )
        return backend_figure

    @staticmethod
    def apply_variable_selector(figure, backend_figure, spec):
        """Adds one Plotly dropdown (updatemenus, method="restyle") letting
        an already-drawn static figure itself switch which variable is
        plotted - the static-export equivalent of DashApplication's
        dropdown widget, since a Dash app's own client/server round-trip
        isn't available once a figure is exported.

        Only meaningful for the static/non-Dash path (Figure.show()/
        save_html()/save_png()/save_svg() call this after draw(); Dash's
        own callback loop already handles VariableSpec entirely server-side
        and does not need this). Mirrors animate()'s replay-and-capture
        approach: temporarily set `spec` to each candidate, replay
        Artist.update() to compute what that trace would look like, and
        capture its Plotly attributes into one restyle button per candidate.
        """
        affected = [
            (plot, artist)
            for plot, cells in figure.artists.items()
            if spec in plot.variable_specs
            for artist in cells.values()
        ]
        if not affected:
            return backend_figure

        original_state = spec.state
        trace_indices = [artist.figure_idx for _, artist in affected]
        buttons = []
        for candidate in spec.candidates:
            spec.state = candidate
            per_key_values: dict = {}
            for plot, artist in affected:
                artist.update(backend_figure, plot)
                trace_json = backend_figure.data[artist.figure_idx].to_plotly_json()
                for key, value in trace_json.items():
                    if key in ("type", "uid"):
                        continue
                    per_key_values.setdefault(key, []).append(value)
            buttons.append(
                dict(label=candidate.name, method="restyle", args=[per_key_values, trace_indices])
            )
        spec.state = original_state
        for plot, artist in affected:
            artist.update(backend_figure, plot)  # leave the live traces at the initial state

        # animate() (TimeSpec) sets its own updatemenus wholesale, so this
        # only ever drops/replaces entries it added itself, tagged by name -
        # both can coexist on one figure.
        menu_name = f"variable_selector_{id(spec)}"
        kept = [m for m in backend_figure.layout.updatemenus if m.name != menu_name]
        backend_figure.update_layout(
            updatemenus=kept
            + [dict(name=menu_name, buttons=buttons, direction="down", showactive=True)]
        )
        return backend_figure

    @staticmethod
    def show(backend_figure):
        # Plotly no longer auto-loads MathJax (removed in plotly.js v2) - the
        # $...$-wrapped titles PlotSpec.math_name/axis_names_from_variable
        # produce render as literal dollar-sign text without this. Harmless
        # (silently ignored) for mimebundle-based renderers like "vscode",
        # whose own webview extension is responsible for LaTeX rendering.
        backend_figure.show(include_mathjax="cdn")

    @staticmethod
    def save_html(backend_figure, path):
        backend_figure.write_html(path, include_mathjax="cdn")

    @staticmethod
    def save_gif(backend_figure, path, fps=10):
        """Rasterizes each animation frame to a PNG (via Plotly's optional
        'kaleido' static-image backend) and assembles them into a looping
        GIF (via Pillow). Both are imported lazily, right here, rather than
        being hard dependencies of the visualization package - same
        convention as HDF5DataSet.from_file()'s h5py import: importing this
        module (or even Figure.draw()'s animation path) never requires
        either package, only actually calling save_gif() does."""
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

        # Static per-frame figures, not the live one: strip the play/pause/
        # slider UI (meaningless in a still image) and never mutate the
        # interactive backend_figure this call was handed.
        static_layout = go.Layout(backend_figure.layout)
        static_layout.updatemenus = ()
        static_layout.sliders = ()

        images = [
            Image.open(io.BytesIO(go.Figure(data=frame.data, layout=static_layout).to_image(format="png")))
            for frame in backend_figure.frames
        ]

        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )

    @staticmethod
    def save_image(backend_figure, path, format, **kwargs):
        """Rasterizes the figure's current state via Plotly's optional
        'kaleido' static-image backend - same lazy-import convention as
        save_gif()."""
        try:
            import kaleido  # noqa: F401 - presence check
        except ImportError as e:
            raise ImportError(
                "save_png()/save_svg() require the optional 'kaleido' package. "
                "Install via: pip install kaleido"
            ) from e
        backend_figure.write_image(path, format=format, **kwargs)
